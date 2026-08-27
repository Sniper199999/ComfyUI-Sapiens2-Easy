from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from .constants import ARCH_SPECS
from .progress import NodeProgress
from .types import Sapiens2Model


_MODEL_CACHE: dict[tuple, Sapiens2Model] = {}

_DENSE_CONFIG_TEMPLATES = {
    "segmentation": (
        "sapiens/dense/configs/seg/shutterstock_goliath/"
        "{arch}_seg_shutterstock_goliath-1024x768.py"
    ),
    "normal": (
        "sapiens/dense/configs/normal/metasim_render_people/"
        "{arch}_normal_metasim_render_people-1024x768.py"
    ),
    "pointmap": (
        "sapiens/dense/configs/pointmap/render_people/"
        "{arch}_pointmap_render_people-1024x768.py"
    ),
}


def _candidate_repo_paths(repo_path: str) -> List[Path]:
    custom_node_root = Path(__file__).resolve().parents[1]
    candidates = []
    if repo_path.strip():
        candidates.append(Path(os.path.expanduser(os.path.expandvars(repo_path))))
    env_path = os.environ.get("SAPIENS2_REPO", "")
    if env_path:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(env_path))))
    candidates.extend(
        [
            custom_node_root / "vendor" / "sapiens2",
            Path.home() / "sapiens2",
            Path.home() / "repo" / "github.com" / "facebookresearch" / "sapiens2",
        ]
    )
    return candidates


def _ensure_sapiens_importable(repo_path: str) -> None:
    searched = _candidate_repo_paths(repo_path)
    for candidate in searched:
        if (candidate / "sapiens").is_dir():
            candidate_path = str(candidate)
            if candidate_path not in sys.path:
                sys.path.insert(0, candidate_path)
            break

    try:
        importlib.import_module("sapiens.backbones")
        importlib.import_module("sapiens.dense")
    except Exception as exc:
        searched_text = ", ".join(str(path) for path in searched)
        raise RuntimeError(
            "Could not import the official Sapiens2 code. Check that the repo path is "
            "correct and that its Python dependencies can import in this ComfyUI venv. "
            "You can clone https://github.com/facebookresearch/sapiens2 into this custom "
            "node's vendor/sapiens2 directory, set SAPIENS2_REPO, or pass sapiens_repo_path. "
            f"Searched: {searched_text}. Original import error: {exc}"
        ) from exc


def get_sapiens_repo_path(repo_path: str) -> Path:
    for candidate in _candidate_repo_paths(repo_path):
        if (candidate / "sapiens").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate the official Sapiens2 repo. Clone "
        "https://github.com/facebookresearch/sapiens2 into this custom node's "
        "vendor/sapiens2 directory, set SAPIENS2_REPO, or pass sapiens_repo_path."
    )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(device)


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "fp32" or device.type in ("cpu", "mps"):
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class FP8Linear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = torch.nn.Parameter(
            linear.weight.detach().to(torch.float8_e4m3fn), requires_grad=False
        )
        if linear.bias is not None:
            self.bias = torch.nn.Parameter(linear.bias.detach(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight.to(x.dtype), self.bias)


def _apply_fp8_weights(model: torch.nn.Module) -> torch.nn.Module:
    """
    Recursively replaces nn.Linear modules with FP8Linear (weights quantized to float8_e4m3fn).
    """
    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.Linear) and child.weight.is_floating_point():
            setattr(model, name, FP8Linear(child))
        else:
            _apply_fp8_weights(child)
    return model


def _has_fp8_params(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if p.dtype in (torch.float8_e4m3fn, getattr(torch, "float8_e5m2", None)):
            return True
    for m in model.modules():
        if isinstance(m, FP8Linear):
            return True
    return False


def _apply_torch_compile(model: torch.nn.Module) -> torch.nn.Module:
    """
    Applies torch.compile to optimize the neural network module with TorchInductor.
    Gracefully skips Triton FP8 compilation on GPU architectures below SM 8.9 (e.g. RTX 30-series)
    where Triton lacks fp8e4nv JIT instruction support.
    """
    if torch.cuda.is_available() and _has_fp8_params(model):
        cap = torch.cuda.get_device_capability()
        if cap < (8, 9):
            gpu_name = torch.cuda.get_device_name()
            print(
                f"\033[93m[Sapiens2] Notice: GPU {gpu_name} (SM {cap[0]}.{cap[1]}) does not support Triton JIT compilation for FP8 e4m3fn (requires SM 8.9+ / RTX 40-series). "
                f"Running FP8 in PyTorch eager mode without compilation.\033[0m"
            )
            return model

    print("\033[93m[Sapiens2] Applying torch.compile (first inference will compile Triton kernels)...\033[0m")
    try:
        return torch.compile(model, backend="inductor")
    except Exception as exc:
        print(f"\033[91m[Sapiens2] torch.compile failed: {exc}. Falling back to eager mode.\033[0m")
        return model



def unload_sapiens2_model(sapiens2_model) -> int:
    """
    Removes a model (or all models if None) from cache, moves inner module to CPU, and frees VRAM.
    """
    import gc
    from .pose import POSE_MODEL_CACHE

    removed = 0
    nn_model = getattr(sapiens2_model, "model", None)

    # Remove from dense model cache
    keys_to_remove = [k for k, v in _MODEL_CACHE.items() if v is sapiens2_model or getattr(v, "model", None) is nn_model]
    for k in keys_to_remove:
        del _MODEL_CACHE[k]
        removed += 1

    # Remove from pose model cache
    pose_keys_to_remove = [k for k, v in POSE_MODEL_CACHE.items() if v is sapiens2_model or getattr(v, "model", None) is nn_model]
    for k in pose_keys_to_remove:
        del POSE_MODEL_CACHE[k]
        removed += 1

    if sapiens2_model is None:
        removed += len(_MODEL_CACHE) + len(POSE_MODEL_CACHE)
        _MODEL_CACHE.clear()
        POSE_MODEL_CACHE.clear()

    if nn_model is not None:
        try:
            nn_model.to("cpu")
        except Exception:
            pass
        del nn_model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return removed


def _normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "_orig_mod.")
    changed = True
    while state_dict and changed:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in state_dict):
                state_dict = {key[len(prefix) :]: value for key, value in state_dict.items()}
                changed = True
                break
    return state_dict


def _read_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path, device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    return _normalize_state_dict(state_dict)


def _checkpoint_key_map(keys: Iterable[str]) -> dict[str, str]:
    original_keys = list(keys)
    normalized_keys = list(original_keys)
    changed = True
    while normalized_keys and changed:
        changed = False
        for prefix in ("module.", "_orig_mod."):
            if all(key.startswith(prefix) for key in normalized_keys):
                normalized_keys = [key[len(prefix) :] for key in normalized_keys]
                changed = True
                break
    return dict(zip(normalized_keys, original_keys))


def _detect_prefix_from_keys(keys: set[str]) -> str:
    if "backbone.patch_embed.projection.weight" in keys:
        return "backbone."
    if "patch_embed.projection.weight" in keys:
        return ""
    raise ValueError("Could not locate patch_embed.projection.weight in checkpoint.")


def _detect_prefix(state_dict: Dict[str, torch.Tensor]) -> str:
    return _detect_prefix_from_keys(set(state_dict))


def _detect_arch_from_embed_dim(embed_dim: int) -> str:
    for arch, spec in ARCH_SPECS.items():
        if spec["embed_dim"] == embed_dim:
            return arch
    raise ValueError(f"Unsupported Sapiens2 embed dim in checkpoint: {embed_dim}")


def _detect_arch(state_dict: Dict[str, torch.Tensor]) -> str:
    prefix = _detect_prefix(state_dict)
    embed_dim = int(state_dict[f"{prefix}patch_embed.projection.weight"].shape[0])
    return _detect_arch_from_embed_dim(embed_dim)


def _detect_task_from_keys(keys: set[str]) -> str:
    task_keys = {
        "segmentation": "decode_head.conv_seg.weight",
        "normal": "decode_head.conv_normal.weight",
        "pointmap": "decode_head.conv_pointmap.weight",
    }
    for task, key in task_keys.items():
        if key in keys:
            return task
    if "decode_head.conv_pose.weight" in keys:
        return "pose"
    raise ValueError("Could not infer dense task from checkpoint decode_head keys.")


def _detect_task(state_dict: Dict[str, torch.Tensor]) -> str:
    task = _detect_task_from_keys(set(state_dict))
    if task == "pose":
        raise ValueError("This checkpoint is a Sapiens2 pose model. Load it with task=pose.")
    return task


def _safetensors_tensor_shape(handle, tensor_name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in handle.get_slice(tensor_name).get_shape())
    except Exception:
        return tuple(int(dim) for dim in handle.get_tensor(tensor_name).shape)


def _inspect_safetensors_checkpoint(checkpoint_path: str) -> tuple[str, str]:
    from safetensors import safe_open

    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        key_map = _checkpoint_key_map(handle.keys())
        keys = set(key_map)
        task = _detect_task_from_keys(keys)
        prefix = _detect_prefix_from_keys(keys)
        shape = _safetensors_tensor_shape(handle, key_map[f"{prefix}patch_embed.projection.weight"])
    return task, _detect_arch_from_embed_dim(shape[0])


def inspect_checkpoint_task_arch(checkpoint_path: str) -> tuple[str, str]:
    if checkpoint_path.lower().endswith(".safetensors"):
        return _inspect_safetensors_checkpoint(checkpoint_path)
    state_dict = _read_checkpoint_state_dict(checkpoint_path)
    return _detect_task_from_keys(set(state_dict)), _detect_arch(state_dict)


def init_sapiens_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    registry_module: str,
    progress: NodeProgress | None = None,
):
    importlib.import_module(registry_module)
    from sapiens.engine.config import Config
    from sapiens.engine.datasets import Compose
    from sapiens.registry import MODELS

    started_at = time.perf_counter()
    config = Config.fromfile(config_path)
    if "init_cfg" in config.model["backbone"]:
        config.model["backbone"].pop("init_cfg")

    previous_dtype = torch.get_default_dtype()
    try:
        if dtype != torch.float32:
            torch.set_default_dtype(dtype)
        model = MODELS.build(config.model)
    finally:
        torch.set_default_dtype(previous_dtype)
    built_at = time.perf_counter()
    if progress is not None:
        progress.update()

    if checkpoint_path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path, device="cpu")
    else:
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint_data.get("state_dict", checkpoint_data.get("model", checkpoint_data))

    incompat = model.load_state_dict(state_dict, strict=False)
    del state_dict
    loaded_at = time.perf_counter()
    if incompat.missing_keys:
        print(f"Missing keys: {incompat.missing_keys}")
    if incompat.unexpected_keys:
        print(f"Unexpected keys: {incompat.unexpected_keys}")
    print(f"\033[96mModel loaded from {checkpoint_path}\033[0m")
    if progress is not None:
        progress.update()

    model.cfg = config
    model.data_preprocessor = MODELS.build(config.data_preprocessor)
    model.pipeline = Compose(config.test_pipeline)
    model.to(device)
    model.eval()
    moved_at = time.perf_counter()
    print(
        "Sapiens2 load timings: "
        f"build={built_at - started_at:.1f}s, "
        f"checkpoint={loaded_at - built_at:.1f}s, "
        f"move_to_{device}={moved_at - loaded_at:.1f}s"
    )
    if progress is not None:
        progress.update()

    return model


def _resolve_task_arch(
    requested_task: str,
    requested_arch: str,
    detected_task: str,
    detected_arch: str,
) -> tuple[str, str]:
    task = detected_task if requested_task == "auto" else requested_task
    arch = detected_arch if requested_arch == "auto" else requested_arch
    if task != detected_task:
        raise ValueError(
            f"Checkpoint appears to be task {detected_task!r}, but {task!r} was requested."
        )
    if arch != detected_arch:
        raise ValueError(
            f"Checkpoint appears to be arch {detected_arch!r}, but {arch!r} was requested."
        )
    return task, arch


def _dense_config_path(repo_path: str, task: str, arch: str) -> Path:
    template = _DENSE_CONFIG_TEMPLATES.get(task)
    if template is None:
        raise ValueError(f"Unsupported dense Sapiens2 task: {task}")
    config_path = get_sapiens_repo_path(repo_path) / template.format(arch=arch)
    if not config_path.is_file():
        raise FileNotFoundError(f"Expected official Sapiens2 config at: {config_path}")
    return config_path


def load_sapiens2_model(
    task: str,
    arch: str,
    device: str,
    dtype: str,
    checkpoint_path: str,
    sapiens_repo_path: str = "",
    use_fp8: bool = False,
    use_compile: bool = False,
) -> Sapiens2Model:
    checkpoint_path = os.path.abspath(os.path.expanduser(os.path.expandvars(checkpoint_path)))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Sapiens2 checkpoint not found: {checkpoint_path}")
    stat = os.stat(checkpoint_path)
    cache_key = (
        checkpoint_path,
        task,
        arch,
        device,
        dtype,
        sapiens_repo_path,
        stat.st_mtime_ns,
        stat.st_size,
        use_fp8,
        use_compile,
    )
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    progress = NodeProgress(7)
    _ensure_sapiens_importable(sapiens_repo_path)
    progress.update()
    detected_task, detected_arch = inspect_checkpoint_task_arch(checkpoint_path)
    if detected_task == "pose":
        raise ValueError("This checkpoint is a Sapiens2 pose model. Load it with task=pose.")
    progress.update()
    task, arch = _resolve_task_arch(task, arch, detected_task, detected_arch)
    config_path = _dense_config_path(sapiens_repo_path, task, arch)
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype, resolved_device)
    progress.update()

    model = init_sapiens_model(
        str(config_path),
        checkpoint_path,
        resolved_device,
        resolved_dtype,
        "sapiens.dense.src.models.init_model",
        progress,
    )

    if use_fp8 and hasattr(torch, "float8_e4m3fn"):
        model = _apply_fp8_weights(model)
        print("\033[93m[Sapiens2] FP8 weight quantization applied.\033[0m")
    if use_compile:
        model = _apply_torch_compile(model)

    loaded = Sapiens2Model(
        model=model,
        task=task,
        arch=arch,
        checkpoint_path=checkpoint_path,
        device=resolved_device,
        dtype=resolved_dtype,
        config_path=str(config_path),
        use_fp8=use_fp8,
        use_compile=use_compile,
    )
    _MODEL_CACHE[cache_key] = loaded
    progress.update()
    return loaded
