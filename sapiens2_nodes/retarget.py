"""
Sapiens2PoseRetarget — transfers a Target Person's 3D pose & orientation onto a Source Person
while strictly preserving the Source Person's anatomical proportions, body build, and relative height.

Mathematical Pipeline:
  1. Unprojects (u, v, depth) to camera space (X, Y, Z) if depth maps are provided.
  2. Extracts dimensionless anatomical proportion ratio vectors for both Source and Target.
  3. Calibrates scene scale on the Target canvas (with metric height ratio scaling if heights are known).
  4. Computes 2D directional unit vectors and 3D perspective foreshortening factors f in (0, 1] from Target pose.
  5. Applies Forward Kinematics (FK) tree traversal: P_child = P_parent + u_dir * (L_source * f_foreshorten).
  6. Clamps feet to the Target scene's ground contact plane.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import cv2
import numpy as np
import torch

# ─── BODY_25 Topology ─────────────────────────────────────────────────────────

_FK_ORDER = [
    (8, 1),   # MidHip  → Neck
    (1, 0),   # Neck    → Nose
    (0, 15), (15, 17),              # face right
    (0, 16), (16, 18),              # face left
    (1, 2),  (2, 3),  (3, 4),      # right arm
    (1, 5),  (5, 6),  (6, 7),      # left arm
    (8, 9),  (9, 10), (10, 11),    # right leg
    (11, 22),(22, 23),(11, 24),     # right foot
    (8, 12), (12, 13),(13, 14),    # left leg
    (14, 19),(19, 20),(14, 21),     # left foot
]

# Segment names for ratio lookup
_SEG_NAMES = {
    (8, 1): "r_torso",
    (1, 0): "r_neck_nose",
    (0, 15): "r_eye_span",
    (15, 17): "r_ear_span",
    (0, 16): "r_eye_span",
    (16, 18): "r_ear_span",
    (1, 2): "r_shoulder_span",
    (2, 3): "r_upper_arm",
    (3, 4): "r_forearm",
    (1, 5): "r_shoulder_span",
    (5, 6): "r_upper_arm",
    (6, 7): "r_forearm",
    (8, 9): "r_hip_span",
    (9, 10): "r_thigh",
    (10, 11): "r_shin",
    (11, 22): "r_foot_len",
    (22, 23): "r_foot_len",
    (11, 24): "r_foot_len",
    (8, 12): "r_hip_span",
    (12, 13): "r_thigh",
    (13, 14): "r_shin",
    (14, 19): "r_foot_len",
    (19, 20): "r_foot_len",
    (14, 21): "r_foot_len",
}

# Ideal unforeshortened canonical human proportions (used to preserve 2D perspective foreshortening)
_CANONICAL_RATIOS = {
    "r_torso": 0.28,
    "r_neck_nose": 0.08,
    "r_eye_span": 0.04,
    "r_ear_span": 0.05,
    "r_shoulder_span": 0.12,
    "r_upper_arm": 0.16,
    "r_forearm": 0.14,
    "r_hip_span": 0.09,
    "r_thigh": 0.24,
    "r_shin": 0.23,
    "r_foot_len": 0.06,
}

_BODY25_EDGES = (
    (1, 8),
    (1, 2), (1, 5),
    (2, 3), (3, 4),
    (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (8, 12),(12, 13),(13, 14),
    (0, 1),
    (0, 15),(15, 17),
    (0, 16),(16, 18),
    (14, 19),(19, 20),(14, 21),
    (11, 22),(22, 23),(11, 24),
)

_EDGE_COLORS = (
    (255,   0,  85),
    (255,  85,   0),
    (255, 170,   0),
    (170, 255,   0),
    ( 85, 255,   0),
    (  0, 255,  85),
    (  0, 255, 170),
    (  0, 170, 255),
    (  0,  85, 255),
    ( 85,   0, 255),
    (170,   0, 255),
    (255,   0, 170),
)

_INTERPUPILLARY_CM = 6.3
_BIAURICULAR_CM    = 14.0

_SAPIENS_TO_BODY25 = (0, 69, 6, 8, 41, 5, 7, 62, (9, 10), 10, 12, 14, 9, 11, 13, 2, 1, 4, 3, 15, 16, 17, 18, 19, 20)

_COCO18_TO_BODY25 = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14,
    14: 15, 15: 16, 16: 17, 17: 18,
}


def _safe_normalize(v: np.ndarray, eps: float = 1e-7, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Safely normalizes a vector with numerical epsilon and fallback for zero-norm vectors.
    """
    norm = float(np.linalg.norm(v))
    if norm > eps:
        return (v / norm).astype(np.float32)
    if fallback is not None:
        return fallback.astype(np.float32)
    fb = np.zeros_like(v, dtype=np.float32)
    if len(fb) > 0:
        fb[0] = 1.0
    return fb


def _build_orthonormal_frame(
    origin: np.ndarray,
    up_target: np.ndarray,
    right_target: np.ndarray,
) -> np.ndarray:
    """
    Constructs a robust SO(3) orthonormal basis matrix [x, y, z] using Gram-Schmidt orthogonalization.
    x points right (lateral), y points up (vertical), z points forward (out of coronal plane).
    """
    y = _safe_normalize(up_target - origin, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    right_raw = _safe_normalize(right_target - origin, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    z = _safe_normalize(np.cross(right_raw, y), fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    x = _safe_normalize(np.cross(y, z), fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    return np.column_stack([x, y, z]).astype(np.float32)


def _robust_rotation_matrix(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """
    Computes 3x3 rotation matrix R in SO(3) aligning v_from to v_to using Rodrigues' formula.
    Includes dynamic argmin(|v|) axis selection to handle 180° antiparallel vectors without singularities.
    """
    a = _safe_normalize(v_from, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    b = _safe_normalize(v_to, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))

    c = float(np.dot(a, b))  # cosine

    if c >= 0.999999:
        return np.eye(3, dtype=np.float32)

    if c <= -0.999999:
        # Antiparallel 180° rotation: select axis with smallest absolute component to guarantee non-zero cross product
        abs_a = np.abs(a)
        min_dim = int(np.argmin(abs_a))
        ortho_axis = np.zeros(3, dtype=np.float32)
        ortho_axis[min_dim] = 1.0
        axis = _safe_normalize(np.cross(a, ortho_axis))
        # R = 2 * (axis @ axis.T) - I
        return (2.0 * np.outer(axis, axis) - np.eye(3)).astype(np.float32)

    v = np.cross(a, b)
    vx, vy, vz = v[0], v[1], v[2]
    K = np.array([
        [0.0, -vz, vy],
        [vz, 0.0, -vx],
        [-vy, vx, 0.0],
    ], dtype=np.float32)

    eye = np.eye(3, dtype=np.float32)
    R = eye + K + (K @ K) * (1.0 / (1.0 + c))
    return R.astype(np.float32)


def _compute_hinge_normal(p_root: np.ndarray, p_mid: np.ndarray, p_tip: np.ndarray) -> np.ndarray:
    """
    Computes anatomical hinge normal plane for a 3-joint limb chain (Root -> Mid -> Tip),
    e.g. Shoulder -> Elbow -> Wrist or Hip -> Knee -> Ankle, to lock the axial Twist degree of freedom.
    """
    v_upper = p_mid - p_root
    v_lower = p_tip - p_mid
    n = np.cross(v_upper, v_lower)
    return _safe_normalize(n, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))


def _remap_308_to_body25(triples: np.ndarray) -> np.ndarray:
    out = np.zeros((25, 3), dtype=np.float32)
    for bi, spec in enumerate(_SAPIENS_TO_BODY25):
        if isinstance(spec, tuple):
            valid = [triples[idx] for idx in spec if 0 <= idx < len(triples) and triples[idx, 2] > 0]
            if valid:
                arr = np.stack(valid)
                out[bi] = np.array([arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].min()], dtype=np.float32)
        elif 0 <= int(spec) < len(triples):
            out[bi] = triples[int(spec)]
    return out


def _remap_coco18_to_body25(triples: np.ndarray) -> np.ndarray:
    out = np.zeros((25, 3), dtype=np.float32)
    for c18_idx, b25_idx in _COCO18_TO_BODY25.items():
        if c18_idx < len(triples):
            out[b25_idx] = triples[c18_idx]
    if out[9, 2] > 0 and out[12, 2] > 0:
        out[8, 0] = (out[9, 0] + out[12, 0]) * 0.5
        out[8, 1] = (out[9, 1] + out[12, 1]) * 0.5
        out[8, 2] = min(out[9, 2], out[12, 2])
    return out


def _flat_to_triples(flat: list, expected_count: int) -> tuple[np.ndarray, np.ndarray]:
    if not flat or len(flat) < 3:
        return np.zeros((expected_count, 2), dtype=np.float32), np.zeros(expected_count, dtype=np.float32)
    n = min(len(flat) // 3, expected_count)
    pts = np.zeros((expected_count, 2), dtype=np.float32)
    conf = np.zeros(expected_count, dtype=np.float32)
    for i in range(n):
        pts[i] = [float(flat[i * 3]), float(flat[i * 3 + 1])]
        conf[i] = float(flat[i * 3 + 2])
    return pts, conf


def _parse_json(pose_input: Any):
    """
    Parses OpenPose / DWPose JSON string, dictionary, or POSE_KEYPOINT list.
    Returns:
      (body_kps, body_conf, meta, (face_kps, face_conf), (lhand_kps, lhand_conf), (rhand_kps, rhand_conf))
    """
    if isinstance(pose_input, (dict, list)):
        data = pose_input
    elif isinstance(pose_input, str) and pose_input.strip():
        try:
            data = json.loads(pose_input)
        except Exception:
            return None, None, {}, (None, None), (None, None), (None, None)
    else:
        return None, None, {}, (None, None), (None, None), (None, None)

    if isinstance(data, list):
        if not data:
            return None, None, {}, (None, None), (None, None), (None, None), (None, None)
        data = data[0]
    
    if not isinstance(data, dict):
        return None, None, {}, (None, None), (None, None), (None, None), (None, None)

    meta = data.get("sapiens_meta", {})
    people = data.get("people", [])
    if not people:
        return None, None, meta, (None, None), (None, None), (None, None), (None, None)

    person = people[0]
    target_name = person.get("target") or data.get("target") or meta.get("target")
    if target_name:
        meta["target"] = target_name

    sapiens_flat = person.get("sapiens_keypoints_2d", [])
    flat = person.get("pose_keypoints_2d", [])
    face_flat = person.get("face_keypoints_2d", [])
    lhand_flat = person.get("hand_left_keypoints_2d", [])
    rhand_flat = person.get("hand_right_keypoints_2d", [])

    if flat and len(flat) > 0:
        n = len(flat) // 3
        triples = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            triples[i] = [flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]]
        if n >= 308:
            b25_triples = _remap_308_to_body25(triples)
        elif n == 18:
            b25_triples = _remap_coco18_to_body25(triples)
            # If sapiens_flat has 308 keypoints, extract authentic feet (15..20 -> 19..24 in Body25)
            if sapiens_flat and len(sapiens_flat) >= 308 * 3:
                s_n = len(sapiens_flat) // 3
                s_triples = np.zeros((s_n, 3), dtype=np.float32)
                for i in range(s_n):
                    s_triples[i] = [sapiens_flat[i * 3], sapiens_flat[i * 3 + 1], sapiens_flat[i * 3 + 2]]
                # 15: LBigToe (19), 16: LSmallToe (20), 17: LHeel (21)
                # 18: RBigToe (22), 19: RSmallToe (23), 20: RHeel (24)
                for s_idx, b_idx in ((15, 19), (16, 20), (17, 21), (18, 22), (19, 23), (20, 24)):
                    if s_idx < s_n and s_triples[s_idx, 2] > 0.05:
                        b25_triples[b_idx] = s_triples[s_idx]
        else:
            b25_triples = np.zeros((25, 3), dtype=np.float32)
            m = min(n, 25)
            b25_triples[:m] = triples[:m]
        body_kps, body_conf = b25_triples[:, :2].copy(), b25_triples[:, 2].copy()
    elif sapiens_flat and len(sapiens_flat) >= 308 * 3:
        n = len(sapiens_flat) // 3
        triples = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            triples[i] = [sapiens_flat[i * 3], sapiens_flat[i * 3 + 1], sapiens_flat[i * 3 + 2]]
        b25_triples = _remap_308_to_body25(triples)
        body_kps, body_conf = b25_triples[:, :2].copy(), b25_triples[:, 2].copy()
    else:
        return None, None, meta, (None, None), (None, None), (None, None), (None, None)

    face_data = _flat_to_triples(face_flat, 68) if face_flat else (None, None)
    lhand_data = _flat_to_triples(lhand_flat, 21) if lhand_flat else (None, None)
    rhand_data = _flat_to_triples(rhand_flat, 21) if rhand_flat else (None, None)
    sapiens_data = _flat_to_triples(sapiens_flat, 308) if (sapiens_flat and len(sapiens_flat) >= 308 * 3) else (None, None)

    return body_kps, body_conf, meta, face_data, lhand_data, rhand_data, sapiens_data


def _pad_to(kps: np.ndarray, conf: np.ndarray, n: int = 25):
    if len(kps) < n:
        kps  = np.vstack([kps,  np.zeros((n - len(kps),  2), np.float32)])
        conf = np.concatenate([conf, np.zeros(n - len(conf), np.float32)])
    return kps[:n].copy(), conf[:n].copy()


def _extract_depth_map_array(depth_map: Optional[torch.Tensor], img_hw: tuple[int, int]) -> Optional[np.ndarray]:
    """
    Extracts a 2D float32 depth map normalized to relative depth [0, 1].
    """
    if depth_map is None:
        return None
    try:
        t = depth_map.detach().cpu().float()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3:
            t = t.mean(dim=-1) if t.shape[-1] in (1, 3, 4) else t[0]
        d_np = t.numpy()
        H, W = img_hw
        if d_np.shape[:2] != (H, W):
            d_np = cv2.resize(d_np, (W, H), interpolation=cv2.INTER_LINEAR)
        d_min, d_max = float(d_np.min()), float(d_np.max())
        if d_max > d_min:
            d_np = (d_np - d_min) / (d_max - d_min)
        else:
            d_np = np.zeros_like(d_np)
        return d_np.astype(np.float32)
    except Exception as exc:
        print(f"[Sapiens2 Retarget] Depth map extraction failed: {exc}")
        return None


def _extract_raw_depth_array(depth_map: Optional[torch.Tensor], img_hw: tuple[int, int]) -> Optional[np.ndarray]:
    """
    Extracts a raw 2D float32 depth map WITHOUT normalization, preserving metric or physical depth scale.
    """
    if depth_map is None:
        return None
    try:
        t = depth_map.detach().cpu().float()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3:
            t = t.mean(dim=-1) if t.shape[-1] in (1, 3, 4) else t[0]
        d_np = t.numpy().astype(np.float32)
        H, W = img_hw
        if d_np.shape[:2] != (H, W):
            d_np = cv2.resize(d_np, (W, H), interpolation=cv2.INTER_LINEAR)
        return d_np
    except Exception as exc:
        print(f"[Sapiens2 3D] Raw depth map extraction failed: {exc}")
        return None


def _extract_pointmap_array(
    pointmap_input: Any,
    img_hw: tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Extracts a raw float32 (H, W, 3) pointmap array with (X, Y, Z) per pixel in camera space.
    Does NOT normalize or flip axes, preserving true 3D spatial coordinates.
    """
    if pointmap_input is None:
        return None
    try:
        if isinstance(pointmap_input, torch.Tensor):
            t = pointmap_input.detach().cpu().float()
            # If shape is [B, C, H, W] where C == 3
            if t.ndim == 4 and t.shape[1] == 3:
                t = t[0].permute(1, 2, 0)
            # If shape is [B, H, W, C] where C == 3
            elif t.ndim == 4 and t.shape[-1] == 3:
                t = t[0]
            # If shape is [C, H, W] where C == 3
            elif t.ndim == 3 and t.shape[0] == 3 and t.shape[-1] != 3:
                t = t.permute(1, 2, 0)
            elif t.ndim == 3 and t.shape[-1] == 3:
                pass
            else:
                return None
            arr = t.numpy().astype(np.float32)
        elif isinstance(pointmap_input, np.ndarray):
            arr = pointmap_input.astype(np.float32)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
                arr = np.transpose(arr, (1, 2, 0))
        else:
            return None

        H, W = img_hw
        if arr.shape[:2] != (H, W):
            arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
        return arr
    except Exception as exc:
        print(f"[Sapiens2 3D] Pointmap extraction failed: {exc}")
        return None


def _sample_pointmap_at_keypoints(
    kps: np.ndarray,
    conf: np.ndarray,
    pointmap_np: np.ndarray,
    img_hw: tuple[int, int],
    thr: float = 0.1,
) -> np.ndarray:
    """
    Samples (H, W, 3) pointmap directly at (u, v) joint pixel locations.
    Returns (N, 3) XYZ in camera coordinates (NaN for undetected/invalid points).
    """
    H, W = img_hw
    n_kps = len(kps)
    kps_3d = np.full((n_kps, 3), np.nan, dtype=np.float32)
    for i in range(n_kps):
        if conf[i] < thr:
            continue
        u = int(np.clip(round(float(kps[i, 0])), 0, W - 1))
        v = int(np.clip(round(float(kps[i, 1])), 0, H - 1))
        xyz = pointmap_np[v, u]
        if np.all(np.isfinite(xyz)) and xyz[2] > 0:
            kps_3d[i] = xyz
    return kps_3d


def _unproject_keypoints_3d(
    kps: np.ndarray,
    conf: np.ndarray,
    depth_np: np.ndarray,
    img_hw: tuple[int, int],
    fov_deg: float = 60.0,
    thr: float = 0.1,
) -> np.ndarray:
    """
    Back-projects 2D (u, v) keypoints to 3D (X, Y, Z) world/camera coordinates using pinhole model.
    """
    H, W = img_hw
    n_kps = len(kps)
    kps_3d = np.full((n_kps, 3), np.nan, dtype=np.float32)

    fov_rad = np.radians(max(10.0, min(160.0, float(fov_deg))))
    fx = float(W / (2.0 * np.tan(fov_rad / 2.0)))
    fy = fx
    cx = float(W / 2.0)
    cy = float(H / 2.0)

    for i in range(n_kps):
        if conf[i] < thr:
            continue
        u = int(np.clip(round(float(kps[i, 0])), 0, W - 1))
        v = int(np.clip(round(float(kps[i, 1])), 0, H - 1))
        z_val = float(depth_np[v, u])
        if z_val <= 0 or not np.isfinite(z_val):
            continue
        x_val = (float(kps[i, 0]) - cx) * z_val / fx
        y_val = (float(kps[i, 1]) - cy) * z_val / fy
        kps_3d[i] = [x_val, y_val, z_val]
    return kps_3d


def _dist_3d_or_2d(
    i: int,
    j: int,
    kps: np.ndarray,
    conf: np.ndarray,
    kps_3d: Optional[np.ndarray] = None,
    thr: float = 0.15,
) -> float:
    """
    Returns true 3D Euclidean distance if kps_3d is available and valid,
    otherwise falls back to 2D Euclidean distance.
    """
    if i >= len(conf) or j >= len(conf) or conf[i] < thr or conf[j] < thr:
        return 0.0
    if kps_3d is not None and i < len(kps_3d) and j < len(kps_3d):
        p1_3d = kps_3d[i]
        p2_3d = kps_3d[j]
        if np.all(np.isfinite(p1_3d)) and np.all(np.isfinite(p2_3d)):
            return float(np.linalg.norm(p1_3d - p2_3d))
    return float(np.linalg.norm(kps[i] - kps[j]))


def _build_kps_3d(
    kps: np.ndarray,
    conf: np.ndarray,
    pointmap_input: Any = None,
    depth_input: Any = None,
    img_hw: tuple[int, int] = (512, 512),
    fov_deg: float = 60.0,
    thr: float = 0.1,
) -> tuple[Optional[np.ndarray], str]:
    """
    Builds 3D keypoint coordinates using prioritized pipeline:
      1. Sapiens Pointmap direct sampling (highest quality native 3D)
      2. Pinhole depth unprojection (camera geometry)
      3. 2D fallback (None, '2d_only')
    """
    pointmap_arr = _extract_pointmap_array(pointmap_input, img_hw)
    if pointmap_arr is not None:
        kps_3d = _sample_pointmap_at_keypoints(kps, conf, pointmap_arr, img_hw, thr=thr)
        valid_count = np.sum(np.all(np.isfinite(kps_3d), axis=1))
        if valid_count >= 4:
            return kps_3d, "pointmap_3d"

    raw_depth_arr = _extract_raw_depth_array(depth_input, img_hw)
    if raw_depth_arr is not None:
        d_min, d_max = float(raw_depth_arr.min()), float(raw_depth_arr.max())
        d_arr = raw_depth_arr.copy()
        if d_max <= 1.05 and d_max > d_min:
            # Normalized relative depth: map [0, 1] to typical human distance [1.5m, 4.5m]
            d_arr = 1.5 + (1.0 - (d_arr - d_min) / (d_max - d_min)) * 3.0
        kps_3d = _unproject_keypoints_3d(kps, conf, d_arr, img_hw, fov_deg=fov_deg, thr=thr)
        valid_count = np.sum(np.all(np.isfinite(kps_3d), axis=1))
        if valid_count >= 4:
            return kps_3d, "pinhole_unproject"

    return None, "2d_only"


def _extract_proportions(
    kps: np.ndarray,
    conf: np.ndarray,
    depth_map: Optional[np.ndarray],
    img_hw: tuple[int, int],
    thr: float = 0.15,
    kps_3d: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Measures true unforeshortened bone lengths using 3D coordinates (pointmap or unprojected depth)
    or falls back to 2D pixel distance + depth tilt recovery,
    then computes dimensionless ratios r_i = L_i / H_total.
    """
    H, W = img_hw

    def dist(i: int, j: int) -> float:
        if kps_3d is not None:
            d = _dist_3d_or_2d(i, j, kps, conf, kps_3d=kps_3d, thr=thr)
            if d > 0:
                return d
        if i >= len(conf) or j >= len(conf) or conf[i] < thr or conf[j] < thr:
            return 0.0
        p1 = kps[i]
        p2 = kps[j]
        d_2d = float(np.linalg.norm(p1 - p2))
        if depth_map is not None:
            u1, v1 = int(np.clip(p1[0], 0, W - 1)), int(np.clip(p1[1], 0, H - 1))
            u2, v2 = int(np.clip(p2[0], 0, W - 1)), int(np.clip(p2[1], 0, H - 1))
            torso_2d = float(np.linalg.norm(kps[1] - kps[8])) if conf[1] > thr and conf[8] > thr else 120.0
            sz_pixels = max(torso_2d * 1.5, 50.0)
            dz_norm = float(depth_map[v1, u1] - depth_map[v2, u2])
            dz_px = dz_norm * sz_pixels
            return float(np.sqrt(d_2d ** 2 + dz_px ** 2))
        return d_2d

    # Torso: Neck (1) -> MidHip (8)
    torso = dist(1, 8)
    if torso <= 0:
        torso = dist(1, 9) if dist(1, 9) > 0 else dist(1, 12)
    if torso <= 0:
        torso = 120.0 if kps_3d is None else 0.45

    neck_nose = dist(1, 0)
    if neck_nose <= 0:
        neck_nose = torso * 0.25

    eye_span = dist(15, 16)
    if eye_span <= 0:
        eye_span = torso * 0.15

    ear_span = dist(17, 18)
    if ear_span <= 0:
        ear_span = torso * 0.30

    shoulder_span = dist(2, 5)
    if shoulder_span <= 0:
        shoulder_span = torso * 0.80

    hip_span = dist(9, 12)
    if hip_span <= 0:
        hip_span = torso * 0.45

    def bilateral_avg(i_r: int, j_r: int, i_l: int, j_l: int, min_len: float, default_len: float) -> float:
        d_r = dist(i_r, j_r)
        d_l = dist(i_l, j_l)
        c_r = float(min(conf[i_r], conf[j_r])) if i_r < len(conf) and j_r < len(conf) else 0.0
        c_l = float(min(conf[i_l], conf[j_l])) if i_l < len(conf) and j_l < len(conf) else 0.0
        r_ok = d_r >= min_len and c_r > thr
        l_ok = d_l >= min_len and c_l > thr
        if r_ok and l_ok:
            return float((d_r * c_r + d_l * c_l) / (c_r + c_l + 1e-7))
        elif r_ok:
            return d_r
        elif l_ok:
            return d_l
        else:
            return default_len

    # Arms: Upper arms (2->3, 5->6) and Forearms (3->4, 6->7)
    avg_uarm = bilateral_avg(2, 3, 5, 6, min_len=torso * 0.25, default_len=torso * 0.55)
    avg_farm = bilateral_avg(3, 4, 6, 7, min_len=torso * 0.20, default_len=torso * 0.48)

    # Legs: Thighs (9->10, 12->13) and Shins (10->11, 13->14)
    avg_thigh = bilateral_avg(9, 10, 12, 13, min_len=torso * 0.30, default_len=torso * 0.88)
    avg_shin = bilateral_avg(10, 11, 13, 14, min_len=torso * 0.30, default_len=torso * 0.85)

    # Feet (11->22, 14->19)
    foot_len = bilateral_avg(11, 22, 14, 19, min_len=torso * 0.10, default_len=torso * 0.22)

    # Crown above nose: ~15% of nose-to-neck (suprasternal notch at clavicle)
    head_height = neck_nose * 1.15
    total_height = head_height + torso + avg_thigh + avg_shin + foot_len * 0.40

    # Estimated 2D total height in pixels
    if conf[0] > thr and conf[11] > thr:
        total_height_px = float(np.linalg.norm(kps[0] - kps[11]))
    else:
        total_height_px = total_height

    return {
        "r_torso": torso / total_height,
        "r_head": head_height / total_height,
        "r_neck_nose": neck_nose / total_height,
        "r_eye_span": (eye_span * 0.5) / total_height,
        "r_ear_span": (ear_span * 0.5) / total_height,
        "r_shoulder_span": (shoulder_span * 0.5) / total_height,
        "r_hip_span": (hip_span * 0.5) / total_height,
        "r_upper_arm": avg_uarm / total_height,
        "r_forearm": avg_farm / total_height,
        "r_thigh": avg_thigh / total_height,
        "r_shin": avg_shin / total_height,
        "r_foot_len": (foot_len * 0.40) / total_height,
        "total_height": total_height,
        "total_height_px": total_height_px,
        "eye_span": eye_span,
        "ear_span": ear_span,
    }


def _find_anchor_contact_joint(
    kps_target: np.ndarray,
    conf_target: np.ndarray,
    canvas_wh: tuple[int, int],
    grounding_mode: str,
    thr: float = 0.15,
) -> Optional[int]:
    """
    Finds the keypoint index of the Target person closest to the chosen spatial canvas anchor.
    Supports arbitrary postures: standing, sitting, kneeling, handstands, hanging, wall-leaning.
    """
    cW, cH = canvas_wh
    anchor_map = {
        "bottom_center": (cW / 2.0, float(cH)),
        "bottom_left":   (0.0, float(cH)),
        "bottom_right":  (float(cW), float(cH)),
        "top_center":    (cW / 2.0, 0.0),
        "top_left":      (0.0, 0.0),
        "top_right":     (float(cW), 0.0),
        "left_center":   (0.0, cH / 2.0),
        "right_center":  (float(cW), cH / 2.0),
        "center":        (cW / 2.0, cH / 2.0),
    }
    if grounding_mode not in anchor_map:
        return None

    ax, ay = anchor_map[grounding_mode]
    best_idx = None
    min_dist = float("inf")

    for i in range(min(len(kps_target), 25)):
        if conf_target[i] < thr:
            continue
        px, py = kps_target[i, 0], kps_target[i, 1]
        d = float(np.hypot(px - ax, py - ay))
        if d < min_dist:
            min_dist = d
            best_idx = i

    return best_idx


def _retarget_kinematics(
    ratios_source: dict[str, float],
    ratios_target: dict[str, float],
    kps_target: np.ndarray,
    conf_target: np.ndarray,
    height_ratio: float,
    grounding: str,
    canvas_wh: tuple[int, int],
    thr: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transfers Target Person's pose onto Source Person by scaling each joint vector
    by (r_source_k / r_target_k) * height_ratio.
    
    This strictly preserves:
      - 3D body rotation in space (yaw, pitch, roll)
      - 3D perspective foreshortening and camera tilt
      - Spine curvature and natural posture
      - Source Person's anatomical limb length ratios and relative body scale
    """
    cW, cH = canvas_wh
    ret = np.zeros((25, 2), dtype=np.float32)
    eff_conf = conf_target.copy()
    # Target canvas space standing height for default anatomical FK fallbacks
    standing_h = float(cH * 0.75 * height_ratio)
    if standing_h < 50.0:
        standing_h = float(cH * 0.75)

    # Anatomical default relative offsets for every FK joint (in target canvas coordinates)
    default_offsets = {
        (8, 1): np.array([0.0, -ratios_source.get("r_torso", 0.28) * standing_h], np.float32),
        (1, 0): np.array([0.0, -ratios_source.get("r_neck_nose", 0.08) * standing_h], np.float32),
        (0, 15): np.array([-ratios_source.get("r_eye_span", 0.04) * standing_h, -4.0], np.float32),
        (0, 16): np.array([ratios_source.get("r_eye_span", 0.04) * standing_h, -4.0], np.float32),
        (15, 17): np.array([-ratios_source.get("r_ear_span", 0.05) * standing_h, 0.0], np.float32),
        (16, 18): np.array([ratios_source.get("r_ear_span", 0.05) * standing_h, 0.0], np.float32),
        (1, 2): np.array([-ratios_source.get("r_shoulder_span", 0.12) * standing_h, 8.0], np.float32),
        (2, 3): np.array([0.0, ratios_source.get("r_upper_arm", 0.16) * standing_h], np.float32),
        (3, 4): np.array([0.0, ratios_source.get("r_forearm", 0.14) * standing_h], np.float32),
        (1, 5): np.array([ratios_source.get("r_shoulder_span", 0.12) * standing_h, 8.0], np.float32),
        (5, 6): np.array([0.0, ratios_source.get("r_upper_arm", 0.16) * standing_h], np.float32),
        (6, 7): np.array([0.0, ratios_source.get("r_forearm", 0.14) * standing_h], np.float32),
        (8, 9): np.array([-ratios_source.get("r_hip_span", 0.09) * standing_h, 0.0], np.float32),
        (9, 10): np.array([0.0, ratios_source.get("r_thigh", 0.24) * standing_h], np.float32),
        (10, 11): np.array([0.0, ratios_source.get("r_shin", 0.23) * standing_h], np.float32),
        (8, 12): np.array([ratios_source.get("r_hip_span", 0.09) * standing_h, 0.0], np.float32),
        (12, 13): np.array([0.0, ratios_source.get("r_thigh", 0.24) * standing_h], np.float32),
        (13, 14): np.array([0.0, ratios_source.get("r_shin", 0.23) * standing_h], np.float32),
        (11, 22): np.array([6.0, 10.0], np.float32),
        (22, 23): np.array([5.0, 0.0], np.float32),
        (11, 24): np.array([-5.0, 6.0], np.float32),
        (14, 19): np.array([-6.0, 10.0], np.float32),
        (19, 20): np.array([-5.0, 0.0], np.float32),
        (14, 21): np.array([5.0, 6.0], np.float32),
    }

    # Root placement: Target Person's MidHip on canvas
    root = 8
    if conf_target[root] > thr:
        ret[root] = kps_target[root].copy()
    else:
        valid_tgt = [i for i in [9, 12] if conf_target[i] > thr]
        if valid_tgt:
            ret[root] = np.mean([kps_target[i] for i in valid_tgt], axis=0)
            eff_conf[root] = 0.8
        elif conf_target[1] > thr:
            # Estimate MidHip downward from Neck
            ret[root] = kps_target[1] + np.array([0.0, ratios_source.get("r_torso", 0.28) * standing_h], np.float32)
            eff_conf[root] = 0.6
        else:
            ret[root] = np.array([cW / 2.0, cH * 0.58], np.float32)
            eff_conf[root] = 0.5

    for parent, child in _FK_ORDER:
        if parent >= 25 or child >= 25:
            continue

        tgt_ok = conf_target[parent] > thr and conf_target[child] > thr
        if tgt_ok:
            # 1. Target joint vector in canvas space
            p_par = kps_target[parent]
            p_chi = kps_target[child]
            dv_target = p_chi - p_par

            # 2. Segment proportion ratio scaling (2D Perspective Foreshortening Fix)
            # We divide by the ideal unforeshortened canonical ratio instead of the 2D
            # foreshortened target ratio (r_tgt). This creates a 'body_build_modifier'
            # (e.g., source arm is 10% longer than average human arm) which we apply
            # to the target's exact 2D projection, perfectly preserving its depth foreshortening!
            seg_key = _SEG_NAMES.get((parent, child), "r_torso")
            r_src = ratios_source.get(seg_key, 0.15)
            r_canonical = _CANONICAL_RATIOS.get(seg_key, 0.15)

            if r_canonical > 1e-4:
                body_build_modifier = float(np.clip(r_src / r_canonical, 0.4, 2.5))
            else:
                body_build_modifier = 1.0
                
            bone_scale = float(np.clip(body_build_modifier * height_ratio, 0.25, 4.0))

            ret[child] = ret[parent] + dv_target * bone_scale
            eff_conf[child] = max(conf_target[child], 0.7)
        else:
            # Target connection is broken (parent or both missing): use default anatomical offset from ret[parent]
            dv_default = default_offsets.get((parent, child), np.array([0.0, 20.0], np.float32))
            ret[child] = ret[parent] + dv_default
            eff_conf[child] = max(eff_conf[parent] * 0.7, 0.3)

    # Spatial Multi-Anchor Grounding translation
    if grounding != "none":
        contact_k = _find_anchor_contact_joint(kps_target, conf_target, canvas_wh, grounding, thr=thr)
        if contact_k is not None and conf_target[contact_k] > thr:
            delta = kps_target[contact_k] - ret[contact_k]
            ret += delta

    return ret, eff_conf


def _estimate_biometric_height(
    kps: np.ndarray,
    conf: np.ndarray,
    total_height: float,
    kps_3d: Optional[np.ndarray] = None,
    thr: float = 0.3,
) -> float:
    """
    Robust multi-feature biometric height estimation with outlier rejection.
    Accurate for full-body, half-body, seated, and cropped poses.
    Uses calibrated anatomical scaling laws relating biometric features to full standing height.
    """
    candidates = []

    def get_d(i: int, j: int) -> float:
        if i >= len(conf) or j >= len(conf) or conf[i] < thr or conf[j] < thr:
            return 0.0
        return _dist_3d_or_2d(i, j, kps, conf, kps_3d=kps_3d, thr=thr)

    # 1. Interpupillary distance (~6.3 cm IPD)
    if len(conf) > 16 and conf[15] > thr and conf[16] > thr:
        eye_d = get_d(15, 16)
        min_thresh = 0.02 if kps_3d is not None else 8.0
        if eye_d >= min_thresh:
            candidates.append(total_height * (_INTERPUPILLARY_CM / eye_d))

    # 2. Biauricular ear span (~14.0 cm)
    if len(conf) > 18 and conf[17] > thr and conf[18] > thr:
        ear_d = get_d(17, 18)
        min_thresh = 0.04 if kps_3d is not None else 14.0
        if ear_d >= min_thresh:
            candidates.append(total_height * (_BIAURICULAR_CM / ear_d))

    # 3. Head height (Nose to Neck ~15.0 cm)
    if len(conf) > 1 and conf[0] > thr and conf[1] > thr:
        head_d = get_d(0, 1)
        min_thresh = 0.04 if kps_3d is not None else 12.0
        if head_d >= min_thresh:
            candidates.append(total_height * (15.0 / head_d))

    # 4. Shoulder width (~38.0 cm biacromial)
    if len(conf) > 5 and conf[2] > thr and conf[5] > thr:
        sh_d = get_d(2, 5)
        min_thresh = 0.08 if kps_3d is not None else 25.0
        if sh_d >= min_thresh:
            candidates.append(total_height * (38.0 / sh_d))

    # 5. Torso length (~47.0 cm Neck to MidHip)
    if len(conf) > 8 and conf[1] > thr and conf[8] > thr:
        torso_d = get_d(1, 8)
        min_thresh = 0.10 if kps_3d is not None else 35.0
        if torso_d >= min_thresh:
            candidates.append(total_height * (47.0 / torso_d))

    valid = [h for h in candidates if 120.0 <= h <= 215.0]
    return float(np.median(valid)) if valid else 172.0


_HAND21_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

_DWPOSE_HAND_EDGE_COLORS = (
    (255, 0, 0),    (255, 76, 0),   (255, 153, 0),  (255, 229, 0),
    (204, 255, 0),  (128, 255, 0),  (51, 255, 0),   (0, 255, 25),
    (0, 255, 102),  (0, 255, 178),  (0, 255, 255),  (0, 178, 255),
    (0, 102, 255),  (0, 25, 255),   (51, 0, 255),   (127, 0, 255),
    (204, 0, 255),  (255, 0, 230),  (255, 0, 153),  (255, 0, 77),
)


def _retarget_face_dwpose(
    face_src: Optional[tuple[np.ndarray, np.ndarray]],
    face_tgt: Optional[tuple[np.ndarray, np.ndarray]],
    kps_ret: np.ndarray,
    conf_ret: np.ndarray,
    canvas_wh: tuple[int, int],
    thr: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retargets 68-point facial landmarks by transferring the Target's facial expression
    (mouth opening/smile, eyelid blinking, eyebrow elevation) onto the Source person's
    authentic facial bone structure, jawline, eye spacing, and identity.
    Un-rotates source roll first so retargeted face tilt matches target 1:1 without adding source tilt.
    """
    pts_out = np.zeros((68, 2), dtype=np.float32)
    conf_out = np.zeros(68, dtype=np.float32)

    nose_pt = kps_ret[0]
    neck_pt = kps_ret[1]
    r_eye_pt = kps_ret[15] if len(kps_ret) > 15 and conf_ret[15] > thr else None
    l_eye_pt = kps_ret[16] if len(kps_ret) > 16 and conf_ret[16] > thr else None

    # Expected anatomical eye span relative to retargeted torso length (prevents face shrinking when foreshortened)
    torso_ret_len = float(np.linalg.norm(kps_ret[1] - kps_ret[8])) if (conf_ret[1] > thr and conf_ret[8] > thr) else 120.0
    expected_eye_span = max(torso_ret_len * 0.18, 14.0)

    # Head angle theta (roll) and scale proportional to eye distance
    if r_eye_pt is not None and l_eye_pt is not None:
        dx = l_eye_pt[0] - r_eye_pt[0]
        dy = l_eye_pt[1] - r_eye_pt[1]
        # If target eyes are swapped (person facing backwards or looking over shoulder), avoid 180° upside-down flip
        if dx < -5.0:
            head_angle = float(np.arctan2(-dy, -dx)) # maintain upright facial axis
        else:
            head_angle = float(np.arctan2(dy, dx))
        head_scale = max(float(np.hypot(dx, dy)), expected_eye_span)
        if conf_ret[0] <= thr:
            eye_mid = (r_eye_pt + l_eye_pt) * 0.5
            dir_down = np.array([-np.sin(head_angle), np.cos(head_angle)], dtype=np.float32)
            nose_pt = eye_mid + dir_down * (head_scale * 0.75)
    elif conf_ret[0] > thr and conf_ret[1] > thr:
        dx = nose_pt[0] - neck_pt[0]
        dy = nose_pt[1] - neck_pt[1]
        head_angle = float(np.arctan2(dx, -dy))
        head_scale = max(float(np.hypot(dx, dy)) * 0.65, expected_eye_span)
    else:
        head_angle = 0.0
        head_scale = expected_eye_span

    src_has_face = face_src is not None and face_src[0] is not None and np.sum(face_src[1] > thr) >= 10
    tgt_has_face = face_tgt is not None and face_tgt[0] is not None and np.sum(face_tgt[1] > thr) >= 6

    # If Target face does not exist or target head is turned completely away,
    # strictly avoid drawing front-facing facial landmarks on the back of the head
    if not tgt_has_face:
        return pts_out, conf_out

    # If only Target face exists
    if not src_has_face and tgt_has_face:
        pts_t, conf_t = face_tgt
        c_tgt = pts_t[30] if conf_t[30] > thr else np.mean(pts_t[conf_t > thr], axis=0)
        s_tgt = float(np.linalg.norm(pts_t[45] - pts_t[36])) if (conf_t[45] > thr and conf_t[36] > thr) else 40.0
        if s_tgt < 1.0: s_tgt = 40.0
        scale_f = head_scale / s_tgt
        for i in range(68):
            if conf_t[i] > thr:
                rel = pts_t[i] - c_tgt
                pts_out[i] = nose_pt + rel * scale_f
                conf_out[i] = conf_t[i]
        return pts_out, conf_out

    # BOTH Source and Target faces exist: Execute Expression Transfer with Source Identity Preservation!
    pts_s, conf_s = face_src
    pts_t, conf_t = face_tgt

    c_src = pts_s[30] if conf_s[30] > thr else np.mean(pts_s[conf_s > thr], axis=0)
    d_eyes_src = float(np.linalg.norm(pts_s[45] - pts_s[36])) if (conf_s[45] > thr and conf_s[36] > thr) else 40.0
    if d_eyes_src < 1.0: d_eyes_src = 40.0

    # Un-rotate source face so source mesh is level (0° roll) before expression & target rotation
    if conf_s[36] > thr and conf_s[45] > thr:
        src_roll = float(np.arctan2(pts_s[45, 1] - pts_s[36, 1], pts_s[45, 0] - pts_s[36, 0]))
    else:
        src_roll = 0.0
    cos_s, sin_s = np.cos(-src_roll), np.sin(-src_roll)
    unrot_s = np.array([[cos_s, -sin_s], [sin_s, cos_s]], dtype=np.float32)

    F_src_raw = (pts_s - c_src) / d_eyes_src
    F_src_norm = (unrot_s @ F_src_raw.T).T
    F_ret_norm = F_src_norm.copy()

    c_tgt = pts_t[30] if conf_t[30] > thr else np.mean(pts_t[conf_t > thr], axis=0)
    d_eyes_tgt = float(np.linalg.norm(pts_t[45] - pts_t[36])) if (conf_t[45] > thr and conf_t[36] > thr) else 40.0
    if d_eyes_tgt < 1.0: d_eyes_tgt = 40.0
    F_tgt_norm = (pts_t - c_tgt) / d_eyes_tgt

    # 1. Eyebrow Elevation & Furrow Transfer
    if conf_t[36] > thr and conf_t[45] > thr:
        for k in range(17, 22):
            if conf_t[k] > thr and conf_s[k] > thr:
                delta_brow_r = F_tgt_norm[k] - F_tgt_norm[36]
                F_ret_norm[k] = F_src_norm[36] + delta_brow_r
        for k in range(22, 27):
            if conf_t[k] > thr and conf_s[k] > thr:
                delta_brow_l = F_tgt_norm[k] - F_tgt_norm[45]
                F_ret_norm[k] = F_src_norm[45] + delta_brow_l

    # 2. Eye Openness / Blinking Transfer
    if np.all(conf_t[36:42] > thr):
        ear_r_tgt = (np.linalg.norm(F_tgt_norm[37] - F_tgt_norm[41]) + np.linalg.norm(F_tgt_norm[38] - F_tgt_norm[40])) / (2.0 * np.linalg.norm(F_tgt_norm[36] - F_tgt_norm[39]) + 1e-5)
        s_eye_r = float(np.clip(ear_r_tgt / 0.22, 0.05, 1.8))
        c_eye_r = (F_src_norm[36] + F_src_norm[39]) * 0.5
        for k in (37, 38, 40, 41):
            F_ret_norm[k, 1] = c_eye_r[1] + (F_src_norm[k, 1] - c_eye_r[1]) * s_eye_r

    if np.all(conf_t[42:48] > thr):
        ear_l_tgt = (np.linalg.norm(F_tgt_norm[43] - F_tgt_norm[47]) + np.linalg.norm(F_tgt_norm[44] - F_tgt_norm[46])) / (2.0 * np.linalg.norm(F_tgt_norm[42] - F_tgt_norm[45]) + 1e-5)
        s_eye_l = float(np.clip(ear_l_tgt / 0.22, 0.05, 1.8))
        c_eye_l = (F_src_norm[42] + F_src_norm[45]) * 0.5
        for k in (43, 44, 46, 47):
            F_ret_norm[k, 1] = c_eye_l[1] + (F_src_norm[k, 1] - c_eye_l[1]) * s_eye_l

    # 3. Mouth Expression & Smile Transfer
    if conf_t[48] > thr and conf_t[54] > thr and conf_s[48] > thr and conf_s[54] > thr:
        w_tgt = float(np.linalg.norm(F_tgt_norm[54] - F_tgt_norm[48]))
        w_src = float(np.linalg.norm(F_src_norm[54] - F_src_norm[48]))
        h_tgt = float(np.linalg.norm(F_tgt_norm[57] - F_tgt_norm[51]))
        h_src = float(np.linalg.norm(F_src_norm[57] - F_src_norm[51]))

        s_w = float(np.clip(w_tgt / (w_src + 1e-5), 0.7, 1.4))
        s_h = float(np.clip(h_tgt / (h_src + 1e-5), 0.5, 3.0))

        c_mouth_src = (F_src_norm[48] + F_src_norm[54]) * 0.5
        c_mouth_tgt = (F_tgt_norm[48] + F_tgt_norm[54]) * 0.5
        y_smile_delta = (c_mouth_tgt[1] - (F_tgt_norm[48, 1] + F_tgt_norm[54, 1]) * 0.5)

        for k in range(48, 68):
            dx = F_src_norm[k, 0] - c_mouth_src[0]
            dy = F_src_norm[k, 1] - c_mouth_src[1]
            F_ret_norm[k, 0] = c_mouth_src[0] + dx * s_w
            F_ret_norm[k, 1] = c_mouth_src[1] + dy * s_h
            if k in (48, 54):
                F_ret_norm[k, 1] -= y_smile_delta

        if h_tgt > 0.15:
            drop = (h_tgt - 0.10) * 0.5
            for k in (65, 66, 67):
                F_ret_norm[k, 1] += drop

    # Rigid Head Rotation & Placement
    scale_f = head_scale
    cos_a = np.cos(head_angle)
    sin_a = np.sin(head_angle)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    for i in range(68):
        # Target landmark visibility governs output landmark rendering, supplemented by source identity face landmarks
        c_i = conf_t[i]
        c_s = conf_s[i] if src_has_face else 0.0
        if c_i > thr or c_s > thr:
            rotated = rot_mat @ (F_ret_norm[i] * scale_f)
            pts_out[i] = nose_pt + rotated
            conf_out[i] = max(c_i, c_s)

    # Synchronize retargeted body keypoints with computed face landmarks for perfect alignment
    if conf_out[30] > thr and np.sum(conf_out > thr) >= 10:
        kps_ret[0] = pts_out[30]
        conf_ret[0] = max(conf_ret[0], conf_out[30])
    if np.all(conf_out[36:42] > thr):
        kps_ret[15] = np.mean(pts_out[36:42], axis=0)
        conf_ret[15] = max(conf_ret[15], float(np.mean(conf_out[36:42])))
    if np.all(conf_out[42:48] > thr):
        kps_ret[16] = np.mean(pts_out[42:48], axis=0)
        conf_ret[16] = max(conf_ret[16], float(np.mean(conf_out[42:48])))
    if conf_out[0] > thr:
        kps_ret[17] = pts_out[0]
        conf_ret[17] = max(conf_ret[17], conf_out[0])
    if conf_out[16] > thr:
        kps_ret[18] = pts_out[16]
        conf_ret[18] = max(conf_ret[18], conf_out[16])

    return pts_out, conf_out


def _retarget_hand_dwpose(
    hand_tgt: Optional[tuple[np.ndarray, np.ndarray]],
    hand_src: Optional[tuple[np.ndarray, np.ndarray]],
    wrist_pt: np.ndarray,
    elbow_pt: np.ndarray,
    wrist_src_pt: Optional[np.ndarray] = None,
    elbow_src_pt: Optional[np.ndarray] = None,
    canvas_wh: tuple[int, int] = (512, 512),
    thr: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Attaches Target hand gesture (21 keypoints) onto the retargeted wrist joint,
    strictly preserving the Source Person's individual hand-to-forearm proportion ratio.
    Prevents tiny hand shrinkage under 2D perspective foreshortening.
    """
    pts_out = np.zeros((21, 2), dtype=np.float32)
    conf_out = np.zeros(21, dtype=np.float32)

    if hand_tgt is None or hand_tgt[0] is None or np.sum(hand_tgt[1] > thr) < 4:
        return pts_out, conf_out

    pts_t, conf_t = hand_tgt
    root_t = pts_t[0]

    measured_farm_2d = float(np.linalg.norm(wrist_pt - elbow_pt))
    src_farm_2d = float(np.linalg.norm(wrist_src_pt - elbow_src_pt)) if (wrist_src_pt is not None and elbow_src_pt is not None) else 0.0
    forearm_len = max(measured_farm_2d, src_farm_2d * 0.85, 45.0)

    # 1. Determine Source Person's authentic hand-to-forearm ratio if available
    hand_to_forearm_ratio = 0.45  # Anatomical baseline (hand length ≈ 45% of forearm)
    if (
        hand_src is not None
        and hand_src[0] is not None
        and wrist_src_pt is not None
        and elbow_src_pt is not None
        and src_farm_2d > 10.0
    ):
        pts_s, conf_s = hand_src
        if len(pts_s) > 12 and conf_s[12] > thr:
            src_hand_len = float(np.linalg.norm(pts_s[12] - pts_s[0]))
        elif len(pts_s) > 9 and conf_s[9] > thr:
            src_hand_len = float(np.linalg.norm(pts_s[9] - pts_s[0])) * 1.85
        else:
            src_hand_len = 0.0
        if src_hand_len > 5.0:
            measured_ratio = src_hand_len / src_farm_2d
            hand_to_forearm_ratio = float(np.clip(measured_ratio, 0.35, 0.55))

    # 2. Target hand gesture span
    if len(pts_t) > 12 and conf_t[12] > thr:
        hand_span_t = float(np.linalg.norm(pts_t[12] - pts_t[0]))
    elif len(pts_t) > 9 and conf_t[9] > thr:
        hand_span_t = float(np.linalg.norm(pts_t[9] - pts_t[0])) * 1.85
    else:
        hand_span_t = 50.0
    if hand_span_t < 1.0:
        hand_span_t = 50.0

    target_hand_len = forearm_len * hand_to_forearm_ratio
    scale = target_hand_len / hand_span_t

    for i in range(21):
        if conf_t[i] > thr:
            rel = pts_t[i] - root_t
            pts_out[i] = wrist_pt + rel * scale
            conf_out[i] = conf_t[i]

    return pts_out, conf_out


_SAPIENS_308_LINKS = (
    (13, 11), (11, 9), (14, 12), (12, 10), (9, 10), (5, 9), (6, 10), (5, 6), (5, 7), (6, 8),
    (7, 62), (8, 41), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (13, 15),
    (13, 16), (13, 17), (14, 18), (14, 19), (14, 20), (62, 45), (45, 44), (44, 43), (43, 42),
    (62, 49), (49, 48), (48, 47), (47, 46), (62, 53), (53, 52), (52, 51), (51, 50), (62, 57),
    (57, 56), (56, 55), (55, 54), (62, 61), (61, 60), (60, 59), (59, 58), (41, 24), (24, 23),
    (23, 22), (22, 21), (41, 28), (28, 27), (27, 26), (26, 25), (41, 32), (32, 31), (31, 30),
    (30, 29), (41, 36), (36, 35), (35, 34), (34, 33), (41, 40), (40, 39), (39, 38), (38, 37),
)

_SAPIENS_308_LINK_COLORS = (
    (0, 255, 0), (0, 255, 0), (255, 128, 0), (255, 128, 0), (51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255),
    (0, 255, 0), (255, 128, 0), (0, 255, 0), (255, 128, 0), (51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255),
    (51, 153, 255), (51, 153, 255), (51, 153, 255), (0, 255, 0), (0, 255, 0), (0, 255, 0), (255, 128, 0), (255, 128, 0),
    (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 153, 255), (255, 153, 255), (255, 153, 255),
    (255, 153, 255), (102, 178, 255), (102, 178, 255), (102, 178, 255), (102, 178, 255), (255, 51, 51), (255, 51, 51), (255, 51, 51),
    (255, 51, 51), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0),
    (255, 128, 0), (255, 153, 255), (255, 153, 255), (255, 153, 255), (255, 153, 255), (102, 178, 255), (102, 178, 255), (102, 178, 255),
    (102, 178, 255), (255, 51, 51), (255, 51, 51), (255, 51, 51), (255, 51, 51), (0, 255, 0), (0, 255, 0), (0, 255, 0),
    (0, 255, 0),
)


_DWPOSE_BODY_EDGES = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), # arms
    (1, 9), (9, 10), (10, 11),                       # right leg: Neck -> RHip -> RKnee -> RAnkle
    (1, 12), (12, 13), (13, 14),                     # left leg: Neck -> LHip -> LKnee -> LAnkle
    (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),   # head: Neck -> Nose -> REye -> REar, Nose -> LEye -> LEar
    # Feet from BODY_25 ONLY (attached to ankles):
    (11, 22), (22, 23), (11, 24),                   # right foot: RAnkle -> RBigToe -> RSmallToe, RAnkle -> RHeel
    (14, 19), (19, 20), (14, 21),                   # left foot: LAnkle -> LBigToe -> LSmallToe, LAnkle -> LHeel
)


_SAPIENS_POSE_COLORS = (
    (255, 0, 85),
    (255, 85, 0),
    (255, 170, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 170, 255),
    (0, 85, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 170),
)


def _draw_pose_sapiens(
    canvas: np.ndarray,
    triples: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    threshold: float,
    radius: int = 3,
    thickness: int = 3,
    show_points: bool = True,
    show_skeleton: bool = True,
) -> None:
    height, width = canvas.shape[:2]
    colors = _SAPIENS_POSE_COLORS
    if show_skeleton:
        for index, (left, right) in enumerate(edges):
            if left >= len(triples) or right >= len(triples):
                continue
            a = triples[left]
            b = triples[right]
            if a[2] < threshold or b[2] < threshold:
                continue
            ax, ay = int(round(float(a[0]))), int(round(float(a[1])))
            bx, by = int(round(float(b[0]))), int(round(float(b[1])))
            if 0 <= ax < width and 0 <= ay < height and 0 <= bx < width and 0 <= by < height:
                cv2.line(canvas, (ax, ay), (bx, by), colors[index % len(colors)], max(1, int(thickness)), lineType=cv2.LINE_AA)
    if show_points and radius > 0:
        for index, point in enumerate(triples):
            if point[2] < threshold:
                continue
            x, y = int(round(float(point[0]))), int(round(float(point[1])))
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(canvas, (x, y), max(1, int(radius)), colors[index % len(colors)], -1, lineType=cv2.LINE_AA)


def _draw_points_sapiens(
    canvas: np.ndarray,
    triples: np.ndarray,
    threshold: float,
    radius: int = 1,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    height, width = canvas.shape[:2]
    for point in triples:
        if point[2] < threshold:
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (x, y), max(1, int(radius)), color, -1, lineType=cv2.LINE_AA)


def _draw_sapiens308(
    canvas: np.ndarray,
    sapiens_data: Any,
    thr: float = 0.15,
    thickness: int = 2,
    radius: int = 2,
    **kwargs: Any,
) -> None:
    if "threshold" in kwargs:
        thr = float(kwargs["threshold"])
    canvas_c = np.ascontiguousarray(canvas)
    H, W = canvas_c.shape[:2]
    if isinstance(sapiens_data, tuple):
        pts, conf = sapiens_data
    elif isinstance(sapiens_data, np.ndarray) and "conf" in kwargs:
        pts, conf = sapiens_data, kwargs["conf"]
    else:
        return
    colors = _SAPIENS_POSE_COLORS

    # 1. Draw Skeleton Links (65 links connecting Sapiens body keypoints)
    for link_idx, (src, dst) in enumerate(_SAPIENS_308_LINKS):
        if src >= len(pts) or dst >= len(pts):
            continue
        if conf[src] < thr or conf[dst] < thr:
            continue
        x1, y1 = int(round(float(pts[src, 0]))), int(round(float(pts[src, 1])))
        x2, y2 = int(round(float(pts[dst, 0]))), int(round(float(pts[dst, 1])))
        if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
            col = _SAPIENS_308_LINK_COLORS[link_idx % len(_SAPIENS_308_LINK_COLORS)]
            cv2.line(canvas_c, (x1, y1), (x2, y2), col, max(1, thickness), cv2.LINE_AA)

    # 2. Draw Keypoints
    if radius > 0:
        for idx in range(len(pts)):
            if conf[idx] > thr:
                x = int(round(float(pts[idx, 0])))
                y = int(round(float(pts[idx, 1])))
                if 0 <= x < W and 0 <= y < H:
                    col = colors[idx % len(colors)]
                    cv2.circle(canvas_c, (x, y), max(1, radius), col, -1, cv2.LINE_AA)


def _draw_dwpose_full(
    canvas: np.ndarray,
    body_kps: np.ndarray,
    body_conf: np.ndarray,
    face_data: Optional[tuple[np.ndarray, np.ndarray]],
    lhand_data: Optional[tuple[np.ndarray, np.ndarray]],
    rhand_data: Optional[tuple[np.ndarray, np.ndarray]],
    thr: float = 0.15,
    line_thickness: int = 4,
    point_radius: int = 4,
) -> None:
    """
    Renders DWPose matching the Sapiens2 Pose node style:
      - COCO 18 body connections (2-line torso from Neck to Hips, no pelvis bar)
      - Feet from BODY_25 attached to ankles
      - Head lines (Neck -> Nose -> Eyes -> Ears) perfectly aligned with face landmarks
      - Left and Right hands (21 keypoints each)
      - Face 68 landmarks
    """
    canvas_c = np.ascontiguousarray(canvas)

    has_face = face_data is not None and face_data[0] is not None and np.sum(face_data[1] > thr) >= 10

    # Ensure body eye & ear keypoints align with face landmarks and ears are never placed at 0,0
    draw_kps = body_kps.copy()
    draw_conf = body_conf.copy()
    if has_face:
        f_pts, f_conf = face_data
        if f_conf[30] > thr and f_pts[30, 0] > 10.0:
            draw_kps[0] = f_pts[30]
            draw_conf[0] = max(draw_conf[0], f_conf[30])
        if np.all(f_conf[36:42] > thr):
            draw_kps[15] = np.mean(f_pts[36:42], axis=0)
            draw_conf[15] = max(draw_conf[15], float(np.mean(f_conf[36:42])))
        if np.all(f_conf[42:48] > thr):
            draw_kps[16] = np.mean(f_pts[42:48], axis=0)
            draw_conf[16] = max(draw_conf[16], float(np.mean(f_conf[42:48])))
        if draw_conf[15] > thr and draw_conf[16] > thr:
            eye_y = (draw_kps[15, 1] + draw_kps[16, 1]) * 0.5
            ear_offset = float(np.linalg.norm(draw_kps[16] - draw_kps[15])) * 0.7
            draw_kps[17] = [draw_kps[15, 0] - ear_offset, eye_y + 2.0]
            draw_kps[18] = [draw_kps[16, 0] + ear_offset, eye_y + 2.0]
            draw_conf[17] = max(draw_conf[17], 0.9)
            draw_conf[18] = max(draw_conf[18], 0.9)

    # 1. Body with DWPose body edges and feet only
    body_triples = np.column_stack([draw_kps, draw_conf])
    _draw_pose_sapiens(
        canvas_c,
        body_triples,
        _DWPOSE_BODY_EDGES,
        threshold=thr,
        radius=point_radius,
        thickness=line_thickness,
        show_points=(point_radius > 0),
        show_skeleton=True,
    )

    # 2. Hands (left & right 21 keypoints) matching Sapiens2 Pose node
    hand_thick = max(1, line_thickness - 1)
    hand_rad = max(1, point_radius - 2) if point_radius > 0 else 0
    for hand_data in (lhand_data, rhand_data):
        if hand_data is not None and hand_data[0] is not None:
            pts, conf = hand_data
            if np.sum(conf > thr) >= 4:
                hand_triples = np.column_stack([pts, conf])
                _draw_pose_sapiens(
                    canvas_c,
                    hand_triples,
                    _HAND21_EDGES,
                    threshold=thr,
                    radius=hand_rad,
                    thickness=hand_thick,
                    show_points=(hand_rad > 0),
                    show_skeleton=True,
                )

    # 3. Face (68 landmarks) matching Sapiens2 Pose node
    if has_face:
        pts, conf = face_data
        face_triples = np.column_stack([pts, conf])
        face_rad = max(1, point_radius // 2) if point_radius > 0 else 1
        _draw_points_sapiens(canvas_c, face_triples, threshold=thr, radius=face_rad, color=(255, 255, 255))


def _draw_skel(canvas: np.ndarray, kps: np.ndarray, conf: np.ndarray,
               style: str, thr: float, thickness: int = 4, dot_r: int = 5) -> None:
    canvas_c = np.ascontiguousarray(canvas)
    for ei, (p, c) in enumerate(_BODY25_EDGES):
        if p >= len(kps) or c >= len(kps): continue
        if conf[p] < thr or conf[c] < thr: continue
        col = (200, 200, 200) if style == "wireframe" else _EDGE_COLORS[ei % len(_EDGE_COLORS)]
        cv2.line(canvas_c,
                 (int(kps[p, 0]), int(kps[p, 1])),
                 (int(kps[c, 0]), int(kps[c, 1])),
                 col, thickness, cv2.LINE_AA)
    if style != "wireframe" and dot_r > 0:
        for i in range(min(len(kps), 25)):
            if conf[i] < thr: continue
            col = _EDGE_COLORS[i % len(_EDGE_COLORS)]
            pt  = (int(kps[i, 0]), int(kps[i, 1]))
            cv2.circle(canvas_c, pt, dot_r,  col,   -1, cv2.LINE_AA)
            cv2.circle(canvas_c, pt, dot_r,  (255,255,255), 1, cv2.LINE_AA)


def _render_skeleton(
    kps: np.ndarray,
    conf: np.ndarray,
    canvas_hw: tuple[int, int],
    style: str,
    thr: float = 0.15,
    face_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    lhand_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    rhand_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    sapiens_308_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    line_thickness: int = 4,
    point_radius: int = 4,
) -> np.ndarray:
    H, W = canvas_hw
    canvas = np.zeros((H, W, 3), np.uint8)
    if style == "dwpose":
        _draw_dwpose_full(
            canvas, kps, conf, face_data, lhand_data, rhand_data, thr,
            line_thickness=line_thickness, point_radius=point_radius,
        )
    elif style == "sapiens_308" and sapiens_308_data is not None and sapiens_308_data[0] is not None:
        _draw_sapiens308(
            canvas, sapiens_308_data, thr=thr,
            thickness=line_thickness, radius=point_radius,
        )
    else:
        _draw_skel(canvas, kps, conf, style, thr, thickness=line_thickness, dot_r=point_radius)
    return canvas


def _overlay_skeleton(
    base_rgb: np.ndarray,
    kps: np.ndarray,
    conf: np.ndarray,
    style: str,
    thr: float = 0.15,
    face_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    lhand_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    rhand_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    sapiens_308_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    line_thickness: int = 4,
    point_radius: int = 4,
    skeleton_opacity: float = 1.0,
) -> np.ndarray:
    if skeleton_opacity <= 0.001:
        return base_rgb.copy()

    # Draw directly on a copy of base_rgb so OpenCV anti-aliased lines blend with background (no black borders)
    canvas_overlay = base_rgb.copy()
    if style == "sapiens_308" and sapiens_308_data is not None and sapiens_308_data[0] is not None:
        _draw_sapiens308(
            canvas_overlay, sapiens_308_data,
            thr=thr, radius=point_radius, thickness=line_thickness,
        )
    elif style == "dwpose":
        _draw_dwpose_full(
            canvas_overlay, kps, conf,
            face_data=face_data, lhand_data=lhand_data, rhand_data=rhand_data,
            thr=thr, line_thickness=line_thickness, point_radius=point_radius,
        )
    else:
        _draw_skel(
            canvas_overlay, kps, conf, style, thr,
            thickness=line_thickness, dot_r=point_radius,
        )

    if skeleton_opacity >= 0.999:
        return canvas_overlay

    alpha = float(np.clip(skeleton_opacity, 0.0, 1.0))
    return cv2.addWeighted(canvas_overlay, alpha, base_rgb, 1.0 - alpha, 0.0)


def _comparison(img_src, skel_src, skel_tgt, skel_ret) -> np.ndarray:
    target_h = max(img_src.shape[0], skel_src.shape[0], skel_tgt.shape[0], skel_ret.shape[0])
    def rsz(im):
        f = target_h / max(im.shape[0], 1)
        return cv2.resize(im, (max(1, int(im.shape[1] * f)), target_h))
    divider = np.full((target_h, 4, 3), 60, np.uint8)
    panels  = [rsz(img_src), divider, rsz(skel_src), divider, rsz(skel_tgt), divider, rsz(skel_ret)]
    return np.concatenate(panels, axis=1)


def _to_comfy(img_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)


def _to_openpose_dict(
    kps: np.ndarray,
    conf: np.ndarray,
    canvas_wh: tuple,
    face_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    lhand_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    rhand_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    sapiens_308_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    extra_meta: Optional[dict] = None,
) -> dict:
    W, H = canvas_wh
    def to_flat(pts, conf_arr, count):
        if pts is None or conf_arr is None:
            return []
        flat = []
        for i in range(min(len(pts), count)):
            flat += [float(pts[i, 0]), float(pts[i, 1]), float(conf_arr[i]) if i < len(conf_arr) else 0.0]
        while len(flat) < count * 3:
            flat += [0.0, 0.0, 0.0]
        return flat

    flat_body = []
    for i in range(min(len(kps), 25)):
        flat_body += [float(kps[i, 0]), float(kps[i, 1]), float(conf[i]) if i < len(conf) else 0.0]
    while len(flat_body) < 75:
        flat_body += [0.0, 0.0, 0.0]

    person = {
        "person_id": [-1],
        "pose_keypoints_2d": flat_body,
        "face_keypoints_2d": to_flat(face_data[0], face_data[1], 68) if (face_data and face_data[0] is not None) else [],
        "hand_left_keypoints_2d": to_flat(lhand_data[0], lhand_data[1], 21) if (lhand_data and lhand_data[0] is not None) else [],
        "hand_right_keypoints_2d": to_flat(rhand_data[0], rhand_data[1], 21) if (rhand_data and rhand_data[0] is not None) else [],
    }
    if sapiens_308_data is not None and sapiens_308_data[0] is not None:
        person["sapiens_keypoints_2d"] = to_flat(sapiens_308_data[0], sapiens_308_data[1], 308)

    payload = {
        "version": 1.3,
        "canvas_width": int(W),
        "canvas_height": int(H),
        "people": [person],
    }
    if extra_meta:
        payload["sapiens_meta"] = extra_meta
    return payload


def _to_openpose_json(
    kps: np.ndarray,
    conf: np.ndarray,
    canvas_wh: tuple,
    face_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    lhand_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    rhand_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    sapiens_308_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    extra_meta: Optional[dict] = None,
) -> str:
    payload = _to_openpose_dict(kps, conf, canvas_wh, face_data, lhand_data, rhand_data, sapiens_308_data, extra_meta)
    return json.dumps(payload, ensure_ascii=True)


_GROUNDING_MODES = [
    "bottom_center",
    "bottom_left",
    "bottom_right",
    "top_center",
    "top_left",
    "top_right",
    "left_center",
    "right_center",
    "center",
    "none",
]
_SKEL_STYLES     = ["openpose_classic", "dwpose", "sapiens_308", "wireframe"]
_CANVAS_MODES    = ["match_target_image", "match_source_image",
                    "square_512", "square_768", "square_1024"]


class Sapiens2PoseRenderConfig:
    """
    Modular configuration node for customizing OpenPose / DWPose / Sapiens2
    skeleton rendering appearance (line thickness, point radius, opacity, and threshold).
    Plug into Sapiens2PoseRetarget or Sapiens2PoseToTPose.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "line_thickness": (
                    "INT",
                    {"default": 4, "min": 1, "max": 32, "step": 1, "tooltip": "Thickness of skeleton bone lines in pixels."},
                ),
                "point_radius": (
                    "INT",
                    {"default": 4, "min": 0, "max": 32, "step": 1, "tooltip": "Radius of joint keypoint dots in pixels (0 = hide dots)."},
                ),
                "skeleton_opacity": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Opacity of skeleton for overlay image (0.0 = transparent, 1.0 = solid)."},
                ),
                "keypoint_threshold": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Confidence threshold to display keypoints."},
                ),
            }
        }

    RETURN_TYPES = ("SAPIENS2_POSE_CONFIG",)
    RETURN_NAMES = ("render_config",)
    FUNCTION = "get_config"
    CATEGORY = "Sapiens2"

    def get_config(
        self,
        line_thickness: int = 4,
        point_radius: int = 4,
        skeleton_opacity: float = 1.0,
        keypoint_threshold: float = 0.05,
    ) -> tuple[dict[str, Any]]:
        return ({
            "line_thickness": line_thickness,
            "point_radius": point_radius,
            "skeleton_opacity": skeleton_opacity,
            "keypoint_threshold": keypoint_threshold,
        },)


def _detect_skeleton_style(
    target_name: Optional[str] = None,
    sapiens_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    face_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    lhand_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    rhand_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    body_kps_count: int = 25,
) -> str:
    """
    Automatically detects whether an input skeleton is Sapiens 308, DWPose, or OpenPose Classic.
    Strictly respects the target mode selected in Sapiens2Pose.
    """
    if target_name:
        t = str(target_name).lower().strip()
        if "308" in t or t == "sapiens_308":
            return "sapiens_308"
        if "dwpose" in t:
            return "dwpose"
        if "coco" in t or "body_25" in t or "classic" in t:
            return "openpose_classic"

    # Fallback inspection of component keypoints
    has_face = face_data is not None and face_data[0] is not None and np.sum(face_data[1] > 0.05) >= 4
    has_lhand = lhand_data is not None and lhand_data[0] is not None and np.sum(lhand_data[1] > 0.05) >= 4
    has_rhand = rhand_data is not None and rhand_data[0] is not None and np.sum(rhand_data[1] > 0.05) >= 4

    if has_face or has_lhand or has_rhand:
        return "dwpose"

    if body_kps_count >= 308:
        return "sapiens_308"

    return "openpose_classic"


def _sanitize_ratios(ratios: dict[str, float]) -> dict[str, float]:
    """
    Clamps raw proportion ratios extracted from any pose (seated, cropped, foreshortened)
    into anthropometrically invariant human bounds.
    """
    r = ratios.copy()
    r["r_torso"]         = float(np.clip(r.get("r_torso", 0.285),       0.25, 0.32))
    r["r_thigh"]         = float(np.clip(r.get("r_thigh", 0.265),        0.23, 0.29))
    r["r_shin"]          = float(np.clip(r.get("r_shin",  0.245),         0.21, 0.27))
    r["r_upper_arm"]     = float(np.clip(r.get("r_upper_arm", 0.17),     0.14, 0.20))
    r["r_forearm"]       = float(np.clip(r.get("r_forearm", 0.15),       0.12, 0.18))
    r["r_shoulder_span"] = float(np.clip(r.get("r_shoulder_span", 0.11), 0.08, 0.14))
    r["r_hip_span"]      = float(np.clip(r.get("r_hip_span", 0.075),     0.05, 0.10))
    r["r_neck_nose"]     = float(np.clip(r.get("r_neck_nose", 0.08),     0.06, 0.10))
    r["r_eye_span"]      = float(np.clip(r.get("r_eye_span", 0.035),     0.02, 0.05))
    r["r_ear_span"]      = float(np.clip(r.get("r_ear_span", 0.075),     0.05, 0.10))
    r["r_foot_len"]      = float(np.clip(r.get("r_foot_len", 0.035),     0.02, 0.05))
    return r


def _build_canonical_tpose_sapiens308(
    tpose_kps: np.ndarray,
    ratios: dict[str, float],
    canvas_wh: tuple[int, int],
    sapiens_src_data: Optional[tuple[np.ndarray, np.ndarray]],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Constructs an authentic 308-keypoint canonical T-Pose representation with symmetric limb layout.
    """
    if sapiens_src_data is None or sapiens_src_data[0] is None:
        return None
    pts_s, conf_s = sapiens_src_data
    if len(pts_s) < 308:
        return None

    pts_out = np.zeros((308, 2), dtype=np.float32)
    conf_out = conf_s.copy()

    # Exact Sapiens 308 keypoint mapping to T-Pose body joints
    joint_map = {
        0: 0,    # nose -> Nose (0)
        1: 16,   # left_eye -> LEye (16)
        2: 15,   # right_eye -> REye (15)
        3: 18,   # left_ear -> LEar (18)
        4: 17,   # right_ear -> REar (17)
        5: 5,    # left_shoulder -> LShoulder (5)
        6: 2,    # right_shoulder -> RShoulder (2)
        7: 6,    # left_elbow -> LElbow (6)
        8: 3,    # right_elbow -> RElbow (3)
        9: 12,   # left_hip -> LHip (12)
        10: 9,   # right_hip -> RHip (9)
        11: 13,  # left_knee -> LKnee (13)
        12: 10,  # right_knee -> RKnee (10)
        13: 14,  # left_ankle -> LAnkle (14)
        14: 11,  # right_ankle -> RAnkle (11)
        15: 19,  # left_big_toe -> LBigToe (19)
        16: 20,  # left_small_toe -> LSmallToe (20)
        17: 21,  # left_heel -> LHeel (21)
        18: 22,  # right_big_toe -> RBigToe (22)
        19: 23,  # right_small_toe -> RSmallToe (23)
        20: 24,  # right_heel -> RHeel (24)
        41: 4,   # right_wrist -> RWrist (4)
        62: 7,   # left_wrist -> LWrist (7)
        63: 6,   # left_olecranon -> LElbow (6)
        64: 3,   # right_olecranon -> RElbow (3)
        65: 14,  # left ankle joint -> LAnkle (14)
        66: 11,  # right ankle joint -> RAnkle (11)
        67: 5,   # left_acromion -> LShoulder (5)
        68: 2,   # right_acromion -> RShoulder (2)
        69: 1,   # neck -> Neck (1)
    }

    # Reference torso length in T-Pose
    torso_len = float(np.linalg.norm(tpose_kps[1] - tpose_kps[8])) if (tpose_kps[1, 1] > 0 and tpose_kps[8, 1] > 0) else 120.0
    face_scale = (torso_len * 0.18) / 35.0
    farm_len = float(np.linalg.norm(tpose_kps[4] - tpose_kps[2])) * 0.5
    hand_scale = (farm_len * 0.45) / 35.0

    for i in range(308):
        if i in joint_map:
            tj = joint_map[i]
            pts_out[i] = tpose_kps[tj]
        elif 21 <= i <= 40:
            # Right hand fingers in 308 (extending outward horizontally to the left along -X)
            rel = pts_s[i] - pts_s[41]
            pts_out[i] = tpose_kps[4] + np.array([-abs(rel[0]) * hand_scale - 4.0, (i - 30) * 1.5], dtype=np.float32)
        elif 42 <= i <= 61:
            # Left hand fingers in 308 (extending outward horizontally to the right along +X)
            rel = pts_s[i] - pts_s[62]
            pts_out[i] = tpose_kps[7] + np.array([abs(rel[0]) * hand_scale + 4.0, (i - 51) * 1.5], dtype=np.float32)
        elif 70 <= i <= 242:
            # Face contour & landmarks in 308 (relative to nose 0)
            rel = pts_s[i] - pts_s[0]
            pts_out[i] = tpose_kps[0] + rel * face_scale
        elif 243 <= i <= 267:
            # Torso & spine contour points (anchored between Neck and MidHip)
            fraction = float(i - 243) / 25.0
            spine_center = tpose_kps[1] * (1.0 - fraction) + tpose_kps[8] * fraction
            rel_x = (pts_s[i, 0] - pts_s[69, 0]) * (torso_len / 180.0)
            pts_out[i] = [spine_center[0] + rel_x, spine_center[1]]
        elif 268 <= i <= 287:
            # Left foot dense points (anchored to LAnkle 14)
            rel = pts_s[i] - pts_s[13]
            pts_out[i] = tpose_kps[14] + rel * 0.7
        elif 288 <= i <= 307:
            # Right foot dense points (anchored to RAnkle 11)
            rel = pts_s[i] - pts_s[14]
            pts_out[i] = tpose_kps[11] + rel * 0.7
        else:
            pts_out[i] = tpose_kps[1] + (pts_s[i] - pts_s[69]) * (torso_len / 180.0)

    return pts_out, conf_out


def _retarget_sapiens308(
    sapiens_src_data: Optional[tuple[np.ndarray, np.ndarray]],
    sapiens_tgt_data: Optional[tuple[np.ndarray, np.ndarray]],
    kps_ret: np.ndarray,
    conf_ret: np.ndarray,
    ratios_src: dict[str, float],
    canvas_wh: tuple[int, int],
    thr: float = 0.05,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Retargets full 308-keypoint skeleton applying source proportions, height ratio, and limb rotations.
    """
    data_to_use = sapiens_tgt_data if (sapiens_tgt_data is not None and sapiens_tgt_data[0] is not None and len(sapiens_tgt_data[0]) >= 308) else sapiens_src_data
    if data_to_use is None or data_to_use[0] is None or len(data_to_use[0]) < 308:
        return None

    pts_t, conf_t = data_to_use
    pts_out = np.zeros((308, 2), dtype=np.float32)
    conf_out = conf_t.copy()

    joint_map = {
        0: 0, 1: 16, 2: 15, 3: 18, 4: 17, 5: 5, 6: 2, 7: 6, 8: 3, 9: 12, 10: 9,
        11: 13, 12: 10, 13: 14, 14: 11, 15: 19, 16: 20, 17: 21, 18: 22, 19: 23, 20: 24,
        41: 4, 62: 7, 63: 6, 64: 3, 65: 14, 66: 11, 67: 5, 68: 2, 69: 1,
    }

    torso_ret_len = float(np.linalg.norm(kps_ret[1] - kps_ret[8])) if (conf_ret[1] > thr and conf_ret[8] > thr) else 120.0
    face_scale = max(torso_ret_len * 0.18, 14.0) / 35.0
    farm_len_r = float(np.linalg.norm(kps_ret[4] - kps_ret[3])) if (conf_ret[4] > thr and conf_ret[3] > thr) else 50.0
    farm_len_l = float(np.linalg.norm(kps_ret[7] - kps_ret[6])) if (conf_ret[7] > thr and conf_ret[6] > thr) else 50.0
    hand_scale_r = max(farm_len_r * 0.45, 20.0) / 45.0
    hand_scale_l = max(farm_len_l * 0.45, 20.0) / 45.0

    for i in range(308):
        if i in joint_map:
            tj = joint_map[i]
            pts_out[i] = kps_ret[tj]
        elif 21 <= i <= 40:
            rel = pts_t[i] - pts_t[41]
            pts_out[i] = kps_ret[4] + rel * hand_scale_r
        elif 42 <= i <= 61:
            rel = pts_t[i] - pts_t[62]
            pts_out[i] = kps_ret[7] + rel * hand_scale_l
        elif 70 <= i <= 242:
            rel = pts_t[i] - pts_t[0]
            pts_out[i] = kps_ret[0] + rel * face_scale
        elif 243 <= i <= 267:
            fraction = float(i - 243) / 25.0
            spine_center = kps_ret[1] * (1.0 - fraction) + kps_ret[8] * fraction
            rel_x = (pts_t[i, 0] - pts_t[69, 0]) * (torso_ret_len / 120.0)
            pts_out[i] = [spine_center[0] + rel_x, spine_center[1]]
        elif 268 <= i <= 287:
            rel = pts_t[i] - pts_t[13]
            pts_out[i] = kps_ret[14] + rel * 0.7
        elif 288 <= i <= 307:
            rel = pts_t[i] - pts_t[14]
            pts_out[i] = kps_ret[11] + rel * 0.7
        else:
            rel = pts_t[i] - pts_t[69]
            pts_out[i] = kps_ret[1] + rel * (torso_ret_len / 120.0)

    return pts_out, conf_out


class Sapiens2PoseRetarget:
    """
    Unified 3D & 2D Anatomical Pose Retargeting node with full DWPose & Facial Expression transfer.
    Transfers Target Person's pose, joint angles, and facial expressions onto Source Person
    while strictly preserving Source's body proportions, bone lengths, and facial identity.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_pose_json": ("STRING,POSE_KEYPOINT", {"default": "", "tooltip": "OpenPose / DWPose JSON or POSE_KEYPOINT of Source Person (or T-Pose)"}),
                "target_pose_json": ("STRING,POSE_KEYPOINT", {"default": "", "tooltip": "OpenPose / DWPose JSON or POSE_KEYPOINT of Target Person"}),
                "source_image":     ("IMAGE", {"tooltip": "Reference image of Source Person"}),
                "target_image":     ("IMAGE", {"tooltip": "Reference image of Target Person"}),
                "grounding_mode":   (_GROUNDING_MODES, {"default": "bottom_center", "tooltip": "Canvas spatial anchor for foot/ground alignment"}),
                "canvas_mode":      (_CANVAS_MODES,    {"default": "match_target_image", "tooltip": "Output canvas resolution"}),
                "use_source_tpose": ("BOOLEAN", {"default": False, "tooltip": "If True, transforms source person into canonical T-pose internally before retargeting."}),
            },
            "optional": {
                "render_config": (
                    "SAPIENS2_POSE_CONFIG",
                    {"tooltip": "Optional modular rendering configuration from Sapiens2PoseRenderConfig node (line thickness, point radius, opacity, threshold)."},
                ),
                "source_pointmap": (
                    "SAPIENS2_POINTMAP",
                    {"tooltip": "Optional 3D surface pointmap tensor from Sapiens2 Pointmap node for Source Person (highest quality 3D)."},
                ),
                "target_pointmap": (
                    "SAPIENS2_POINTMAP",
                    {"tooltip": "Optional 3D surface pointmap tensor from Sapiens2 Pointmap node for Target Person (highest quality 3D)."},
                ),
                "source_depth_map": (
                    "IMAGE",
                    {"tooltip": "Optional fallback 2D depth map. Not needed if 'source_pointmap' is connected."},
                ),
                "target_depth_map": (
                    "IMAGE",
                    {"tooltip": "Optional fallback 2D depth map. Not needed if 'target_pointmap' is connected."},
                ),
                "source_height_cm": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 300.0, "step": 0.5,
                        "tooltip": "Known Source person height in cm. (0 = auto match scene scale or use metadata from T-Pose node)",
                    },
                ),
                "target_height_cm": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 300.0, "step": 0.5,
                        "tooltip": "Known Target person height in cm. (0 = auto match scene scale)",
                    },
                ),
                "camera_fov_deg": (
                    "FLOAT",
                    {
                        "default": 60.0, "min": 10.0, "max": 150.0, "step": 1.0,
                        "tooltip": "Approximate camera horizontal FOV in degrees for pinhole unprojection.",
                    },
                ),
            },
        }

    RETURN_TYPES  = ("STRING", "IMAGE", "IMAGE", "IMAGE", "FLOAT", "FLOAT", "STRING", "POSE_KEYPOINT")
    RETURN_NAMES  = (
        "openpose_json",
        "skeleton_image",
        "skeleton_overlay",
        "comparison_image",
        "source_height_cm",
        "target_height_cm",
        "limb_scale_info",
        "pose_keypoint",
    )
    FUNCTION = "retarget"
    CATEGORY = "Sapiens2"
    def retarget(
        self,
        source_pose_json: Any,
        target_pose_json: Any,
        source_image: torch.Tensor,
        target_image: torch.Tensor,
        grounding_mode: str  = "bottom_center",
        canvas_mode: str     = "match_target_image",
        use_source_tpose: bool = False,
        render_config: Optional[dict[str, Any]] = None,
        source_pointmap: Optional[Any] = None,
        target_pointmap: Optional[Any] = None,
        source_depth_map: Optional[torch.Tensor] = None,
        target_depth_map: Optional[torch.Tensor] = None,
        source_height_cm: float = 0.0,
        target_height_cm: float = 0.0,
        camera_fov_deg: float = 60.0,
    ):
        if render_config is not None and isinstance(render_config, dict):
            line_thickness = render_config.get("line_thickness", 4)
            point_radius = render_config.get("point_radius", 4)
            skeleton_opacity = render_config.get("skeleton_opacity", 1.0)
            keypoint_threshold = render_config.get("keypoint_threshold", 0.05)
        else:
            line_thickness = 4
            point_radius = 4
            skeleton_opacity = 1.0
            keypoint_threshold = 0.05

        kps_src, conf_src, meta_src, face_src, lhand_src, rhand_src, sapiens_src = _parse_json(source_pose_json)
        kps_tgt, conf_tgt, meta_tgt, face_tgt, lhand_tgt, rhand_tgt, sapiens_tgt = _parse_json(target_pose_json)
        if kps_src is None:
            raise ValueError("[Sapiens2 Retarget] source_pose_json is empty or invalid.")
        if kps_tgt is None:
            raise ValueError("[Sapiens2 Retarget] target_pose_json is empty or invalid.")
        kps_src, conf_src = _pad_to(kps_src, conf_src)
        kps_tgt, conf_tgt = _pad_to(kps_tgt, conf_tgt)
        # Automatically detect skeleton style from input data
        skeleton_style = _detect_skeleton_style(
            target_name=meta_tgt.get("target") or meta_src.get("target"),
            sapiens_data=sapiens_tgt or sapiens_src,
            face_data=face_tgt or face_src,
            lhand_data=lhand_tgt or lhand_src,
            rhand_data=rhand_tgt or rhand_src,
            body_kps_count=len(kps_tgt),
        )

        img_src_np = (source_image[0].cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
        img_tgt_np = (target_image[0].cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
        H_src, W_src = img_src_np.shape[:2]
        H_tgt, W_tgt = img_tgt_np.shape[:2]

        cW, cH = {
            "match_target_image": (W_tgt, H_tgt),
            "match_source_image": (W_src, H_src),
            "square_512":         (512, 512),
            "square_768":         (768, 768),
            "square_1024":        (1024, 1024),
        }.get(canvas_mode, (W_tgt, H_tgt))

        # 3D keypoint construction (Pointmap > Pinhole Unprojection > 2D fallback)
        kps_3d_src, mode_src = _build_kps_3d(kps_src, conf_src, source_pointmap, source_depth_map, (H_src, W_src), camera_fov_deg)
        kps_3d_tgt, mode_tgt = _build_kps_3d(kps_tgt, conf_tgt, target_pointmap, target_depth_map, (H_tgt, W_tgt), camera_fov_deg)

        # Depth extraction (for fallback heuristic if kps_3d is None)
        depth_src_arr = _extract_depth_map_array(source_depth_map, (H_src, W_src))
        depth_tgt_arr = _extract_depth_map_array(target_depth_map, (H_tgt, W_tgt))

        # Proportions of Source and Target (sanitized to invariant anthropometric bounds)
        raw_ratios_src = _extract_proportions(kps_src, conf_src, depth_src_arr, (H_src, W_src), kps_3d=kps_3d_src)
        if meta_src and "proportion_ratios" in meta_src:
            for k, v in meta_src["proportion_ratios"].items():
                raw_ratios_src[k] = float(v)
        ratios_src = _sanitize_ratios(raw_ratios_src)

        raw_ratios_tgt = _extract_proportions(kps_tgt, conf_tgt, depth_tgt_arr, (H_tgt, W_tgt), kps_3d=kps_3d_tgt)
        ratios_tgt = _sanitize_ratios(raw_ratios_tgt)

        # Height calibration (cm)
        h_src_input = float(source_height_cm)
        if h_src_input <= 0 and meta_src and "measured_height_cm" in meta_src:
            h_src_input = float(meta_src["measured_height_cm"])

        h_tgt_input = float(target_height_cm)
        if h_tgt_input <= 0 and meta_tgt and "measured_height_cm" in meta_tgt:
            h_tgt_input = float(meta_tgt["measured_height_cm"])

        if h_src_input > 0 and h_tgt_input > 0:
            height_ratio = h_src_input / h_tgt_input
            h_src_display = h_src_input
            h_tgt_display = h_tgt_input
        elif h_src_input > 0 or h_tgt_input > 0:
            h_src_display = h_src_input if h_src_input > 0 else _estimate_biometric_height(kps_src, conf_src, ratios_src["total_height"], kps_3d=kps_3d_src)
            h_tgt_display = h_tgt_input if h_tgt_input > 0 else _estimate_biometric_height(kps_tgt, conf_tgt, ratios_tgt["total_height"], kps_3d=kps_3d_tgt)
            height_ratio = float(np.clip(h_src_display / max(h_tgt_display, 50.0), 0.5, 2.0))
        else:
            h_src_display = _estimate_biometric_height(kps_src, conf_src, ratios_src["total_height"], kps_3d=kps_3d_src)
            h_tgt_display = _estimate_biometric_height(kps_tgt, conf_tgt, ratios_tgt["total_height"], kps_3d=kps_3d_tgt)
            height_ratio = float(np.clip(h_src_display / max(h_tgt_display, 50.0), 0.5, 2.0))

        # Retarget kinematics
        kps_ret, conf_ret = _retarget_kinematics(
            ratios_source=ratios_src,
            ratios_target=ratios_tgt,
            kps_target=kps_tgt,
            conf_target=conf_tgt,
            height_ratio=height_ratio,
            grounding=grounding_mode,
            canvas_wh=(cW, cH),
            thr=keypoint_threshold,
        )

        # Zero out confidence of any joints positioned outside canvas (prevents bottom-border accumulation)
        for i in range(len(kps_ret)):
            if kps_ret[i, 0] < 0 or kps_ret[i, 0] >= cW or kps_ret[i, 1] < 0 or kps_ret[i, 1] >= cH:
                conf_ret[i] = 0.0

        # DWPose Face Expression Retargeting (Source Identity + Target Expression & Visibility Gating)
        face_ret_pts, face_ret_conf = _retarget_face_dwpose(
            face_src=face_src,
            face_tgt=face_tgt,
            kps_ret=kps_ret,
            conf_ret=conf_ret,
            canvas_wh=(cW, cH),
            thr=keypoint_threshold,
        )
        face_ret_data = (face_ret_pts, face_ret_conf) if np.sum(face_ret_conf > keypoint_threshold) >= 6 else None

        # DWPose Hand Retargeting (Target Gesture attached to Retargeted Wrists, Source Hand Scale Preserved)
        lhand_ret_pts, lhand_ret_conf = _retarget_hand_dwpose(
            hand_tgt=lhand_tgt,
            hand_src=lhand_src,
            wrist_pt=kps_ret[7],
            elbow_pt=kps_ret[6],
            wrist_src_pt=kps_src[7] if conf_src[7] > keypoint_threshold else None,
            elbow_src_pt=kps_src[6] if conf_src[6] > keypoint_threshold else None,
            canvas_wh=(cW, cH),
            thr=keypoint_threshold,
        )
        lhand_ret_data = (lhand_ret_pts, lhand_ret_conf) if np.sum(lhand_ret_conf > keypoint_threshold) >= 4 else None

        rhand_ret_pts, rhand_ret_conf = _retarget_hand_dwpose(
            hand_tgt=rhand_tgt,
            hand_src=rhand_src,
            wrist_pt=kps_ret[4],
            elbow_pt=kps_ret[3],
            wrist_src_pt=kps_src[4] if conf_src[4] > keypoint_threshold else None,
            elbow_src_pt=kps_src[3] if conf_src[3] > keypoint_threshold else None,
            canvas_wh=(cW, cH),
            thr=keypoint_threshold,
        )
        rhand_ret_data = (rhand_ret_pts, rhand_ret_conf) if np.sum(rhand_ret_conf > keypoint_threshold) >= 4 else None

        # Sapiens 308 Full Retargeting
        sapiens_ret_data = _retarget_sapiens308(
            sapiens_src_data=sapiens_src,
            sapiens_tgt_data=sapiens_tgt,
            kps_ret=kps_ret,
            conf_ret=conf_ret,
            ratios_src=ratios_src,
            canvas_wh=(cW, cH),
            thr=keypoint_threshold,
        )

        # Renders
        skel_img = _render_skeleton(
            kps_ret, conf_ret, (cH, cW), skeleton_style,
            thr=keypoint_threshold,
            face_data=face_ret_data, lhand_data=lhand_ret_data, rhand_data=rhand_ret_data,
            sapiens_308_data=sapiens_ret_data,
            line_thickness=line_thickness, point_radius=point_radius,
        )

        if use_source_tpose:
            from .tpose import _build_canonical_tpose, _build_canonical_tpose_hands, _build_canonical_tpose_face
            tpose_kps_src, tpose_conf_src = _build_canonical_tpose(ratios_src, (cW, cH), ground_anchor=True)
            farm_len = ratios_src["r_forearm"] * cH * 0.78
            tpose_lhand_src, tpose_rhand_src = _build_canonical_tpose_hands(tpose_kps_src, (cW, cH), arm_scale=farm_len)
            tpose_face_src = _build_canonical_tpose_face(tpose_kps_src, face_src)
            tpose_sapiens_src = _build_canonical_tpose_sapiens308(tpose_kps_src, ratios_src, (cW, cH), sapiens_src)
            skel_src_img = _render_skeleton(
                tpose_kps_src, tpose_conf_src, (cH, cW), skeleton_style,
                thr=keypoint_threshold,
                face_data=tpose_face_src, lhand_data=tpose_lhand_src, rhand_data=tpose_rhand_src,
                sapiens_308_data=tpose_sapiens_src,
                line_thickness=line_thickness, point_radius=point_radius,
            )
        else:
            skel_src_img = _render_skeleton(
                kps_src, conf_src, (H_src, W_src), skeleton_style,
                thr=keypoint_threshold,
                face_data=face_src, lhand_data=lhand_src, rhand_data=rhand_src,
                sapiens_308_data=sapiens_src,
                line_thickness=line_thickness, point_radius=point_radius,
            )

        skel_tgt_img = _render_skeleton(
            kps_tgt, conf_tgt, (H_tgt, W_tgt), skeleton_style,
            thr=keypoint_threshold,
            face_data=face_tgt, lhand_data=lhand_tgt, rhand_data=rhand_tgt,
            sapiens_308_data=sapiens_tgt,
            line_thickness=line_thickness, point_radius=point_radius,
        )

        img_tgt_rsz = cv2.resize(img_tgt_np, (cW, cH))
        overlay = _overlay_skeleton(
            img_tgt_rsz, kps_ret, conf_ret, skeleton_style,
            thr=keypoint_threshold,
            face_data=face_ret_data, lhand_data=lhand_ret_data, rhand_data=rhand_ret_data,
            sapiens_308_data=sapiens_ret_data,
            line_thickness=line_thickness, point_radius=point_radius,
            skeleton_opacity=skeleton_opacity,
        )
        comp = _comparison(img_src_np, skel_src_img, skel_tgt_img, skel_img)

        # Build Full DWPose JSON and POSE_KEYPOINT outputs
        openpose_dict = _to_openpose_dict(
            kps_ret, conf_ret, (cW, cH),
            face_data=face_ret_data, lhand_data=lhand_ret_data, rhand_data=rhand_ret_data,
            sapiens_308_data=sapiens_ret_data,
        )
        json_out = json.dumps(openpose_dict, ensure_ascii=True)
        pose_keypoint_out = [openpose_dict]

        is_3d_enhanced = mode_src != "2d_only" or mode_tgt != "2d_only"
        limb_info_out = json.dumps(
            {
                "source_height_cm": round(h_src_display, 1),
                "target_height_cm": round(h_tgt_display, 1),
                "height_scale_ratio": round(height_ratio, 3),
                "source_proportion_ratios": {k: round(v, 4) for k, v in ratios_src.items() if k.startswith("r_")},
                "retarget_pipeline": "3D_Proportion_Preserving_Kinematics" if is_3d_enhanced else "2D_Kinematics",
                "depth_mode_source": mode_src,
                "depth_mode_target": mode_tgt,
                "depth_enhanced": is_3d_enhanced,
                "dwpose_face_retargeted": face_ret_data is not None,
                "dwpose_hands_retargeted": lhand_ret_data is not None or rhand_ret_data is not None,
                "grounding_mode": grounding_mode,
            },
            indent=2,
        )

        return (
            json_out,
            _to_comfy(skel_img),
            _to_comfy(overlay),
            _to_comfy(comp),
            float(h_src_display),
            float(h_tgt_display),
            limb_info_out,
            pose_keypoint_out,
        )



