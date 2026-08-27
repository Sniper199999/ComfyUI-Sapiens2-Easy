"""
Sapiens2PoseTransitionAnimation — generates a smooth keypoint morphing animation
transitioning from a Starting Pose (or target dimensions) to an Ending Pose (or source retargeted dimensions).

Outputs batch IMAGE tensors [B, H, W, 3] ready for video/GIF export or AnimateDiff preview in ComfyUI.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import cv2
import numpy as np
import torch

from .retarget import (
    _detect_skeleton_style,
    _overlay_skeleton,
    _parse_json,
    _render_skeleton,
    _to_comfy,
    _to_openpose_dict,
)


def _ease(t: float, mode: str = "smooth_in_out") -> float:
    """
    Computes normalized interpolation easing factor for t in [0, 1].
    """
    t = float(np.clip(t, 0.0, 1.0))
    if mode == "linear":
        return t
    elif mode == "ease_in":
        return t * t
    elif mode == "ease_out":
        return t * (2.0 - t)
    elif mode == "smooth_in_out":
        return 0.5 * (1.0 - np.cos(t * np.pi))
    elif mode == "bounce":
        n1 = 7.5625
        d1 = 2.75
        if t < 1.0 / d1:
            return float(n1 * t * t)
        elif t < 2.0 / d1:
            t_sub = t - (1.5 / d1)
            return float(n1 * t_sub * t_sub + 0.75)
        elif t < 2.5 / d1:
            t_sub = t - (2.25 / d1)
            return float(n1 * t_sub * t_sub + 0.9375)
        else:
            t_sub = t - (2.625 / d1)
            return float(n1 * t_sub * t_sub + 0.984375)
    return 0.5 * (1.0 - np.cos(t * np.pi))


def _interpolate_keypoints_2d(
    kps_a: np.ndarray,
    conf_a: np.ndarray,
    kps_b: np.ndarray,
    conf_b: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolates between two sets of 2D keypoints with confidence blending.
    """
    n = max(len(kps_a), len(kps_b))
    kps_out = np.zeros((n, 2), dtype=np.float32)
    conf_out = np.zeros(n, dtype=np.float32)

    for i in range(n):
        pa = kps_a[i] if i < len(kps_a) else np.zeros(2, dtype=np.float32)
        ca = conf_a[i] if i < len(conf_a) else 0.0
        pb = kps_b[i] if i < len(kps_b) else np.zeros(2, dtype=np.float32)
        cb = conf_b[i] if i < len(conf_b) else 0.0

        if ca > 0.05 and cb > 0.05:
            kps_out[i] = pa * (1.0 - alpha) + pb * alpha
            conf_out[i] = ca * (1.0 - alpha) + cb * alpha
        elif ca > 0.05:
            kps_out[i] = pa
            conf_out[i] = ca * (1.0 - alpha)
        elif cb > 0.05:
            kps_out[i] = pb
            conf_out[i] = cb * alpha
        else:
            kps_out[i] = [0.0, 0.0]
            conf_out[i] = 0.0

    return kps_out, conf_out


class Sapiens2PoseTransitionAnimation:
    """
    Renders a smooth keypoint transition animation from a Start Pose to an End Pose.
    Visualizes skeleton retargeting, bone scaling, and posture transformation across frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_pose_json": ("STRING,POSE_KEYPOINT", {"tooltip": "Initial pose JSON or POSE_KEYPOINT (e.g. Target pose)."}),
                "end_pose_json":   ("STRING,POSE_KEYPOINT", {"tooltip": "Final pose JSON or POSE_KEYPOINT (e.g. Retargeted Source pose)."}),
                "frame_count":     ("INT", {"default": 30, "min": 2, "max": 300, "step": 1, "tooltip": "Total number of animation frames to render."}),
                "easing":          (["smooth_in_out", "linear", "ease_in", "ease_out"], {"default": "smooth_in_out", "tooltip": "Animation easing motion curve."}),
                "loop_mode":       (["ping_pong", "forward_only", "loop"], {"default": "ping_pong", "tooltip": "Animation loop behavior (ping-pong reverses smoothly back to start)."}),
            },
            "optional": {
                "background_image": ("IMAGE", {"tooltip": "Optional reference background image for overlay rendering."}),
                "render_config":    ("SAPIENS2_POSE_CONFIG", {"tooltip": "Optional modular rendering styling (thickness, point radius, opacity, threshold)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("animated_skeleton", "animated_overlay", "interpolated_json_list")
    FUNCTION = "animate_transition"
    CATEGORY = "Sapiens2"

    def animate_transition(
        self,
        start_pose_json: Any,
        end_pose_json: Any,
        frame_count: int = 30,
        easing: str = "smooth_in_out",
        loop_mode: str = "ping_pong",
        background_image: Optional[torch.Tensor] = None,
        render_config: Optional[dict[str, Any]] = None,
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

        kps_a, conf_a, meta_a, face_a, lhand_a, rhand_a, sapiens_a = _parse_json(start_pose_json)
        kps_b, conf_b, meta_b, face_b, lhand_b, rhand_b, sapiens_b = _parse_json(end_pose_json)

        if kps_a is None or kps_b is None:
            raise ValueError("[Sapiens2 PoseTransitionAnimation] One or both pose inputs are invalid or empty.")

        # Detect skeleton style
        skeleton_style = _detect_skeleton_style(
            target_name=meta_b.get("target") or meta_a.get("target"),
            sapiens_data=sapiens_b or sapiens_a,
            face_data=face_b or face_a,
            lhand_data=lhand_b or lhand_a,
            rhand_data=rhand_b or rhand_a,
            body_kps_count=max(len(kps_a), len(kps_b)),
        )

        # Canvas dimensions
        if background_image is not None:
            bg_np = (background_image[0].cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
            H, W = bg_np.shape[:2]
        else:
            H = int(meta_b.get("canvas_height") or meta_a.get("canvas_height") or 512)
            W = int(meta_b.get("canvas_width") or meta_a.get("canvas_width") or 512)
            bg_np = np.zeros((H, W, 3), dtype=np.uint8)

        total_frames = max(2, int(frame_count))
        if loop_mode == "ping_pong":
            # Generate forward sequence then reverse
            alphas_forward = [_ease(i / (total_frames - 1), easing) for i in range(total_frames)]
            alphas_backward = alphas_forward[-2:0:-1]
            alphas = alphas_forward + alphas_backward
        else:
            alphas = [_ease(i / (total_frames - 1), easing) for i in range(total_frames)]

        skel_frames = []
        overlay_frames = []
        json_frames = []

        for frame_idx, alpha in enumerate(alphas):
            # Interpolate Body
            kps_inter, conf_inter = _interpolate_keypoints_2d(kps_a, conf_a, kps_b, conf_b, alpha)

            # Interpolate Face
            if (face_a[0] is not None or face_b[0] is not None):
                f_kps_a = face_a[0] if face_a[0] is not None else np.zeros((68, 2), np.float32)
                f_conf_a = face_a[1] if face_a[1] is not None else np.zeros(68, np.float32)
                f_kps_b = face_b[0] if face_b[0] is not None else np.zeros((68, 2), np.float32)
                f_conf_b = face_b[1] if face_b[1] is not None else np.zeros(68, np.float32)
                face_inter = _interpolate_keypoints_2d(f_kps_a, f_conf_a, f_kps_b, f_conf_b, alpha)
            else:
                face_inter = None

            # Interpolate Left Hand
            if (lhand_a[0] is not None or lhand_b[0] is not None):
                lh_kps_a = lhand_a[0] if lhand_a[0] is not None else np.zeros((21, 2), np.float32)
                lh_conf_a = lhand_a[1] if lhand_a[1] is not None else np.zeros(21, np.float32)
                lh_kps_b = lhand_b[0] if lhand_b[0] is not None else np.zeros((21, 2), np.float32)
                lh_conf_b = lhand_b[1] if lhand_b[1] is not None else np.zeros(21, np.float32)
                lhand_inter = _interpolate_keypoints_2d(lh_kps_a, lh_conf_a, lh_kps_b, lh_conf_b, alpha)
            else:
                lhand_inter = None

            # Interpolate Right Hand
            if (rhand_a[0] is not None or rhand_b[0] is not None):
                rh_kps_a = rhand_a[0] if rhand_a[0] is not None else np.zeros((21, 2), np.float32)
                rh_conf_a = rhand_a[1] if rhand_a[1] is not None else np.zeros(21, np.float32)
                rh_kps_b = rhand_b[0] if rhand_b[0] is not None else np.zeros((21, 2), np.float32)
                rh_conf_b = rhand_b[1] if rhand_b[1] is not None else np.zeros(21, np.float32)
                rhand_inter = _interpolate_keypoints_2d(rh_kps_a, rh_conf_a, rh_kps_b, rh_conf_b, alpha)
            else:
                rhand_inter = None

            # Interpolate Sapiens 308
            if (sapiens_a[0] is not None or sapiens_b[0] is not None):
                s_kps_a = sapiens_a[0] if sapiens_a[0] is not None else np.zeros((308, 2), np.float32)
                s_conf_a = sapiens_a[1] if sapiens_a[1] is not None else np.zeros(308, np.float32)
                s_kps_b = sapiens_b[0] if sapiens_b[0] is not None else np.zeros((308, 2), np.float32)
                s_conf_b = sapiens_b[1] if sapiens_b[1] is not None else np.zeros(308, np.float32)
                sapiens_inter = _interpolate_keypoints_2d(s_kps_a, s_conf_a, s_kps_b, s_conf_b, alpha)
            else:
                sapiens_inter = None

            # Render Frame Skeletons
            skel_img = _render_skeleton(
                kps_inter, conf_inter, (H, W), skeleton_style,
                thr=keypoint_threshold,
                face_data=face_inter, lhand_data=lhand_inter, rhand_data=rhand_inter,
                sapiens_308_data=sapiens_inter,
                line_thickness=line_thickness, point_radius=point_radius,
            )

            overlay_img = _overlay_skeleton(
                bg_np.copy(), kps_inter, conf_inter, skeleton_style,
                thr=keypoint_threshold,
                face_data=face_inter, lhand_data=lhand_inter, rhand_data=rhand_inter,
                sapiens_308_data=sapiens_inter,
                line_thickness=line_thickness, point_radius=point_radius,
                skeleton_opacity=skeleton_opacity,
            )

            skel_frames.append(torch.from_numpy(skel_img.astype(np.float32) / 255.0))
            overlay_frames.append(torch.from_numpy(overlay_img.astype(np.float32) / 255.0))

            frame_dict = _to_openpose_dict(
                kps_inter, conf_inter, (W, H),
                face_data=face_inter, lhand_data=lhand_inter, rhand_data=rhand_inter,
                sapiens_308_data=sapiens_inter,
                extra_meta={"frame_index": frame_idx, "alpha": round(alpha, 4)},
            )
            json_frames.append(frame_dict)

        # Batch image tensors [B, H, W, 3]
        skel_batch = torch.stack(skel_frames, dim=0)
        overlay_batch = torch.stack(overlay_frames, dim=0)
        json_output = json.dumps(json_frames, ensure_ascii=True)

        return (skel_batch, overlay_batch, json_output)
