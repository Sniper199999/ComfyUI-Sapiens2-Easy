"""
Sapiens2PoseToTPose — extracts scale-invariant anatomical proportion ratios
(torso, upper/lower legs, upper/lower arms, shoulder span, hip span, head scale)
and renders a standardized, canonical neutral T-Pose OpenPose skeleton.

Math:
  1. Samples 3D coordinates directly from Sapiens Pointmap or unprojects (u, v, depth) to camera space (X, Y, Z).
  2. Measures true unforeshortened bone lengths along the full 3D anatomical kinematic chain.
  3. Extracts dimensionless proportion ratios r_i = L_i / H_total.
  4. Calibrates metric height (cm) using 3D facial biometrics (interpupillary ~6.3cm, ear-to-ear ~14cm) or user input.
  5. Renders canonical T-pose skeleton grounded cleanly on canvas.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import cv2
import numpy as np
import torch

from .retarget import (
    _SKEL_STYLES,
    _build_kps_3d,
    _dist_3d_or_2d,
    _estimate_biometric_height,
    _extract_depth_map_array,
    _extract_proportions,
    _overlay_skeleton,
    _parse_json,
    _render_skeleton,
    _sanitize_ratios,
    _to_comfy,
    _to_openpose_dict,
)

_CANVAS_MODES = ["match_image", "square_512", "square_768", "square_1024"]


def _kinematic_chain_height_3d(
    kps: np.ndarray,
    conf: np.ndarray,
    kps_3d: Optional[np.ndarray] = None,
    thr: float = 0.15,
) -> float:
    """
    Measures standing height by summing cumulative Euclidean distances along the anatomical kinematic chain:
    Head Top -> Nose -> Neck -> Mid-Hip -> Thighs (avg) -> Shins (avg) -> Feet (avg).
    Handles bent knees, hunched spine, or walking poses accurately in 3D or 2D.
    """
    def seg_d(i: int, j: int, default_val: float = 0.0) -> float:
        d = _dist_3d_or_2d(i, j, kps, conf, kps_3d=kps_3d, thr=thr)
        return d if d > 0 else default_val

    # 1. Torso: Neck (1) -> MidHip (8)
    torso = seg_d(1, 8)
    if torso <= 0:
        torso = seg_d(1, 9) if seg_d(1, 9) > 0 else seg_d(1, 12)
    if torso <= 0:
        torso = 120.0 if kps_3d is None else 0.45

    # 2. Head: Nose (0) -> Neck (1) * 1.15 (to head crown)
    neck_nose = seg_d(0, 1, torso * 0.25)
    head_len = neck_nose * 1.15

    # 3. Legs: MidHip -> Thigh -> Shin -> Foot
    # If legs are severely cropped / cut off by image boundary, reconstruct from torso
    r_thigh = seg_d(9, 10, 0.0)
    l_thigh = seg_d(12, 13, 0.0)
    min_thigh = torso * 0.30
    if r_thigh < min_thigh and l_thigh < min_thigh:
        thigh = torso * 0.88
    elif r_thigh < min_thigh:
        thigh = l_thigh
    elif l_thigh < min_thigh:
        thigh = r_thigh
    else:
        thigh = (r_thigh + l_thigh) * 0.5

    r_shin = seg_d(10, 11, 0.0)
    l_shin = seg_d(13, 14, 0.0)
    min_shin = torso * 0.30
    if r_shin < min_shin and l_shin < min_shin:
        shin = torso * 0.85
    elif r_shin < min_shin:
        shin = l_shin
    elif l_shin < min_shin:
        shin = r_shin
    else:
        shin = (r_shin + l_shin) * 0.5

    r_foot = seg_d(11, 22, 0.0)
    l_foot = seg_d(14, 19, 0.0)
    min_foot = torso * 0.10
    if r_foot < min_foot and l_foot < min_foot:
        foot = torso * 0.22
    elif r_foot < min_foot:
        foot = l_foot
    elif l_foot < min_foot:
        foot = r_foot
    else:
        foot = (r_foot + l_foot) * 0.5

    # Vertical foot contribution to standing height is heel drop (~40% of ankle-to-toe span)
    return float(head_len + torso + thigh + shin + foot * 0.40)


def _build_canonical_tpose(
    ratios: dict[str, float],
    canvas_wh: tuple[int, int],
    ground_anchor: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constructs an anatomically grounded BODY_25 T-Pose skeleton from pre-sanitized ratios.
    Expects ratios already sanitized by _sanitize_ratios().
    """
    cW, cH = canvas_wh
    kps = np.zeros((25, 2), dtype=np.float32)
    conf = np.ones(25, dtype=np.float32)

    # Base scale: 78% of canvas height
    standing_h = cH * 0.78
    center_x = cW / 2.0

    torso_len = ratios["r_torso"] * standing_h
    neck_nose = ratios["r_neck_nose"] * standing_h
    eye_span  = ratios["r_eye_span"] * 2.0 * standing_h
    ear_span  = ratios["r_ear_span"] * 2.0 * standing_h
    sh_span   = ratios["r_shoulder_span"] * 2.0 * standing_h
    hip_span  = ratios["r_hip_span"] * 2.0 * standing_h
    uarm_len  = ratios["r_upper_arm"] * standing_h
    farm_len  = ratios["r_forearm"] * standing_h
    thigh_len = ratios["r_thigh"] * standing_h
    shin_len  = ratios["r_shin"] * standing_h
    foot_len  = ratios["r_foot_len"] * standing_h

    if ground_anchor:
        foot_y = cH * 0.94
        ankle_y = foot_y - foot_len * 0.35
        knee_y = ankle_y - shin_len
        hip_y = knee_y - thigh_len
        neck_y = hip_y - torso_len
        nose_y = neck_y - neck_nose
    else:
        hip_y = cH * 0.55
        neck_y = hip_y - torso_len
        nose_y = neck_y - neck_nose
        knee_y = hip_y + thigh_len
        ankle_y = knee_y + shin_len

    # 1. MidHip (8)
    kps[8] = [center_x, hip_y]

    # 2. Neck (1) & Head
    kps[1] = [center_x, neck_y]
    kps[0] = [center_x, nose_y]
    kps[15] = [center_x - eye_span * 0.5, nose_y - neck_nose * 0.08]   # REye
    kps[16] = [center_x + eye_span * 0.5, nose_y - neck_nose * 0.08]   # LEye
    kps[17] = [center_x - ear_span * 0.5, nose_y - neck_nose * 0.05]   # REar
    kps[18] = [center_x + ear_span * 0.5, nose_y - neck_nose * 0.05]   # LEar

    # 3. Shoulders & Arms (T-Pose straight horizontal)
    r_sh_x = center_x - sh_span * 0.5
    l_sh_x = center_x + sh_span * 0.5
    sh_y = neck_y + 8.0

    kps[2] = [r_sh_x, sh_y]
    kps[3] = [r_sh_x - uarm_len, sh_y]
    kps[4] = [r_sh_x - uarm_len - farm_len, sh_y]

    kps[5] = [l_sh_x, sh_y]
    kps[6] = [l_sh_x + uarm_len, sh_y]
    kps[7] = [l_sh_x + uarm_len + farm_len, sh_y]

    # 4. Hips & Legs (straight vertical)
    r_hip_x = center_x - hip_span * 0.5
    l_hip_x = center_x + hip_span * 0.5

    kps[9] = [r_hip_x, hip_y]
    kps[10] = [r_hip_x, knee_y]
    kps[11] = [r_hip_x, ankle_y]

    kps[12] = [l_hip_x, hip_y]
    kps[13] = [l_hip_x, knee_y]
    kps[14] = [l_hip_x, ankle_y]

    # 5. Feet
    kps[24] = [r_hip_x - 5.0, ankle_y + 6.0]
    kps[22] = [r_hip_x + 10.0, ankle_y + 10.0]
    kps[23] = [r_hip_x + 15.0, ankle_y + 8.0]

    kps[21] = [l_hip_x + 5.0, ankle_y + 6.0]
    kps[19] = [l_hip_x - 10.0, ankle_y + 10.0]
    kps[20] = [l_hip_x - 15.0, ankle_y + 8.0]

    return kps, conf


def _build_canonical_tpose_hands(
    tpose_kps: np.ndarray,
    canvas_wh: tuple[int, int],
    arm_scale: float = 60.0,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Builds canonical open-palm T-Pose hands (21 keypoints each) extending horizontally from wrists.
    Hand length is scaled to 45% of forearm length (anatomical standard) with natural radial finger splay.
    """
    hand_len = arm_scale * 0.45

    def make_hand(wrist_pt: np.ndarray, is_right: bool):
        pts = np.zeros((21, 2), dtype=np.float32)
        conf = np.ones(21, dtype=np.float32)
        sign = -1.0 if is_right else 1.0
        pts[0] = wrist_pt.copy()

        # 5 fingers: Thumb (top -Y), Index, Middle, Ring, Pinky (bottom +Y)
        finger_angles = [-32.0, -12.0, 0.0, 12.0, 25.0]
        finger_lens   = [0.65,  0.92,  1.0, 0.88, 0.75]
        palm_offsets  = [0.20,  0.35,  0.38, 0.35, 0.30]

        for f_idx in range(5):
            rad = np.radians(finger_angles[f_idx])
            f_len = finger_lens[f_idx] * hand_len
            mcp_dist = palm_offsets[f_idx] * hand_len

            base_idx = 1 + f_idx * 4
            for seg in range(4):
                d = mcp_dist + (seg + 1) * ((f_len - mcp_dist) / 4.0)
                px = wrist_pt[0] + sign * (d * np.cos(rad))
                py = wrist_pt[1] + (d * np.sin(rad))
                pts[base_idx + seg] = [px, py]

        return pts, conf

    r_hand = make_hand(tpose_kps[4], is_right=True)
    l_hand = make_hand(tpose_kps[7], is_right=False)
    return l_hand, r_hand


def _build_canonical_tpose_face(
    tpose_kps: np.ndarray,
    face_src_data: Optional[tuple[np.ndarray, np.ndarray]],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Positions Source's facial landmarks in a canonical neutral upright expression on the T-Pose head.
    Un-rotates any head tilt from the source image so landmarks are perfectly level with eyes/ears.
    """
    if face_src_data is None or face_src_data[0] is None or np.sum(face_src_data[1] > 0.1) < 10:
        return None
    pts_s, conf_s = face_src_data
    nose_pt = tpose_kps[0]
    neck_pt = tpose_kps[1]

    # Calculate source head tilt / roll angle from eye corners (36 right outer, 45 left outer)
    if conf_s[36] > 0.1 and conf_s[45] > 0.1:
        d_eyes_src = float(np.linalg.norm(pts_s[45] - pts_s[36]))
        src_roll = float(np.arctan2(pts_s[45, 1] - pts_s[36, 1], pts_s[45, 0] - pts_s[36, 0]))
    else:
        d_eyes_src = 40.0
        src_roll = 0.0
    if d_eyes_src < 5.0: d_eyes_src = 40.0

    # Target eye span on T-pose head
    head_h = float(np.linalg.norm(neck_pt - nose_pt))
    target_eye_span = max(head_h * 0.55, 14.0)
    scale = target_eye_span / d_eyes_src

    c_src = pts_s[30] if conf_s[30] > 0.1 else np.mean(pts_s[conf_s > 0.1], axis=0)

    # Rotation matrix to un-tilt the face so it is strictly upright (0 roll)
    cos_a = np.cos(-src_roll)
    sin_a = np.sin(-src_roll)
    unrot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    pts_out = np.zeros((68, 2), dtype=np.float32)
    conf_out = np.zeros(68, dtype=np.float32)
    for i in range(68):
        if conf_s[i] > 0.1:
            rel = pts_s[i] - c_src
            rel_upright = unrot_mat @ rel
            pts_out[i] = nose_pt + rel_upright * scale
            conf_out[i] = conf_s[i]

    # Synchronize T-Pose body head keypoints with canonical T-Pose face landmarks
    if conf_out[30] > 0.1 and pts_out[30, 0] > 10.0:
        tpose_kps[0] = pts_out[30]
    if np.all(conf_out[36:42] > 0.1):
        tpose_kps[15] = np.mean(pts_out[36:42], axis=0)
    if np.all(conf_out[42:48] > 0.1):
        tpose_kps[16] = np.mean(pts_out[42:48], axis=0)

    # Position ears symmetrically on the sides of the head relative to the eyes
    eye_y = (tpose_kps[15, 1] + tpose_kps[16, 1]) * 0.5
    ear_offset = max(target_eye_span * 0.7, 14.0)
    tpose_kps[17] = [tpose_kps[15, 0] - ear_offset, eye_y + 2.0]
    tpose_kps[18] = [tpose_kps[16, 0] + ear_offset, eye_y + 2.0]

    return pts_out, conf_out


class Sapiens2PoseToTPose:
    """
    Extracts a person's anatomical proportion ratios and renders a standardized
    canonical T-Pose OpenPose / DWPose skeleton for inspection and calibrated retargeting.
    Supports Sapiens 3D Pointmaps, Pinhole Depth Unprojection, and 2D fallback.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "openpose_json": ("STRING,POSE_KEYPOINT", {"default": "", "tooltip": "OpenPose / DWPose JSON or POSE_KEYPOINT input"}),
                "image": ("IMAGE",),
                "canvas_mode": (_CANVAS_MODES, {"default": "match_image"}),
                "ground_anchor": ("BOOLEAN", {"default": True, "tooltip": "Anchors feet to bottom of canvas"}),
            },
            "optional": {
                "render_config": (
                    "SAPIENS2_POSE_CONFIG",
                    {"tooltip": "Optional modular rendering configuration from Sapiens2PoseRenderConfig node (line thickness, point radius, opacity, threshold)."},
                ),
                "pointmap": ("SAPIENS2_POINTMAP",),
                "depth_map": (
                    "IMAGE",
                    {"tooltip": "Optional fallback 2D depth map. Not needed if 'pointmap' is connected."},
                ),
                "height_cm": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 300.0,
                        "step": 0.5,
                        "tooltip": "Known height in cm. 0 = auto-estimate from 3D/2D biometrics.",
                    },
                ),
                "camera_fov_deg": (
                    "FLOAT",
                    {
                        "default": 60.0,
                        "min": 10.0,
                        "max": 150.0,
                        "step": 1.0,
                        "tooltip": "Approximate camera horizontal FOV in degrees for depth unprojection.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "FLOAT", "STRING", "POSE_KEYPOINT")
    RETURN_NAMES = (
        "tpose_openpose_json",
        "tpose_skeleton_image",
        "tpose_overlay_image",
        "height_cm",
        "limb_proportions_json",
        "pose_keypoint",
    )
    FUNCTION = "generate_tpose"
    CATEGORY = "Sapiens2"

    def generate_tpose(
        self,
        openpose_json: Any,
        image: torch.Tensor,
        canvas_mode: str = "match_image",
        ground_anchor: bool = True,
        render_config: Optional[dict[str, Any]] = None,
        pointmap: Optional[Any] = None,
        depth_map: Optional[torch.Tensor] = None,
        height_cm: float = 0.0,
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

        kps, conf, meta, face_src, lhand_src, rhand_src, sapiens_src = _parse_json(openpose_json)
        if kps is None:
            raise ValueError("[Sapiens2 PoseToTPose] openpose_json is empty or invalid.")

        from .retarget import _detect_skeleton_style, _build_canonical_tpose_sapiens308
        skeleton_style = _detect_skeleton_style(
            target_name=meta.get("target"),
            sapiens_data=sapiens_src,
            face_data=face_src,
            lhand_data=lhand_src,
            rhand_data=rhand_src,
            body_kps_count=len(kps),
        )

        img_np = (image[0].cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
        H, W = img_np.shape[:2]

        cW, cH = {
            "match_image": (W, H),
            "square_512": (512, 512),
            "square_768": (768, 768),
            "square_1024": (1024, 1024),
        }.get(canvas_mode, (W, H))

        # 3D keypoint construction (Pointmap > Pinhole Depth Unprojection > 2D fallback)
        kps_3d, depth_mode = _build_kps_3d(kps, conf, pointmap, depth_map, (H, W), camera_fov_deg)

        # Depth extraction (for fallback heuristic if kps_3d is None)
        depth_arr = _extract_depth_map_array(depth_map, (H, W))

        # True kinematic chain standing height measurement
        kinematic_height = _kinematic_chain_height_3d(kps, conf, kps_3d=kps_3d, thr=keypoint_threshold)

        # Extract raw proportion ratios, then sanitize to anatomical bounds.
        raw_ratios = _extract_proportions(kps, conf, depth_arr, (H, W), kps_3d=kps_3d, thr=keypoint_threshold)
        ratios = _sanitize_ratios(raw_ratios)

        # Height estimation / calibration
        h_cm = float(height_cm)
        if h_cm <= 0:
            h_cm = _estimate_biometric_height(kps, conf, kinematic_height, kps_3d=kps_3d, thr=keypoint_threshold)

        tpose_kps, tpose_conf = _build_canonical_tpose(ratios, (cW, cH), ground_anchor=ground_anchor)

        # Canonical DWPose Hands & Face
        farm_len = ratios["r_forearm"] * cH * 0.78
        tpose_lhand, tpose_rhand = _build_canonical_tpose_hands(tpose_kps, (cW, cH), arm_scale=farm_len)
        tpose_face = _build_canonical_tpose_face(tpose_kps, face_src)

        # Canonical Sapiens 308 Keypoints
        tpose_sapiens = _build_canonical_tpose_sapiens308(tpose_kps, ratios, (cW, cH), sapiens_src)

        # Render outputs
        skel_img = _render_skeleton(
            tpose_kps, tpose_conf, (cH, cW), skeleton_style,
            thr=keypoint_threshold,
            face_data=tpose_face, lhand_data=tpose_lhand, rhand_data=tpose_rhand,
            sapiens_308_data=tpose_sapiens,
            line_thickness=line_thickness, point_radius=point_radius,
        )

        img_rsz = cv2.resize(img_np, (cW, cH))
        overlay = _overlay_skeleton(
            img_rsz, tpose_kps, tpose_conf, skeleton_style,
            thr=keypoint_threshold,
            face_data=tpose_face, lhand_data=tpose_lhand, rhand_data=tpose_rhand,
            sapiens_308_data=tpose_sapiens,
            line_thickness=line_thickness, point_radius=point_radius,
            skeleton_opacity=skeleton_opacity,
        )

        meta_payload = {
            "measured_height_cm": round(h_cm, 1),
            "proportion_ratios": {k: round(v, 4) for k, v in ratios.items() if k.startswith("r_")},
            "depth_mode": depth_mode,
            "depth_enhanced": depth_mode != "2d_only",
            "kinematic_height_units": round(kinematic_height, 2),
        }
        openpose_dict = _to_openpose_dict(
            tpose_kps, tpose_conf, (cW, cH),
            face_data=tpose_face, lhand_data=tpose_lhand, rhand_data=tpose_rhand,
            sapiens_308_data=tpose_sapiens,
            extra_meta=meta_payload,
        )
        json_out = json.dumps(openpose_dict, ensure_ascii=True)
        pose_keypoint_out = [openpose_dict]
        proportions_summary = json.dumps(meta_payload, indent=2)

        return (
            json_out,
            _to_comfy(skel_img),
            _to_comfy(overlay),
            float(h_cm),
            proportions_summary,
            pose_keypoint_out,
        )
