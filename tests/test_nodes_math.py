import numpy as np
import pytest
from sapiens2_nodes.retarget import (
    _EDGE_COLORS,
    _draw_skel,
    _retarget_face_dwpose,
    _retarget_kinematics,
)
from sapiens2_nodes.transition import _ease


def test_draw_skel_colors():
    """
    Verifies that _draw_skel renders RGB colors without color channel swapping.
    """
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    # Define a simple keypoint array for BODY25 edge (1, 8): Neck (50, 20) -> MidHip (50, 80)
    kps = np.zeros((25, 2), dtype=np.float32)
    conf = np.zeros(25, dtype=np.float32)
    kps[1] = [50.0, 20.0]
    conf[1] = 1.0
    kps[8] = [50.0, 80.0]
    conf[8] = 1.0

    _draw_skel(canvas, kps, conf, style="openpose_classic", thr=0.15, thickness=2, dot_r=0)

    # Line (1, 8) is the 0th edge in _BODY25_EDGES
    expected_color = _EDGE_COLORS[0]  # (255, 0, 85)
    sampled_pixel = canvas[50, 50]
    assert tuple(sampled_pixel) == expected_color, f"Expected RGB {expected_color}, got {tuple(sampled_pixel)}"


def test_kinematics_fallback():
    """
    Verifies that kinematic retargeting maintains connected joints when parent target keypoint is missing.
    """
    ratios_src = {
        "r_torso": 0.28, "r_neck_nose": 0.08, "r_eye_span": 0.04, "r_ear_span": 0.05,
        "r_shoulder_span": 0.12, "r_upper_arm": 0.16, "r_forearm": 0.14, "r_hip_span": 0.09,
        "r_thigh": 0.24, "r_shin": 0.23, "r_foot_len": 0.04, "total_height": 500.0,
    }
    ratios_tgt = ratios_src.copy()

    kps_tgt = np.zeros((25, 2), dtype=np.float32)
    conf_tgt = np.zeros(25, dtype=np.float32)

    # Root (MidHip = 8) is present
    kps_tgt[8] = [256.0, 300.0]
    conf_tgt[8] = 1.0

    # Parent (Neck = 1) is MISSING in target, but Child (RShoulder = 2) is VISIBLE in target
    kps_tgt[2] = [200.0, 150.0]
    conf_tgt[2] = 0.9

    ret_kps, ret_conf = _retarget_kinematics(
        ratios_source=ratios_src,
        ratios_target=ratios_tgt,
        kps_target=kps_tgt,
        conf_target=conf_tgt,
        height_ratio=1.0,
        grounding="none",
        canvas_wh=(512, 512),
        thr=0.15,
    )

    # Neck (1) should be placed via fallback relative to MidHip (8)
    assert ret_conf[1] > 0.0
    # RShoulder (2) must be placed relative to ret_kps[1] rather than copying un-retargeted target coordinates
    dist_from_parent = float(np.linalg.norm(ret_kps[2] - ret_kps[1]))
    assert dist_from_parent > 0.0
    assert not np.array_equal(ret_kps[2], kps_tgt[2]), "ret[child] must not naively copy un-retargeted target coordinates"


def test_face_dwpose_gating():
    """
    Verifies face retargeting keeps facial landmarks when source face is present even if target landmarks are low confidence.
    """
    kps_ret = np.zeros((25, 2), dtype=np.float32)
    conf_ret = np.ones(25, dtype=np.float32)
    kps_ret[0] = [256.0, 100.0]
    kps_ret[1] = [256.0, 200.0]
    kps_ret[8] = [256.0, 350.0]
    kps_ret[15] = [240.0, 95.0]
    kps_ret[16] = [272.0, 95.0]

    # Source face has full 68 points with valid coordinates
    face_src_pts = np.zeros((68, 2), dtype=np.float32)
    face_src_conf = np.ones(68, dtype=np.float32)
    for i in range(68):
        face_src_pts[i] = [256.0 + (i - 34) * 2.0, 100.0 + (i % 5) * 5.0]
    face_src_pts[30] = [256.0, 100.0]
    face_src_pts[36] = [240.0, 95.0]
    face_src_pts[45] = [272.0, 95.0]
    face_src_pts[17] = [235.0, 75.0]
    face_src = (face_src_pts, face_src_conf)

    # Target face has partial detection (e.g. eyebrow landmark 17 missing)
    face_tgt_pts = face_src_pts.copy()
    face_tgt_conf = np.ones(68, dtype=np.float32)
    face_tgt_conf[17] = 0.0  # missing target point
    face_tgt = (face_tgt_pts, face_tgt_conf)

    pts_out, conf_out = _retarget_face_dwpose(
        face_src=face_src,
        face_tgt=face_tgt,
        kps_ret=kps_ret,
        conf_ret=conf_ret,
        canvas_wh=(512, 512),
        thr=0.1,
    )

    # Landmark 17 should still be generated from source identity face mesh
    assert conf_out[17] > 0.1
    assert not np.all(pts_out[17] == 0.0)


def test_bounce_ease():
    """
    Verifies that _ease(t, mode='bounce') cleanly maps [0, 1] -> [0.0, 1.0].
    """
    assert _ease(0.0, "bounce") == 0.0
    assert abs(_ease(1.0, "bounce") - 1.0) < 1e-5
    # Verify non-decreasing progress trend overall ending at 1.0
    val_mid = _ease(0.5, "bounce")
    assert 0.0 < val_mid <= 1.0
