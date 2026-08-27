import numpy as np
import pytest
from sapiens2_nodes.retarget import (
    _EDGE_COLORS,
    _draw_skel,
    _retarget_face_dwpose,
    _retarget_kinematics,
    _safe_normalize,
    _build_orthonormal_frame,
    _robust_rotation_matrix,
    _compute_hinge_normal,
)
from sapiens2_nodes.tpose import _kinematic_chain_height_3d
from sapiens2_nodes.transition import _ease


def test_safe_normalize():
    # Non-zero vector
    v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    normed = _safe_normalize(v)
    assert abs(np.linalg.norm(normed) - 1.0) < 1e-6
    assert np.allclose(normed, [0.6, 0.8, 0.0])

    # Zero-norm vector with fallback
    v_zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    fb = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    normed_zero = _safe_normalize(v_zero, fallback=fb)
    assert np.allclose(normed_zero, fb)


def test_build_orthonormal_frame():
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    right = np.array([2.0, 0.0, 0.0], dtype=np.float32)

    F = _build_orthonormal_frame(origin, up, right)
    # Check shape (3, 3)
    assert F.shape == (3, 3)
    # Check orthogonality: F.T @ F = I
    assert np.allclose(F.T @ F, np.eye(3), atol=1e-5)
    # Check right-handed determinant = +1
    assert abs(np.linalg.det(F) - 1.0) < 1e-5


def test_robust_rotation_matrix():
    # 1. Identity alignment
    v_same = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    R_id = _robust_rotation_matrix(v_same, v_same)
    assert np.allclose(R_id, np.eye(3), atol=1e-5)

    # 2. Antiparallel 180° rotation along Z-axis: [0, 0, 1] -> [0, 0, -1]
    v_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    v_z_anti = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    R_anti_z = _robust_rotation_matrix(v_z, v_z_anti)
    assert not np.isnan(R_anti_z).any()
    rotated_z = R_anti_z @ v_z
    assert np.allclose(rotated_z, v_z_anti, atol=1e-5)

    # 3. Arbitrary 3D vector rotation
    v_a = _safe_normalize(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    v_b = _safe_normalize(np.array([0.0, 1.0, 1.0], dtype=np.float32))
    R_ab = _robust_rotation_matrix(v_a, v_b)
    assert np.allclose(R_ab @ v_a, v_b, atol=1e-5)
    assert np.allclose(R_ab.T @ R_ab, np.eye(3), atol=1e-5)


def test_compute_hinge_normal():
    shoulder = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    elbow = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    wrist = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    n_arm = _compute_hinge_normal(shoulder, elbow, wrist)
    # Cross product of [1, 0, 0] and [0, -1, 0] is [0, 0, -1]
    assert np.allclose(np.abs(n_arm), [0.0, 0.0, 1.0], atol=1e-5)


def test_kinematic_chain_confidence_weighted_bilateral():
    kps = np.zeros((25, 2), dtype=np.float32)
    conf = np.ones(25, dtype=np.float32)

    # Torso: Neck(1) -> MidHip(8)
    kps[1] = [256.0, 100.0]
    kps[8] = [256.0, 200.0]
    # Right thigh: length = 100, conf = 0.9
    kps[9] = [240.0, 200.0]
    kps[10] = [240.0, 300.0]
    conf[9], conf[10] = 0.9, 0.9
    # Left thigh: length = 80, conf = 0.1 (low confidence)
    kps[12] = [272.0, 200.0]
    kps[13] = [272.0, 280.0]
    conf[12], conf[13] = 0.1, 0.1

    # Shins
    kps[11] = [240.0, 400.0]
    kps[14] = [272.0, 400.0]

    h = _kinematic_chain_height_3d(kps, conf)
    assert h > 300.0


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

