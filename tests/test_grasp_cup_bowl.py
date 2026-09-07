"""ROS-free tests for cup/bowl perception geometry and grasp-chunk construction."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from franka_duo_tele_data.action_spec import FrankaDuoActionSpec, rot6d_to_matrix  # noqa: E402
from franka_duo_tele_data.grasp_cup_bowl import (  # noqa: E402
    DATASET_GRASP_ROT6D,
    Detection,
    GraspTarget,
    build_grasp_rows,
    build_retreat_rows,
    expected_rim_axes_px,
    locate_target,
    pick_cup_and_bowl,
    refine_rim_ellipse,
    rim_direction_for_side,
    rim_edge_points,
    rim_ellipse,
    spoon_axis,
    spoon_centroid,
    summarize_detections,
    yaw_align_closing,
)
from franka_duo_tele_data.mcap_to_lerobot import invert_transform, make_transform  # noqa: E402
from franka_duo_tele_data.zed_pnp_calib import project_points  # noqa: E402

K = np.array([[364.0, 0.0, 308.0], [0.0, 364.0, 186.0], [0.0, 0.0, 1.0]])


def camera_pose() -> np.ndarray:
    pitch = math.radians(55)
    r_base_from_link = np.array(
        [[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]]
    )
    r_link_from_optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    return make_transform((0.03, 0.0, 0.34), r_base_from_link @ r_link_from_optical).astype(np.float64)


def scene():
    """Synthetic frame: dark tray, pale green cup and bowl (bowl with dark beans), blue spoon, plate."""
    img = np.full((360, 640, 3), 220, np.uint8)  # RGB
    cv2.rectangle(img, (120, 150), (360, 350), (55, 55, 50), -1)  # tray
    cv2.circle(img, (280, 280), 25, (150, 200, 170), -1)  # cup
    cv2.circle(img, (180, 300), 40, (150, 200, 170), -1)  # bowl
    cv2.circle(img, (180, 305), 28, (25, 20, 15), -1)  # beans
    cv2.rectangle(img, (185, 200), (195, 300), (90, 140, 200), -1)  # spoon handle up
    cv2.circle(img, (270, 210), 45, (150, 200, 170), -1)  # plate
    masks = {}
    for name, (c, r) in {
        "cup": ((280, 280), 25),
        "bowl": ((180, 300), 40),
        "plate": ((270, 210), 45),
    }.items():
        m = np.zeros((360, 640), np.uint8)
        cv2.circle(m, c, r, 255, -1)
        masks[name] = m
    spoon = np.zeros((360, 640), np.uint8)
    cv2.rectangle(spoon, (185, 200), (195, 300), 255, -1)
    dets = [
        Detection("cup", 0.8, masks["cup"]),
        Detection("bowl", 0.6, masks["bowl"]),
        Detection("bowl", 0.4, masks["plate"]),
        Detection("spoon", 0.5, spoon),
    ]
    return img, dets


def test_pick_cup_and_bowl_rejects_plate_and_finds_spoon():
    img, dets = scene()
    objs = summarize_detections(img, dets)
    picked = pick_cup_and_bowl(objs)
    assert picked["cup"].conf == 0.8
    assert picked["bowl"].dark_ratio > 0.25 and abs(picked["bowl"].rim_pixel[0] - 180) < 3
    plate = [o for o in objs if o.cls == "bowl" and o.dark_ratio < 0.05]
    assert len(plate) == 1
    axis = spoon_axis(img, dets)
    assert axis is not None and abs(abs(axis[1]) - 1) < 0.05  # vertical handle
    centre, axes = rim_ellipse(dets[0].mask)
    assert abs(centre[0] - 280) < 2 and min(axes) > 30


def test_locate_target_projects_to_rim_height_and_offsets_bowl_grasp():
    img, dets = scene()
    objs = pick_cup_and_bowl(summarize_detections(img, dets))
    t_base_from_cam = camera_pose()
    cup = locate_target(
        "cup",
        "right",
        objs,
        k_matrix=K,
        t_base_from_cam=t_base_from_cam,
        rim_z=-0.12,
        spoon_pixel=None,
        arm_base_xy=np.array([0.0, 0.35]),
        bowl_rim_radius_m=0.06,
    )
    assert abs(cup.rim_centre_base[2] + 0.12) < 1e-9 and np.allclose(
        cup.grasp_point_base, cup.rim_centre_base
    )
    # round trip: the rim centre must project back to the rim pixel
    back = project_points(cup.rim_centre_base, K, invert_transform(t_base_from_cam))[0]
    assert np.allclose(back, objs["cup"].rim_pixel, atol=1e-3)
    bowl = locate_target(
        "bowl",
        "left",
        objs,
        k_matrix=K,
        t_base_from_cam=t_base_from_cam,
        rim_z=-0.14,
        spoon_pixel=spoon_centroid(img, dets),
        arm_base_xy=np.array([0.0, 0.35]),
        bowl_rim_radius_m=0.06,
    )
    assert abs(np.linalg.norm(bowl.grasp_point_base[:2] - bowl.rim_centre_base[:2]) - 0.06) < 1e-6
    assert bowl.yaw_hint is not None and bowl.grasp_reason == "opposite_spoon"
    # spoon far from the bowl: fall back to the rim point toward the arm base
    far = locate_target(
        "bowl",
        "left",
        objs,
        k_matrix=K,
        t_base_from_cam=t_base_from_cam,
        rim_z=-0.14,
        spoon_pixel=(600.0, 20.0),
        arm_base_xy=np.array([0.0, 0.35]),
        bowl_rim_radius_m=0.06,
    )
    assert far.grasp_reason == "toward_arm_base"
    toward = np.array([0.0, 0.35]) - far.rim_centre_base[:2]
    toward /= np.linalg.norm(toward)
    assert np.allclose((far.grasp_point_base[:2] - far.rim_centre_base[:2]) / 0.06, toward, atol=1e-6)
    with pytest.raises(ValueError):
        locate_target(
            "bowl",
            "left",
            objs,
            k_matrix=K,
            t_base_from_cam=t_base_from_cam,
            rim_z=-0.14,
            spoon_pixel=None,
            arm_base_xy=None,
            bowl_rim_radius_m=0.06,
        )
    with pytest.raises(ValueError):
        locate_target(
            "cup",
            "right",
            {},
            k_matrix=K,
            t_base_from_cam=t_base_from_cam,
            rim_z=-0.12,
            spoon_pixel=None,
            arm_base_xy=np.array([0.0, 0.35]),
            bowl_rim_radius_m=0.06,
        )


def test_build_grasp_rows_shape_steps_grippers_and_other_arm_hold():
    state = np.zeros(20, np.float32)
    state[0:3] = (0.45, 0.30, -0.10)
    state[3:9] = DATASET_GRASP_ROT6D["left"]
    state[9:12] = (0.40, -0.16, -0.11)
    state[12:18] = (0.93, 0.15, -0.33, 0.13, -0.99, -0.09)
    state[18:] = 1.0
    grasp = GraspTarget(
        "cup",
        "right",
        np.array([0.35, 0.03, -0.12]),
        np.array([0.35, 0.03, -0.12]),
        None,
        (285.0, 266.0),
    )
    rows = build_grasp_rows(
        state, grasp, grasp_rot6d=np.asarray(DATASET_GRASP_ROT6D["right"]), step_m=0.01, step_rad=0.08
    )
    assert rows.shape[1] == 20 and len(rows) > 30
    assert np.allclose(rows[:, 0:9], state[0:9]) and np.all(rows[:, 18] == 1.0)
    spec = FrankaDuoActionSpec()
    prev = state
    for row in rows:
        spec.validate(row)
        assert np.linalg.norm(row[9:12] - prev[9:12]) <= 0.0101
        delta = rot6d_to_matrix(prev[12:18]).T @ rot6d_to_matrix(row[12:18])
        assert math.acos(min(1.0, (np.trace(delta) - 1) / 2)) <= 0.081
        prev = row
    closed = np.where(rows[:, 19] == 0.0)[0]
    assert len(closed) >= 14 and rows[closed[0] - 1, 19] == 1.0
    grasp_z = -0.12 - 0.055
    assert abs(rows[closed[0], 11] - grasp_z) < 1e-6
    assert abs(rows[-1, 11] - (grasp_z + 0.10)) < 1e-6
    assert np.allclose(rows[-1, 12:18], DATASET_GRASP_ROT6D["right"], atol=1e-3)
    # first approach row keeps the hand above the target, never below
    assert np.all(rows[: closed[0], 11] >= grasp_z - 1e-6)
    json.dumps(rows.tolist())


def test_expected_rim_axes_shrink_with_distance():
    t_base_from_cam = camera_pose()
    t_cam = invert_transform(t_base_from_cam)
    near = expected_rim_axes_px(np.array([0.35, 0.0, -0.12]), 0.07, K, t_cam)
    far = expected_rim_axes_px(np.array([0.60, 0.0, -0.12]), 0.07, K, t_cam)
    assert near[0] >= near[1] > 0
    assert far[0] < near[0], "the same rim must subtend fewer pixels further away"
    big = expected_rim_axes_px(np.array([0.35, 0.0, -0.12]), 0.115, K, t_cam)
    assert big[0] > near[0]


def test_refine_rim_ellipse_recovers_a_drawn_rim_and_rejects_wrong_size():
    # A pale cup body with a bright rim ellipse drawn on it, plus a distracting arc.
    img = np.full((360, 640, 3), 210, np.uint8)
    centre, axes = (300.0, 250.0), (86.0, 44.0)
    cv2.ellipse(img, (300, 250), (43, 22), 0, 0, 360, (40, 60, 50), 2)
    cv2.ellipse(img, (300, 300), (20, 8), 0, 0, 180, (30, 30, 30), 2)  # decoy near the base
    mask = np.zeros((360, 640), np.uint8)
    cv2.ellipse(mask, (300, 275), (48, 55), 0, 0, 360, 255, -1)

    points = rim_edge_points(img, mask)
    assert len(points) > 50

    result = refine_rim_ellipse(img, mask, expected_axes_px=(axes[0], axes[1]), seed=3)
    assert result is not None
    ellipse, inliers, total = result
    assert np.allclose(ellipse[0], centre, atol=3.0)
    assert abs(max(ellipse[1]) - axes[0]) < 12 and abs(min(ellipse[1]) - axes[1]) < 12
    assert 0 < inliers <= total

    # A rim three times too large cannot be explained by these edges.
    assert refine_rim_ellipse(img, mask, expected_axes_px=(260.0, 130.0), seed=3) is None


def test_rim_direction_picks_the_requested_image_side():
    t_base_from_cam = camera_pose()
    t_cam = invert_transform(t_base_from_cam)
    centre = np.array([0.365, 0.164, -0.165])
    radius = 0.0575
    left = rim_direction_for_side(centre, radius, "image_left", k_matrix=K, t_cam_from_base=t_cam)
    right = rim_direction_for_side(centre, radius, "image_right", k_matrix=K, t_cam_from_base=t_cam)
    assert abs(np.linalg.norm(left) - 1.0) < 1e-9
    # The two sides must be opposite ends of the same rim.
    assert float(np.dot(left, right)) < -0.9
    # Verify against the projection itself rather than assuming a base axis.
    left_px = project_points(centre + np.r_[radius * left, 0.0], K, t_cam)[0]
    right_px = project_points(centre + np.r_[radius * right, 0.0], K, t_cam)[0]
    assert left_px[0] < right_px[0]
    near = rim_direction_for_side(centre, radius, "image_near", k_matrix=K, t_cam_from_base=t_cam)
    far = rim_direction_for_side(centre, radius, "image_far", k_matrix=K, t_cam_from_base=t_cam)
    near_px = project_points(centre + np.r_[radius * near, 0.0], K, t_cam)[0]
    far_px = project_points(centre + np.r_[radius * far, 0.0], K, t_cam)[0]
    # "near" is lower in the image (larger v), "far" is higher up.
    assert near_px[1] > far_px[1]
    assert float(np.dot(near, far)) < -0.9
    with pytest.raises(ValueError):
        rim_direction_for_side(centre, radius, "north", k_matrix=K, t_cam_from_base=t_cam)


def test_yaw_align_closing_puts_the_finger_axis_along_the_rim_radius():
    base = np.asarray(DATASET_GRASP_ROT6D["left"])
    for angle in (0.0, 0.7, -2.1, 3.0):
        target = np.array([math.cos(angle), math.sin(angle)])
        aligned = yaw_align_closing(base, target)
        rot = rot6d_to_matrix(aligned).astype(np.float64)
        closing = rot[:2, 0] / np.linalg.norm(rot[:2, 0])
        # A closing axis is a line: either orientation counts as aligned.
        assert abs(abs(float(np.dot(closing, target))) - 1.0) < 1e-6
        # Only the yaw may change; the approach must keep pointing down.
        original = rot6d_to_matrix(base).astype(np.float64)
        assert abs(rot[2, 2] - original[2, 2]) < 1e-9
        assert rot[2, 2] < -0.9
    with pytest.raises(ValueError):
        yaw_align_closing(base, np.zeros(2))


def test_retreat_rows_interpolate_orientation_and_hold_grippers():
    state = np.zeros(20, np.float32)
    state[0:3] = (0.45, 0.30, -0.10)
    state[3:9] = DATASET_GRASP_ROT6D["left"]
    state[9:12] = (0.40, -0.16, -0.11)
    state[12:18] = DATASET_GRASP_ROT6D["right"]
    state[18:] = 1.0
    # A target rotated far from the current pose: the rows must walk there, not jump.
    target_rot = yaw_align_closing(np.asarray(DATASET_GRASP_ROT6D["left"]), np.array([0.0, 1.0]))
    rows = build_retreat_rows(
        state,
        "left",
        target_xyz=np.array([0.45, 0.37, -0.05]),
        target_rot6d=target_rot,
        step_m=0.01,
        step_rad=0.08,
        gripper=0.0,
        other_gripper=1.0,
    )
    prev = state
    for row in rows:
        step = rot6d_to_matrix(prev[3:9]).astype(np.float64).T @ rot6d_to_matrix(row[3:9]).astype(np.float64)
        angle = math.acos(max(-1.0, min(1.0, (np.trace(step) - 1) / 2)))
        assert angle <= 0.081, "each row must stay inside the rotation step limit"
        assert np.linalg.norm(row[0:3] - prev[0:3]) <= 0.0101
        prev = row
    assert np.allclose(rows[-1, 3:9], target_rot, atol=1e-6)
    assert np.allclose(rows[-1, 0:3], (0.45, 0.37, -0.05), atol=1e-6)
    assert np.all(rows[:, 18] == 0.0) and np.all(rows[:, 19] == 1.0)
    assert np.allclose(rows[:, 9:18], state[9:18])
