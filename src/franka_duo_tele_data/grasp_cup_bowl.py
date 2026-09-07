"""Detect the cup and bowl on the tray and grasp one with one arm through the joint servo.

Perception: yolo11n-seg on the head ZED frame gives instance masks; the cup is
the most confident ``cup``, the bowl is the ``bowl`` whose mask holds the most
dark (coffee bean) pixels, the plate is rejected by that ratio. The rim ellipse
centre of a mask is intersected with the horizontal plane at the object's rim
height using the per-arm ZED extrinsics from ``zed_pnp_calib``. The bowl grasp
point is the rim point opposite the blue spoon handle.

Motion: current EE pose -> pre-grasp above the target -> descend -> close the
gripper with dwell rows -> lift. Rows are 20D dataset-contract actions at 30 Hz
(the servo's ``playback_speed`` scales real time); the other arm holds its
current pose. Published exactly like ``smolvla_stream`` (absolute-step chunk,
``to_link0_action``, servo acknowledgement). Dry-run by default; robot output
needs ``--publish --enable-robot``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .action_spec import matrix_to_rot6d, rot6d_to_matrix
from .cartesian_chunk import pose_distance
from .config_io import load_mapping
from .joint_servo_client import chunk_payload, parse_status
from .mcap_to_lerobot import _gripper_position, invert_transform, load_geometry_manifest
from .rgb20d_io import RGB20DContract
from .zed_pnp_calib import (
    extrinsics_for_arm,
    intersect_height,
    intrinsics_from_camera_info,
    load_calibration,
    project_points,
)

SIDES = ("left", "right")
ARM_SLICE = {"left": slice(0, 9), "right": slice(9, 18)}
GRIPPER_INDEX = {"left": 18, "right": 19}
# Right-arm EE orientation from dataset episode 0 frame 178 (cup grasp) and left
# frame 186 (bowl grasp); both were executed under the impedance controller.
DATASET_GRASP_ROT6D = {
    "right": (0.9724, -0.1398, -0.1868, -0.1723, -0.9701, -0.1708),
    "left": (0.9766, 0.1708, -0.1310, 0.1827, -0.9795, 0.0845),
}


# --------------------------------------------------------------------------- perception (ROS-free)


@dataclasses.dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    mask: np.ndarray  # uint8 HxW, 255 inside


@dataclasses.dataclass(frozen=True)
class ObjectEstimate:
    cls: str
    conf: float
    rim_pixel: tuple[float, float]
    rim_axes_px: tuple[float, float]
    dark_ratio: float
    blue_ratio: float
    area_px: int


def _cv2():
    import cv2

    return cv2


def rim_ellipse(
    mask: np.ndarray, top_fraction: float = 0.45
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Ellipse (centre, axes) fitted to the upper boundary of a mask: the visible rim."""
    cv2 = _cv2()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty mask")
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    points = contour.reshape(-1, 2)
    top = points[points[:, 1] < y + top_fraction * h]
    if len(top) < 5:
        top = points
    (cx, cy), (a, b), _angle = cv2.fitEllipse(top.astype(np.float32))
    return (float(cx), float(cy)), (float(a), float(b))


def expected_rim_axes_px(centre_base, diameter_m: float, k_matrix, t_cam_from_base) -> tuple[float, float]:
    """Pixel length of ``diameter_m`` on the horizontal plane through ``centre_base``.

    Returned as ``(major, minor)``: the rim is a circle seen obliquely, so its
    longest projection is across the view and the shortest along it.
    """
    centre = np.asarray(centre_base, dtype=np.float64).reshape(3)
    half = float(diameter_m) / 2.0
    spans = []
    for axis in (0, 1):
        offset = np.zeros(3)
        offset[axis] = half
        ends = project_points(np.stack([centre - offset, centre + offset]), k_matrix, t_cam_from_base)
        spans.append(float(np.linalg.norm(ends[1] - ends[0])))
    return (max(spans), min(spans))


def _ellipse_radii(points: np.ndarray, ellipse) -> np.ndarray:
    """Normalised radius of each point in the ellipse frame (1.0 lies on the ellipse)."""
    (cx, cy), (axis1, axis2), angle = ellipse
    if axis1 <= 1e-6 or axis2 <= 1e-6:
        return np.full(len(points), np.inf)
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    delta = points - np.array([cx, cy])
    # OpenCV reports the angle of the FIRST axis; the pair is not sorted by length.
    u = delta[:, 0] * cos_t + delta[:, 1] * sin_t
    v = -delta[:, 0] * sin_t + delta[:, 1] * cos_t
    return np.sqrt((u / (axis1 / 2.0)) ** 2 + (v / (axis2 / 2.0)) ** 2)


def _axes_match(axes, expected_axes_px, tolerance: float) -> bool:
    """Compare a fitted axis pair with the expected one by magnitude, not by tuple order."""
    if expected_axes_px is None:
        return True
    wide, narrow = max(expected_axes_px), min(expected_axes_px)
    fit_wide, fit_narrow = max(axes), min(axes)
    return abs(fit_wide - wide) <= tolerance * max(wide, 1.0) and abs(fit_narrow - narrow) <= tolerance * max(
        narrow, 1.0
    )


def rim_edge_points(rgb: np.ndarray, mask: np.ndarray, *, dilate_px: int = 3) -> np.ndarray:
    """Canny edge pixels inside the (slightly dilated) object mask, as (N, 2) float."""
    cv2 = _cv2()
    image = np.ascontiguousarray(rgb)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray[mask > 0])) if int(cv2.countNonZero(mask)) else 128.0
    low = max(10.0, 0.66 * median)
    edges = cv2.Canny(gray, low, min(255.0, 2.0 * low))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    region = cv2.dilate(np.ascontiguousarray(mask, np.uint8), kernel)
    ys, xs = np.nonzero(cv2.bitwise_and(edges, region))
    return np.column_stack([xs, ys]).astype(np.float64)


def refine_rim_ellipse(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    expected_axes_px: tuple[float, float] | None = None,
    size_tolerance: float = 0.45,
    inlier_band: float = 0.08,
    iterations: int = 300,
    min_inliers: int = 30,
    seed: int = 0,
):
    """RANSAC ellipse fit to rim edges inside the ROI; ``None`` when it cannot be trusted.

    ``expected_axes_px`` comes from the calibration and the measured diameter,
    so a fit of the wrong arc or of the object's base is rejected on size.
    """
    cv2 = _cv2()
    points = rim_edge_points(rgb, mask)
    if len(points) < max(min_inliers, 5):
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 5, replace=False)]
        try:
            candidate = cv2.fitEllipse(sample.astype(np.float32))
        except cv2.error:
            continue
        if min(candidate[1]) <= 1.0:
            continue
        if not _axes_match(candidate[1], expected_axes_px, size_tolerance):
            continue
        inliers = np.abs(_ellipse_radii(points, candidate) - 1.0) < inlier_band
        count = int(inliers.sum())
        if best is None or count > best[0]:
            best = (count, inliers)
    if best is None or best[0] < min_inliers:
        return None
    refined = cv2.fitEllipse(points[best[1]].astype(np.float32))
    if not _axes_match(refined[1], expected_axes_px, size_tolerance):
        return None
    return refined, int(best[0]), int(len(points))


def summarize_detections(rgb: np.ndarray, detections: list[Detection]) -> list[ObjectEstimate]:
    cv2 = _cv2()
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    dark = cv2.inRange(hsv, np.array([0, 0, 0], np.uint8), np.array([180, 255, 60], np.uint8))
    blue = cv2.inRange(hsv, np.array([85, 40, 80], np.uint8), np.array([130, 255, 255], np.uint8))
    out = []
    for d in detections:
        mask = np.ascontiguousarray(d.mask, dtype=np.uint8)
        area = int(cv2.countNonZero(mask))
        if area < 50:
            continue
        centre, axes = rim_ellipse(mask)
        out.append(
            ObjectEstimate(
                d.cls,
                float(d.conf),
                centre,
                axes,
                cv2.countNonZero(cv2.bitwise_and(dark, mask)) / area,
                cv2.countNonZero(cv2.bitwise_and(blue, mask)) / area,
                area,
            )
        )
    return out


def spoon_axis(rgb: np.ndarray, detections: list[Detection]) -> np.ndarray | None:
    """Unit image-space direction of the spoon handle (from its mask, else from blue pixels)."""
    cv2 = _cv2()
    spoons = [d for d in detections if d.cls == "spoon"]
    if spoons:
        mask = max(spoons, key=lambda d: d.conf).mask
    else:
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([85, 40, 80], np.uint8), np.array([130, 255, 255], np.uint8))
    contours, _ = cv2.findContours(
        np.ascontiguousarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = [c for c in contours if cv2.contourArea(c) > 80]
    if not contours:
        return None
    vx, vy, _x, _y = cv2.fitLine(max(contours, key=cv2.contourArea), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([float(vx), float(vy)])


def spoon_centroid(rgb: np.ndarray, detections: list[Detection]) -> tuple[float, float] | None:
    """Pixel centroid of the spoon mask (blue pixels as a fallback)."""
    cv2 = _cv2()
    spoons = [d for d in detections if d.cls == "spoon"]
    if spoons:
        mask = np.ascontiguousarray(max(spoons, key=lambda d: d.conf).mask, np.uint8)
    else:
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([85, 40, 80], np.uint8), np.array([130, 255, 255], np.uint8))
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] <= 0:
        return None
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def pick_cup_and_bowl(
    objects: list[ObjectEstimate], *, min_dark_ratio: float = 0.25
) -> dict[str, ObjectEstimate]:
    """Most confident cup; the bowl is the 'bowl' with the most dark content (plate has none)."""
    result: dict[str, ObjectEstimate] = {}
    cups = [o for o in objects if o.cls == "cup"]
    if cups:
        result["cup"] = max(cups, key=lambda o: o.conf)
    bowls = [o for o in objects if o.cls == "bowl" and o.dark_ratio >= min_dark_ratio]
    if bowls:
        result["bowl"] = max(bowls, key=lambda o: o.dark_ratio)
    return result


def run_yolo(rgb: np.ndarray, weights: str, conf: float = 0.15) -> list[Detection]:
    from ultralytics import YOLO

    cv2 = _cv2()
    model = YOLO(weights)
    result = model.predict(rgb[:, :, ::-1].copy(), imgsz=640, conf=conf, verbose=False, device="cpu")[0]
    if result.masks is None:
        return []
    height, width = rgb.shape[:2]
    out = []
    for i, (cls, score) in enumerate(zip(result.boxes.cls.numpy(), result.boxes.conf.numpy(), strict=True)):
        mask = (result.masks.data[i].numpy() > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        out.append(Detection(model.names[int(cls)], float(score), mask))
    return out


def gripper_command_from_measured(
    position: float, *, open_position: float = 0.0, open_epsilon: float = 0.05
) -> float:
    """Command preserving what a gripper does: 1.0 only when it is nearly fully open.

    The binarised dataset state calls a gripper holding a thin object "open",
    which would release it. On a real run the measured position tells the truth.
    """
    return 1.0 if abs(float(position) - float(open_position)) <= open_epsilon else 0.0


def outboard_sign(arm: str) -> float:
    """+1 moves away from the other arm; left base sits at +y, right at -y."""
    if arm not in ARM_SLICE:
        raise ValueError("arm must be left or right")
    return 1.0 if arm == "left" else -1.0


@dataclasses.dataclass(frozen=True)
class GraspTarget:
    target: str
    arm: str
    rim_centre_base: np.ndarray
    grasp_point_base: np.ndarray
    yaw_hint: float | None
    rim_pixel: tuple[float, float]
    grasp_reason: str = "rim_centre"


def locate_target(
    target: str,
    arm: str,
    objects: dict[str, ObjectEstimate],
    *,
    k_matrix: np.ndarray,
    t_base_from_cam: np.ndarray,
    rim_z: float,
    spoon_pixel: tuple[float, float] | None,
    arm_base_xy: np.ndarray | None,
    bowl_rim_radius_m: float,
    spoon_near_factor: float = 0.9,
    bowl_side: str | None = None,
    rim_inset_m: float = 0.0,
) -> GraspTarget:
    """3D rim centre from the rim pixel at the known rim height.

    The cup is grasped at its rim centre.  For the bowl the gripper must sit on
    the rim, so a direction is needed: away from the spoon when the spoon is
    inside or at the bowl, otherwise toward this arm's own base, which is the
    shortest reach and keeps the arm out of the other one's way.
    """
    if target not in objects:
        raise ValueError(f"{target} not detected")
    obj = objects[target]
    centre = intersect_height(np.asarray(obj.rim_pixel), k_matrix, t_base_from_cam, rim_z)
    if target == "cup":
        return GraspTarget(target, arm, centre, centre, None, obj.rim_pixel, "rim_centre")
    rim_px = np.asarray(obj.rim_pixel, dtype=np.float64)
    spoon_at_bowl = spoon_pixel is not None and float(
        np.linalg.norm(np.asarray(spoon_pixel, dtype=np.float64) - rim_px)
    ) < spoon_near_factor * max(obj.rim_axes_px)
    if bowl_side is not None:
        # An explicit side wins: the operator picked which visible rim edge to hold.
        direction = rim_direction_for_side(
            centre,
            bowl_rim_radius_m,
            bowl_side,
            k_matrix=k_matrix,
            t_cam_from_base=invert_transform(t_base_from_cam),
        )
        reason = bowl_side
    elif spoon_at_bowl:
        spoon_base = intersect_height(np.asarray(spoon_pixel), k_matrix, t_base_from_cam, rim_z)
        direction = centre[:2] - spoon_base[:2]
        reason = "opposite_spoon"
    elif arm_base_xy is not None:
        direction = np.asarray(arm_base_xy, dtype=np.float64).reshape(2) - centre[:2]
        reason = "toward_arm_base"
    else:
        raise ValueError("bowl grasp needs the spoon position or the arm base")
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        raise ValueError("degenerate bowl grasp direction")
    direction = direction / norm
    grasp = centre.copy()
    # Sit slightly inside the rim: the fingers close on the wall just inboard of the
    # lip, which is where they actually keep hold of it.
    grasp[:2] += max(0.0, bowl_rim_radius_m - rim_inset_m) * direction
    return GraspTarget(
        target,
        arm,
        centre,
        grasp,
        float(math.atan2(direction[1], direction[0])),
        obj.rim_pixel,
        reason,
    )


def rim_direction_for_side(
    centre: np.ndarray,
    radius_m: float,
    side: str,
    *,
    k_matrix: np.ndarray,
    t_cam_from_base: np.ndarray,
    samples: int = 144,
) -> np.ndarray:
    """Unit 2D direction from the rim centre to the rim point on the requested image side.

    Sides are named in the image, not in the base frame, so a choice keeps meaning
    the same visible part of the rim when the tray or the camera moves.
    """
    picks = {
        "image_left": lambda px: int(np.argmin(px[:, 0])),
        "image_right": lambda px: int(np.argmax(px[:, 0])),
        "image_near": lambda px: int(np.argmax(px[:, 1])),
        "image_far": lambda px: int(np.argmin(px[:, 1])),
    }
    if side not in picks:
        raise ValueError(f"side must be one of {sorted(picks)}")
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    points = np.tile(np.asarray(centre, dtype=np.float64).reshape(3), (samples, 1))
    points[:, :2] += radius_m * directions
    pixels = project_points(points, k_matrix, t_cam_from_base)
    return directions[picks[side](pixels)]


def yaw_align_closing(rot6d: np.ndarray, closing_dir_2d: np.ndarray) -> np.ndarray:
    """Yaw ``rot6d`` about the vertical so the finger-closing axis lies along ``closing_dir_2d``.

    The gripper closes along its own x axis (the Robotiq knuckle joints turn about
    -y), so holding a rim needs that axis radial: one finger inside the bowl, one
    outside.  Only the yaw changes, which keeps the approach tilt that the dataset
    grasps proved reachable under the impedance controller.  The closing axis is a
    line, so the smaller of the two opposite yaws is used to avoid a wrist flip.
    """
    rot = rot6d_to_matrix(np.asarray(rot6d, dtype=np.float64)).astype(np.float64)
    current = rot[:2, 0]
    norm = float(np.linalg.norm(current))
    if norm < 1e-9:
        raise ValueError("gripper closing axis is vertical; cannot yaw-align it")
    current = current / norm
    target = np.asarray(closing_dir_2d, dtype=np.float64).reshape(2)
    target_norm = float(np.linalg.norm(target))
    if target_norm < 1e-9:
        raise ValueError("degenerate closing direction")
    target = target / target_norm
    if float(np.dot(current, target)) < 0.0:  # a closing axis is a line, not a ray
        target = -target
    delta = math.atan2(target[1], target[0]) - math.atan2(current[1], current[0])
    cos_d, sin_d = math.cos(delta), math.sin(delta)
    yaw = np.array([[cos_d, -sin_d, 0.0], [sin_d, cos_d, 0.0], [0.0, 0.0, 1.0]])
    return matrix_to_rot6d(yaw @ rot)


# --------------------------------------------------------------------------- trajectory (ROS-free)


def _slerp_rot6d(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    cv2 = _cv2()
    r0 = rot6d_to_matrix(start).astype(np.float64)
    r1 = rot6d_to_matrix(end).astype(np.float64)
    delta, _ = cv2.Rodrigues(r0.T @ r1)
    step, _ = cv2.Rodrigues(delta * alpha)
    return matrix_to_rot6d(r0 @ step)


def _segment(
    start: np.ndarray, end: np.ndarray, rot0: np.ndarray, rot1: np.ndarray, step_m: float, step_rad: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rows from start (exclusive) to end (inclusive) with bounded position and rotation steps."""
    cv2 = _cv2()
    delta, _ = cv2.Rodrigues(
        rot6d_to_matrix(rot0).astype(np.float64).T @ rot6d_to_matrix(rot1).astype(np.float64)
    )
    n = max(
        1, math.ceil(np.linalg.norm(end - start) / step_m), math.ceil(float(np.linalg.norm(delta)) / step_rad)
    )
    return [(start + (end - start) * i / n, _slerp_rot6d(rot0, rot1, i / n)) for i in range(1, n + 1)]


def build_retreat_rows(
    state: np.ndarray,
    arm: str,
    *,
    target_xyz: np.ndarray,
    target_rot6d: np.ndarray,
    step_m: float = 0.01,
    step_rad: float = 0.08,
    gripper: float | None = None,
    other_gripper: float | None = None,
) -> np.ndarray:
    """Move one arm to an absolute pose, keeping both grippers as they are."""
    sl = ARM_SLICE[arm]
    other = "left" if arm == "right" else "right"
    pos0 = state[sl][:3].astype(np.float64)
    rot0 = state[sl][3:9].astype(np.float64)
    target_pos = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    target_rot = np.asarray(target_rot6d, dtype=np.float64).reshape(6)
    poses = _segment(pos0, target_pos, rot0, target_rot, step_m, step_rad)
    if not poses:
        poses = [(target_pos, target_rot)]
    rows = np.tile(state.astype(np.float32), (len(poses), 1))
    for i, (p, r) in enumerate(poses):
        rows[i, sl] = np.concatenate([p, r]).astype(np.float32)
        if gripper is not None:
            rows[i, GRIPPER_INDEX[arm]] = float(gripper)
        if other_gripper is not None:
            rows[i, GRIPPER_INDEX[other]] = float(other_gripper)
    return rows


def build_grasp_rows(
    state: np.ndarray,
    grasp: GraspTarget,
    *,
    grasp_rot6d: np.ndarray,
    approach_height_m: float = 0.10,
    grasp_depth_below_rim_m: float = 0.055,
    lift_m: float = 0.10,
    step_m: float = 0.01,
    step_rad: float = 0.08,
    settle_rows: int = 6,
    close_rows: int = 14,
    retreat_target_xyz: np.ndarray | None = None,
    retreat_target_rot6d: np.ndarray | None = None,
    other_gripper: float | None = None,
    align_closing_to_rim: bool = False,
) -> np.ndarray:
    """20D rows: hold other arm, move one arm current -> above -> grasp -> close -> lift."""
    arm = grasp.arm
    sl = ARM_SLICE[arm]
    pos0 = state[sl][:3].astype(np.float64)
    rot0 = state[sl][3:9].astype(np.float64)
    grasp_rot = np.asarray(grasp_rot6d, dtype=np.float64)
    if align_closing_to_rim and grasp.yaw_hint is not None:
        # Pinch across the rim wall: the fingers close along the rim's tangent, so
        # the lip passes between them. Closing along the radius instead would push
        # the wall sideways out of the grip.
        grasp_rot = yaw_align_closing(
            grasp_rot, np.array([-math.sin(grasp.yaw_hint), math.cos(grasp.yaw_hint)])
        )
    grasp_pos = grasp.grasp_point_base.copy()
    grasp_pos[2] = grasp.rim_centre_base[2] - grasp_depth_below_rim_m
    above = grasp_pos + np.array([0.0, 0.0, approach_height_m])
    lifted = grasp_pos + np.array([0.0, 0.0, lift_m])
    poses: list[tuple[np.ndarray, np.ndarray, float]] = []
    open_value, closed_value = 1.0, 0.0
    for p, r in _segment(pos0, above, rot0, grasp_rot, step_m, step_rad):
        poses.append((p, r, open_value))
    for p, r in _segment(above, grasp_pos, grasp_rot, grasp_rot, step_m, step_rad):
        poses.append((p, r, open_value))
    poses += [(grasp_pos, grasp_rot, open_value)] * settle_rows
    poses += [(grasp_pos, grasp_rot, closed_value)] * close_rows
    for p, r in _segment(grasp_pos, lifted, grasp_rot, grasp_rot, step_m, step_rad):
        poses.append((p, r, closed_value))
    # Retreat: after lifting, keep going to the requested absolute pose (e.g. the park
    # pose) so the two arms cannot meet over the tray.
    if retreat_target_xyz is not None:
        retreat_pos = np.asarray(retreat_target_xyz, dtype=np.float64).reshape(3)
        retreat_rot = (
            np.asarray(retreat_target_rot6d, dtype=np.float64).reshape(6)
            if retreat_target_rot6d is not None
            else grasp_rot
        )
        for p, r in _segment(lifted, retreat_pos, grasp_rot, retreat_rot, step_m, step_rad):
            poses.append((p, r, closed_value))
    rows = np.tile(state.astype(np.float32), (len(poses), 1))
    other = "left" if arm == "right" else "right"
    for i, (p, r, g) in enumerate(poses):
        rows[i, sl] = np.concatenate([p, r]).astype(np.float32)
        rows[i, GRIPPER_INDEX[arm]] = g
        if other_gripper is not None:
            rows[i, GRIPPER_INDEX[other]] = float(other_gripper)
    return rows


def release_clearance_m(object_height_m: float, grasp_depth_m: float) -> float:
    """EE-origin height above the table when the held object's base rests on it.

    At grasp time the EE origin sits ``grasp_depth_m`` below the rim, so the
    object's base is ``object_height_m - grasp_depth_m`` below the EE origin.
    Placing puts that base on the table, which lifts the EE origin by the same
    amount.
    """
    clearance = float(object_height_m) - float(grasp_depth_m)
    if not math.isfinite(clearance) or clearance <= 0:
        raise ValueError("object height must exceed the grasp depth")
    return clearance


def build_place_rows(
    state: np.ndarray,
    arm: str,
    *,
    table_z: float,
    clearance_m: float,
    lift_m: float = 0.10,
    step_m: float = 0.01,
    step_rad: float = 0.08,
    settle_rows: int = 6,
    open_rows: int = 14,
    regrasp: bool = False,
    close_rows: int = 14,
    margin_m: float = 0.0,
) -> np.ndarray:
    """20D rows placing a held object: descend, open, optionally re-grasp, lift.

    The mirror image of :func:`build_grasp_rows`.  X/Y come from the arm's
    current pose, never from an earlier grasp point: the base has usually driven
    somewhere else by the time an object is put down, so a stored X/Y would send
    the arm to the wrong place.  Only z changes on the way down.

    With ``regrasp`` the gripper closes again after opening and lifts the object
    back up -- the object touches the table but is carried onward.
    """
    if arm not in SIDES:
        raise ValueError("arm must be left or right")
    if lift_m <= 0 or step_m <= 0 or step_rad <= 0:
        raise ValueError("lift and step limits must be positive")
    if settle_rows < 0 or open_rows < 1 or close_rows < 1:
        raise ValueError("open and close dwells must be at least one row")
    sl = ARM_SLICE[arm]
    pos0 = state[sl][:3].astype(np.float64)
    rot0 = state[sl][3:9].astype(np.float64)
    place_pos = pos0.copy()
    place_pos[2] = float(table_z) + float(clearance_m) + float(margin_m)
    if place_pos[2] > pos0[2] + 1e-9:
        raise ValueError(
            f"placement height {place_pos[2]:.4f} m is above the current EE z "
            f"{pos0[2]:.4f} m; the arm is already below the table plane"
        )
    lifted = place_pos + np.array([0.0, 0.0, float(lift_m)])
    open_value, closed_value = 1.0, 0.0

    poses: list[tuple[np.ndarray, np.ndarray, float]] = []
    # Straight down, orientation held: the object is already aligned by the grasp.
    for p, r in _segment(pos0, place_pos, rot0, rot0, step_m, step_rad):
        poses.append((p, r, closed_value))
    poses += [(place_pos, rot0, closed_value)] * settle_rows
    poses += [(place_pos, rot0, open_value)] * open_rows
    carried = open_value
    if regrasp:
        poses += [(place_pos, rot0, closed_value)] * close_rows
        carried = closed_value
    for p, r in _segment(place_pos, lifted, rot0, rot0, step_m, step_rad):
        poses.append((p, r, carried))

    rows = np.tile(state.astype(np.float32), (len(poses), 1))
    for i, (p, r, g) in enumerate(poses):
        rows[i, sl] = np.concatenate([p, r]).astype(np.float32)
        rows[i, GRIPPER_INDEX[arm]] = g
    return rows


def check_rows(
    rows: np.ndarray,
    state: np.ndarray,
    contract: RGB20DContract,
    *,
    max_step_m: float,
    max_step_rad: float,
    first_offset_m: float,
    first_offset_rad: float,
) -> dict[str, float]:
    previous = state
    max_pos = max_rot = 0.0
    for i, row in enumerate(rows):
        contract.validate_action(row)
        pos, rot = pose_distance(previous, row)
        limit_m, limit_rad = (first_offset_m, first_offset_rad) if i == 0 else (max_step_m, max_step_rad)
        if np.any(pos > limit_m) or np.any(rot > limit_rad):
            raise ValueError(f"row {i} exceeds step limit: m={pos.tolist()} rad={rot.tolist()}")
        max_pos, max_rot = max(max_pos, float(pos.max())), max(max_rot, float(rot.max()))
        previous = row
    return {"row_count": int(len(rows)), "max_step_m": max_pos, "max_step_rad": max_rot}


# --------------------------------------------------------------------------- ROS runtime


def _publish_grasp(
    rows,
    args,
    config,
    contract,
    node,
    publisher,
    status,
    reader,
    check_joint_servo_controllers,
) -> int:
    """Publish an action chunk to the joint servo and wait for it to finish."""
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension

    if publisher is None:
        print(json.dumps({"dry_run": True, "rows": int(len(rows))}), flush=True)
        return 0
    check_joint_servo_controllers(node, config, args.speed)
    deadline = time.monotonic() + 10
    while True:
        if node.count_publishers(config["joint_servo_chunk_topic"]) > 1:
            raise RuntimeError("another action-chunk publisher is running")
        current = status()
        if current is not None and publisher.get_subscription_count() == 1:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("joint servo status/subscriber not present")
        time.sleep(0.02)
    if current.started and not current.holding:
        raise RuntimeError("joint servo is still executing a chunk")
    start_step = int(math.ceil(current.step)) + args.append_lead_steps if current.started else 0
    link0 = np.stack([contract.action_spec.to_link0_action(row) for row in rows])
    data, dims, offset = chunk_payload(link0, start_step)
    message = Float32MultiArray(data=data)
    message.layout.dim = [
        MultiArrayDimension(label="rows", size=dims[0], stride=dims[0] * dims[1]),
        MultiArrayDimension(label="action", size=dims[1], stride=dims[1]),
    ]
    message.layout.data_offset = offset
    before = current.chunks
    publisher.publish(message)
    end_step = start_step + len(rows) - 1
    ack = time.monotonic() + 3
    while True:
        current = status()
        if current is not None and current.chunks > before:
            break
        if time.monotonic() > ack:
            raise TimeoutError("servo did not accept the chunk; inspect its IK/velocity log")
        time.sleep(0.02)
    print(
        json.dumps(
            {"accepted": True, "start_step": start_step, "end_step": end_step, "rows": int(len(rows))}
        ),
        flush=True,
    )
    deadline = time.monotonic() + 20 + len(rows) / (contract.fps * args.speed) * 1.5
    max_track = 0.0
    while True:
        time.sleep(0.05)
        current = status()
        if current is not None:
            max_track = max(max_track, current.tracking_error_rad)
            if current.holding and current.last_step >= end_step and current.step >= end_step:
                break
        if time.monotonic() > deadline:
            raise TimeoutError("servo did not finish the grasp chunk in time")
    final = reader.next(timeout_s=3)
    end_pos, end_rot = pose_distance(final.state, rows[-1])
    print(
        json.dumps(
            {
                "finished": True,
                "max_servo_tracking_error_rad": max_track,
                "end_position_error_m": end_pos.tolist(),
                "end_rotation_error_rad": end_rot.tolist(),
                "grippers": final.state[18:].tolist(),
            }
        ),
        flush=True,
    )
    return 0


def run(args) -> int:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from std_msgs.msg import Float32MultiArray, String

    from .replay_rgb20d import check_joint_servo_controllers
    from .rgb20d_io import RGB20DCache, RobotStateReader
    from .ros_utils import image_msg_to_rgb

    if args.publish != args.enable_robot:
        raise ValueError("robot publication requires both --publish and --enable-robot")
    contract = RGB20DContract(args.dataset)
    config = load_mapping(args.config)
    contract.action_spec = dataclasses.replace(
        contract.action_spec,
        workspace_min=tuple(config["workspace_min"]),
        workspace_max=tuple(config["workspace_max"]),
    )
    geometry = load_geometry_manifest(args.dataset / "franka_duo_extras" / "derived_manifest.json")
    arm_base = {"left": geometry.base_from_left_arm, "right": geometry.base_from_right_arm}[args.arm]
    arm_base_xy = np.asarray(arm_base, dtype=np.float64)[:2, 3]
    calibration = load_calibration(args.calibration)
    t_base_from_cam = extrinsics_for_arm(calibration, args.arm)
    latest: dict[str, Any] = {"image": None, "info": None, "status": None, "status_ns": 0}
    lock = threading.Lock()
    rclpy.init()
    node = rclpy.create_node("franka_duo_grasp_cup_bowl")
    cache = RGB20DCache(history_size=120)
    reader = RobotStateReader(cache, contract, max_age_ms=config["max_input_age_ms"])

    def store(key):
        def cb(msg):
            with lock:
                latest[key] = (time.monotonic(), msg)

        return cb

    node.create_subscription(Image, config["topics"]["head"], store("image"), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, args.camera_info_topic, store("info"), qos_profile_sensor_data)
    for side in SIDES:
        node.create_subscription(
            PoseStamped,
            config["topics"][f"{side}_pose"],
            getattr(cache, f"store_{side}_pose"),
            qos_profile_sensor_data,
        )
        node.create_subscription(
            JointState,
            config["topics"][f"{side}_gripper"],
            getattr(cache, f"store_{side}_gripper_states"),
            qos_profile_sensor_data,
        )

    def on_status(msg):
        with lock:
            latest["status"], latest["status_ns"] = msg.data, time.monotonic_ns()

    node.create_subscription(String, config["joint_servo_status_topic"], on_status, 10)
    publisher = (
        node.create_publisher(Float32MultiArray, config["joint_servo_chunk_topic"], 10)
        if args.publish
        else None
    )
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    def status():
        with lock:
            text, ns = latest["status"], latest["status_ns"]
        if text is None:
            return None
        if time.monotonic_ns() - ns > 500_000_000:
            raise TimeoutError("joint servo status expired")
        value = parse_status(text)
        if value.fault:
            raise RuntimeError(f"joint servo fault: {value.fault_reason}")
        return value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.image is not None:
            # Saved frame: perception from an image, robot state still live. The tray must
            # not have moved since the frame was taken; the robot pose does not matter.
            if args.publish:
                age_s = time.time() - args.image.stat().st_mtime
                if age_s > args.max_image_age_s:
                    raise ValueError(
                        f"{args.image} is {age_s:.0f} s old (limit {args.max_image_age_s:.0f} s); "
                        "grab a fresh frame before moving the robot"
                    )
                print(json.dumps({"saved_frame": str(args.image), "age_s": round(age_s, 1)}), flush=True)
            cv2 = _cv2()
            bgr = cv2.imread(str(args.image))
            if bgr is None:
                raise ValueError(f"cannot read {args.image}")
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            k_matrix = np.asarray(calibration["camera"]["K"], dtype=np.float64)
        else:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and (latest["image"] is None or latest["info"] is None):
                time.sleep(0.05)
            if latest["image"] is None or latest["info"] is None:
                raise TimeoutError("no ZED image/camera_info")
            k_matrix, _dist, _size = intrinsics_from_camera_info(latest["info"][1])
            calibrated_k = np.asarray(calibration["camera"]["K"], dtype=np.float64)
            drift = float(np.abs(k_matrix - calibrated_k).max())
            if drift > args.max_intrinsics_drift_px:
                raise ValueError(
                    f"live intrinsics differ from the calibration by {drift:.2f} px; recalibrate"
                )
            if drift > 0.1:
                print(json.dumps({"intrinsics_drift_px": round(drift, 3)}), flush=True)
            with lock:
                rgb = image_msg_to_rgb(latest["image"][1])
        observation = reader.next(timeout_s=3)
        state = observation.state
        # Preserve what each gripper physically does: a gripper holding a thin object
        # reads far from its closed travel but must NOT be released.
        measured = cache.snapshot()
        holds: dict[str, float] = {}
        for side in SIDES:
            samples = measured.get(f"{side}_gripper_states") or []
            if samples:
                holds[side] = gripper_command_from_measured(_gripper_position(samples[-1].message))
        other_arm = "left" if args.arm == "right" else "right"
        other_gripper = holds.get(other_arm)
        print(json.dumps({"gripper_hold": {k: round(v, 2) for k, v in holds.items()}}), flush=True)
        if args.park_target is not None:
            target = json.loads(Path(args.park_target).read_text(encoding="utf-8"))
            rows = build_retreat_rows(
                state,
                args.arm,
                target_xyz=target["xyz"],
                target_rot6d=target["rot6d"],
                gripper=holds.get(args.arm),
                other_gripper=other_gripper,
            )
            stats = check_rows(
                rows,
                state,
                contract,
                max_step_m=config["max_target_step_m"],
                max_step_rad=config["max_target_step_rad"],
                first_offset_m=args.max_first_offset_m,
                first_offset_rad=args.max_first_offset_rad,
            )
            record = {
                "schema": "franka_duo_grasp_cup_bowl_v1",
                "mode": "park_target",
                "target": target.get("name", str(args.park_target)),
                "arm": args.arm,
                "state": state.tolist(),
                "rows": rows.tolist(),
                **stats,
                "published": bool(publisher),
            }
            args.output.write_text(json.dumps(record, indent=1) + "\n")
            print(
                json.dumps({k: v for k, v in record.items() if k not in ("rows", "state")}),
                flush=True,
            )
            return _publish_grasp(
                rows, args, config, contract, node, publisher, status, reader, check_joint_servo_controllers
            )

        detections = run_yolo(rgb, args.weights, args.conf)
        objects = summarize_detections(rgb, detections)
        picked = pick_cup_and_bowl(objects)
        rim_z = (
            args.table_z
            + args.tray_thickness_m
            + (args.cup_height_m if args.target == "cup" else args.bowl_height_m)
        )
        # Stage 2: refine the coarse mask rim with an edge-fitted ellipse inside the ROI.
        refinement: dict[str, Any] = {"used": False}
        if args.target in picked and not args.no_refine:
            coarse = picked[args.target]
            diameter = args.cup_diameter_m if args.target == "cup" else 2 * args.bowl_rim_radius_m
            coarse_base = intersect_height(np.asarray(coarse.rim_pixel), k_matrix, t_base_from_cam, rim_z)
            expected = expected_rim_axes_px(
                coarse_base, diameter, k_matrix, invert_transform(t_base_from_cam)
            )
            match = None
            for d in detections:
                if d.cls == coarse.cls and abs(float(d.conf) - coarse.conf) < 1e-6:
                    match = d
                    break
            result = refine_rim_ellipse(rgb, match.mask, expected_axes_px=expected) if match else None
            refinement = {
                "used": result is not None,
                "expected_axes_px": [round(v, 1) for v in expected],
                "coarse_pixel": [round(v, 1) for v in coarse.rim_pixel],
            }
            if result is not None:
                ellipse, inlier_count, total = result
                picked[args.target] = dataclasses.replace(
                    coarse,
                    rim_pixel=(float(ellipse[0][0]), float(ellipse[0][1])),
                    rim_axes_px=(float(ellipse[1][0]), float(ellipse[1][1])),
                )
                refinement.update(
                    refined_pixel=[round(v, 1) for v in ellipse[0]],
                    refined_axes_px=[round(v, 1) for v in ellipse[1]],
                    angle_deg=round(float(ellipse[2]), 1),
                    inliers=inlier_count,
                    edge_points=total,
                    shift_px=round(float(np.linalg.norm(np.asarray(ellipse[0]) - coarse.rim_pixel)), 1),
                )
            print(json.dumps({"rim_refinement": refinement}), flush=True)
        grasp = locate_target(
            args.target,
            args.arm,
            picked,
            k_matrix=k_matrix,
            t_base_from_cam=t_base_from_cam,
            rim_z=rim_z,
            spoon_pixel=spoon_centroid(rgb, detections),
            arm_base_xy=arm_base_xy,
            bowl_rim_radius_m=args.bowl_rim_radius_m,
            bowl_side=args.bowl_side,
            rim_inset_m=args.rim_inset_m,
        )
        retreat_target = None
        if args.retreat_pose is not None:
            retreat_doc = json.loads(Path(args.retreat_pose).read_text(encoding="utf-8"))
            retreat_target = (retreat_doc["xyz"], retreat_doc["rot6d"])
        rows = build_grasp_rows(
            state,
            grasp,
            grasp_rot6d=np.asarray(DATASET_GRASP_ROT6D[args.arm]),
            approach_height_m=args.approach_height_m,
            grasp_depth_below_rim_m=args.grasp_depth_m,
            lift_m=args.lift_m,
            retreat_target_xyz=None if retreat_target is None else retreat_target[0],
            retreat_target_rot6d=None if retreat_target is None else retreat_target[1],
            other_gripper=other_gripper,
            align_closing_to_rim=args.target == "bowl",
        )
        stats = check_rows(
            rows,
            state,
            contract,
            max_step_m=config["max_target_step_m"],
            max_step_rad=config["max_target_step_rad"],
            first_offset_m=args.max_first_offset_m,
            first_offset_rad=args.max_first_offset_rad,
        )
        record = {
            "schema": "franka_duo_grasp_cup_bowl_v1",
            "target": args.target,
            "arm": args.arm,
            "detections": [
                dataclasses.asdict(o) | {"rim_pixel": list(o.rim_pixel), "rim_axes_px": list(o.rim_axes_px)}
                for o in objects
            ],
            "rim_centre_base": grasp.rim_centre_base.tolist(),
            "grasp_point_base": grasp.grasp_point_base.tolist(),
            "rim_refinement": refinement,
            "yaw_hint": grasp.yaw_hint,
            "grasp_reason": grasp.grasp_reason,
            "arm_base_xy": arm_base_xy.tolist(),
            "rim_z": rim_z,
            "state": state.tolist(),
            "rows": rows.tolist(),
            **stats,
            "published": bool(publisher),
        }
        args.output.write_text(json.dumps(record, indent=1) + "\n")
        cv2 = _cv2()
        vis = rgb[:, :, ::-1].copy()
        for o in objects:
            cv2.circle(vis, (int(o.rim_pixel[0]), int(o.rim_pixel[1])), 4, (0, 255, 0), -1)
            cv2.putText(
                vis,
                f"{o.cls} {o.conf:.2f} d{o.dark_ratio:.2f}",
                (int(o.rim_pixel[0]) + 5, int(o.rim_pixel[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )
        cv2.circle(vis, (int(grasp.rim_pixel[0]), int(grasp.rim_pixel[1])), 8, (0, 0, 255), 2)
        cv2.imwrite(str(args.output.with_suffix(".jpg")), vis)
        print(
            json.dumps({k: v for k, v in record.items() if k not in ("rows", "state", "detections")}),
            flush=True,
        )
        print(
            json.dumps(
                {
                    "objects": [
                        (o.cls, round(o.conf, 2), round(o.dark_ratio, 2), [round(x, 1) for x in o.rim_pixel])
                        for o in objects
                    ]
                }
            ),
            flush=True,
        )
        return _publish_grasp(
            rows, args, config, contract, node, publisher, status, reader, check_joint_servo_controllers
        )
    except KeyboardInterrupt:
        print(
            json.dumps({"stopped_by_user": True, "note": "accepted plan finishes; servo keeps holding"}),
            flush=True,
        )
        return 130
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        thread.join(timeout=2)
        node.destroy_node()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/tmr_rgb20d.yaml"))
    parser.add_argument("--calibration", type=Path, default=Path("configs/zed_pnp_calibration.json"))
    parser.add_argument("--camera-info-topic", default="/head_camera/zed/rgb/color/rect/camera_info")
    parser.add_argument("--weights", default="outputs/zed_pnp/yolo11n-seg.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--max-intrinsics-drift-px", type=float, default=2.0)
    parser.add_argument("--max-image-age-s", type=float, default=120.0)
    parser.add_argument("--image", type=Path, help="saved frame instead of the live ZED image (dry-run aid)")
    parser.add_argument("--target", choices=("cup", "bowl"), default="cup")
    parser.add_argument("--arm", choices=SIDES, default="right")
    parser.add_argument("--table-z", type=float, default=-0.220, help="table top in the midpoint base frame")
    parser.add_argument("--tray-thickness-m", type=float, default=0.010)
    parser.add_argument("--cup-height-m", type=float, default=0.080, help="measured 8 cm")
    parser.add_argument("--bowl-height-m", type=float, default=0.045, help="measured 4.5 cm")
    parser.add_argument("--bowl-rim-radius-m", type=float, default=0.0575, help="measured 11.5 cm diameter")
    parser.add_argument("--cup-diameter-m", type=float, default=0.070, help="measured 7 cm")
    parser.add_argument(
        "--bowl-side",
        choices=("image_left", "image_right", "image_near", "image_far"),
        default="image_near",
        help="which visible rim arc to hold; the fingers straddle it radially",
    )
    parser.add_argument(
        "--rim-inset-m",
        type=float,
        default=0.015,
        help="move the grasp point this far inboard from the rim, toward the bowl centre",
    )
    parser.add_argument("--no-refine", action="store_true", help="skip the stage-2 ellipse fit")
    parser.add_argument("--approach-height-m", type=float, default=0.10)
    parser.add_argument(
        "--grasp-depth-m",
        type=float,
        default=None,
        help="EE origin below the rim at grasp (default cup 5.5 cm, bowl 2.5 cm)",
    )
    parser.add_argument("--lift-m", type=float, default=0.10)
    parser.add_argument(
        "--retreat-pose",
        type=Path,
        help="JSON with xyz/rot6d: after grasping, continue moving this arm to that pose (clear the other arm)",
    )
    parser.add_argument(
        "--park-target",
        type=Path,
        help="JSON with xyz/rot6d: move this arm to that pose with no perception (e.g. park the grasping arm)",
    )
    parser.add_argument("--speed", type=float, default=0.3, help="must equal the servo playback_speed")
    parser.add_argument("--append-lead-steps", type=int, default=3)
    parser.add_argument("--max-first-offset-m", type=float, default=0.03)
    parser.add_argument("--max-first-offset-rad", type=float, default=0.3)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/grasp_cup_bowl.json"))
    args = parser.parse_args(argv)
    if not 0 < args.speed <= 1:
        parser.error("speed must be in (0,1]")
    if args.grasp_depth_m is None:
        args.grasp_depth_m = 0.055 if args.target == "cup" else 0.025
    height = args.cup_height_m if args.target == "cup" else args.bowl_height_m
    if not 0 < args.grasp_depth_m < height:
        parser.error("grasp depth must be positive and less than the object height")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
