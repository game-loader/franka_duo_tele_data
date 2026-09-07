"""Head-camera (ZED) extrinsic calibration by PnP against a marker held in the gripper.

A small saturated marker (orange tape square or a bead) sits at a known offset
``tool_offset_m`` along the EE z axis.  Each sample pairs the marker's 3D
position in the shared midpoint base (from ``current_pose`` and the dataset's
arm-base transforms) with its pixel centroid in the rectified ZED image.
``cv2.solvePnP`` then yields ``T_newbase_from_zed_optical``, which stays valid
when the mobile base moves on a flat floor because the camera is rigid to the
base.

Geometry, detection, solving, verification and the session logic are ROS-free
so they can be unit tested.  ``run`` adds the ROS subscriptions, a stdin
capture loop, a browser front end (``--web PORT``) for hosts without a desktop
session, and a ``--check`` mode for a one-point start-of-day check.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from .config_io import load_mapping
from .mcap_to_lerobot import compose_transform, invert_transform, load_geometry_manifest, pose_to_transform

SCHEMA_SAMPLES = "franka_duo_zed_pnp_samples_v1"
SCHEMA_CALIBRATION = "franka_duo_zed_pnp_calibration_v1"
SIDES = ("left", "right")
ORANGE_HUE = ((5, 28),)


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError("opencv-python(-headless) is required for ZED PnP calibration") from exc
    return cv2


# --------------------------------------------------------------------------- geometry


def intrinsics_from_camera_info(message: Any) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Return ``(K, distortion, (width, height))`` from ``sensor_msgs/CameraInfo``."""
    k = np.asarray(message.k, dtype=np.float64)
    if k.shape != (9,) or not np.isfinite(k).all():
        raise ValueError("camera_info.k must contain nine finite values")
    matrix = k.reshape(3, 3)
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], (0, 0, 1)):
        raise ValueError("camera_info.k is not a valid pinhole matrix")
    distortion = np.asarray(message.d, dtype=np.float64).reshape(-1)
    if distortion.size == 0:
        distortion = np.zeros(5)
    if not np.isfinite(distortion).all():
        raise ValueError("camera_info.d must be finite")
    size = (int(message.width), int(message.height))
    if min(size) <= 0:
        raise ValueError("camera_info width/height must be positive")
    return matrix, distortion, size


def bead_point_in_base(
    t_link0_ee: np.ndarray, t_base_from_link0: np.ndarray, tool_offset_m: float
) -> np.ndarray:
    """Marker centre in the midpoint base: EE origin plus ``tool_offset_m`` along the EE z axis."""
    if not math.isfinite(tool_offset_m):
        raise ValueError("tool_offset_m must be finite")
    t_base_ee = compose_transform(t_base_from_link0, t_link0_ee)
    return (t_base_ee @ np.array([0.0, 0.0, float(tool_offset_m), 1.0]))[:3].astype(np.float64)


def project_points(points: np.ndarray, k_matrix: np.ndarray, t_cam_from_base: np.ndarray) -> np.ndarray:
    """Project ``(N, 3)`` base-frame points to pixels; points behind the camera are rejected."""
    value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(t_cam_from_base, dtype=np.float64)
    camera = value @ t[:3, :3].T + t[:3, 3]
    if np.any(camera[:, 2] <= 1e-6):
        raise ValueError("point lies behind the camera")
    uv = camera[:, :2] / camera[:, 2:3]
    return uv * np.array([k_matrix[0, 0], k_matrix[1, 1]]) + np.array([k_matrix[0, 2], k_matrix[1, 2]])


def pixel_to_base_ray(
    pixel: np.ndarray, k_matrix: np.ndarray, t_base_from_cam: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(origin, unit direction)`` of a pixel's view ray in the base frame."""
    u, v = (float(x) for x in np.asarray(pixel, dtype=np.float64).reshape(2))
    direction_cam = np.array(
        [(u - k_matrix[0, 2]) / k_matrix[0, 0], (v - k_matrix[1, 2]) / k_matrix[1, 1], 1.0]
    )
    t = np.asarray(t_base_from_cam, dtype=np.float64)
    direction = t[:3, :3] @ direction_cam
    return t[:3, 3].copy(), direction / np.linalg.norm(direction)


def intersect_height(
    pixel: np.ndarray, k_matrix: np.ndarray, t_base_from_cam: np.ndarray, height_z: float
) -> np.ndarray:
    """Base-frame point where the pixel's view ray crosses the horizontal plane ``z = height_z``."""
    origin, direction = pixel_to_base_ray(pixel, k_matrix, t_base_from_cam)
    if abs(direction[2]) < 1e-6:
        raise ValueError("view ray is parallel to the plane")
    scale = (float(height_z) - origin[2]) / direction[2]
    if scale <= 0:
        raise ValueError("plane is behind the camera for this pixel")
    return origin + scale * direction


def transform_difference(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Return ``(translation_m, rotation_rad)`` between two rigid transforms."""
    delta = invert_transform(first).astype(np.float64) @ np.asarray(second, dtype=np.float64)
    angle = math.acos(max(-1.0, min(1.0, (np.trace(delta[:3, :3]) - 1) / 2)))
    return float(np.linalg.norm(delta[:3, 3])), float(angle)


# --------------------------------------------------------------------------- detection


@dataclasses.dataclass(frozen=True)
class BeadDetection:
    pixel: tuple[float, float]
    area_px: float
    circularity: float


def detect_bead(
    rgb: np.ndarray,
    *,
    hue_ranges: tuple[tuple[int, int], ...] = ORANGE_HUE,
    min_saturation: int = 100,
    min_value: int = 70,
    min_area_px: float = 20.0,
    min_circularity: float = 0.5,
) -> BeadDetection | None:
    """Largest saturated blob within the hue ranges; sub-pixel centroid from moments.

    The default hue range is orange (OpenCV hue 5..28).  Red would need two
    ranges, e.g. ``((0, 10), (170, 180))``.  A square marker has circularity
    about 0.78; the default threshold tolerates oblique views.
    """
    cv2 = _cv2()
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be a uint8 HWC image")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in hue_ranges:
        lower = np.array([int(low), int(min_saturation), int(min_value)], dtype=np.uint8)
        upper = np.array([int(high), 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area_px:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
        if circularity < min_circularity:
            continue
        if best is None or area > best[0]:
            best = (area, circularity, contour)
    if best is None:
        return None
    area, circularity, contour = best
    blob = np.zeros_like(mask)
    cv2.drawContours(blob, [contour], -1, 255, -1)
    moments = cv2.moments(cv2.bitwise_and(blob, mask), binaryImage=True)
    if moments["m00"] <= 0:
        return None
    centre = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    return BeadDetection(centre, area, circularity)


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    cv2 = _cv2()
    ok, buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buffer.tobytes()


# --------------------------------------------------------------------------- samples and solving


@dataclasses.dataclass(frozen=True)
class Sample:
    side: str
    pixel: tuple[float, float]
    point_base: tuple[float, float, float]
    t_link0_ee: tuple[float, ...]
    stamp_ns: int
    manual: bool = False
    area_px: float = 0.0
    frame_file: str = ""

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> Sample:
        pixel, point = value["pixel"], value["point_base"]
        if str(value["side"]) not in SIDES or len(pixel) != 2 or len(point) != 3:
            raise ValueError("malformed sample")
        sample = cls(
            side=str(value["side"]),
            pixel=(float(pixel[0]), float(pixel[1])),
            point_base=(float(point[0]), float(point[1]), float(point[2])),
            t_link0_ee=tuple(float(x) for x in value["t_link0_ee"]),
            stamp_ns=int(value["stamp_ns"]),
            manual=bool(value.get("manual", False)),
            area_px=float(value.get("area_px", 0.0)),
            frame_file=str(value.get("frame_file", "")),
        )
        if len(sample.t_link0_ee) != 16 or not all(math.isfinite(x) for x in sample.t_link0_ee):
            raise ValueError("sample t_link0_ee must be a finite 4x4 matrix")
        return sample


@dataclasses.dataclass(frozen=True)
class PnPResult:
    t_base_from_cam: np.ndarray
    t_cam_from_base: np.ndarray
    inliers: tuple[int, ...]
    reprojection_px: tuple[float, ...]
    leave_one_out_px: tuple[float, ...]
    leave_one_out_xy_m: tuple[float, ...]

    def summary(self) -> dict[str, Any]:
        reproj = np.asarray(self.reprojection_px)
        loo_px = np.asarray(self.leave_one_out_px)
        loo_m = np.asarray(self.leave_one_out_xy_m)
        return {
            "samples": int(reproj.size),
            "inliers": len(self.inliers),
            "reprojection_mean_px": float(reproj.mean()),
            "reprojection_max_px": float(reproj.max()),
            "leave_one_out_mean_px": float(loo_px.mean()) if loo_px.size else None,
            "leave_one_out_max_px": float(loo_px.max()) if loo_px.size else None,
            "leave_one_out_mean_xy_m": float(loo_m.mean()) if loo_m.size else None,
            "leave_one_out_max_xy_m": float(loo_m.max()) if loo_m.size else None,
        }


def _solve(
    object_points: np.ndarray, image_points: np.ndarray, k_matrix: np.ndarray, distortion: np.ndarray
) -> np.ndarray:
    cv2 = _cv2()
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, k_matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise ValueError("solvePnP failed")
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        k_matrix,
        distortion,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("solvePnP refinement failed")
    rotation, _ = cv2.Rodrigues(rvec)
    t_cam_from_base = np.eye(4)
    t_cam_from_base[:3, :3] = rotation
    t_cam_from_base[:3, 3] = tvec.reshape(3)
    return t_cam_from_base


def solve_extrinsics(
    object_points: np.ndarray,
    image_points: np.ndarray,
    k_matrix: np.ndarray,
    distortion: np.ndarray | None = None,
    *,
    ransac_threshold_px: float = 3.0,
    min_points: int = 6,
) -> PnPResult:
    """RANSAC-filtered PnP with per-point reprojection and leave-one-out errors.

    Leave-one-out xy error intersects the held-out pixel's ray with the plane
    at that point's true height, which is exactly how grasp points are computed.
    """
    cv2 = _cv2()
    points = np.ascontiguousarray(np.asarray(object_points, dtype=np.float64).reshape(-1, 3))
    pixels = np.ascontiguousarray(np.asarray(image_points, dtype=np.float64).reshape(-1, 2))
    if len(points) != len(pixels) or len(points) < min_points:
        raise ValueError(f"need at least {min_points} matching point pairs, got {len(points)}")
    if not np.isfinite(points).all() or not np.isfinite(pixels).all():
        raise ValueError("points must be finite")
    if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-3) < 3:
        raise ValueError("points are coplanar; spread samples over several heights")
    dist = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=np.float64).reshape(-1)
    inliers = np.arange(len(points))
    if len(points) >= 8:
        ok, _rvec, _tvec, mask = cv2.solvePnPRansac(
            points,
            pixels,
            k_matrix,
            dist,
            reprojectionError=float(ransac_threshold_px),
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if ok and mask is not None and len(mask) >= min_points:
            inliers = np.sort(mask.reshape(-1))
    t_cam_from_base = _solve(points[inliers], pixels[inliers], k_matrix, dist)
    t_base_from_cam = invert_transform(t_cam_from_base).astype(np.float64)
    reprojection = np.linalg.norm(project_points(points, k_matrix, t_cam_from_base) - pixels, axis=1)
    loo_px, loo_xy = [], []
    if len(inliers) > min_points:
        for index in inliers:
            keep = inliers[inliers != index]
            t_loo = _solve(points[keep], pixels[keep], k_matrix, dist)
            predicted = project_points(points[index], k_matrix, t_loo)[0]
            loo_px.append(float(np.linalg.norm(predicted - pixels[index])))
            hit = intersect_height(pixels[index], k_matrix, invert_transform(t_loo), points[index, 2])
            loo_xy.append(float(np.linalg.norm(hit[:2] - points[index, :2])))
    return PnPResult(
        t_base_from_cam=t_base_from_cam,
        t_cam_from_base=t_cam_from_base,
        inliers=tuple(int(i) for i in inliers),
        reprojection_px=tuple(float(x) for x in reprojection),
        leave_one_out_px=tuple(loo_px),
        leave_one_out_xy_m=tuple(loo_xy),
    )


def per_side_report(
    samples: list[Sample], pixels: np.ndarray, projected: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Per arm: count, mean/max error and mean signed residual.

    A systematic signed residual on one arm only points at that arm's base transform.
    """
    report: dict[str, dict[str, Any]] = {}
    errors = np.linalg.norm(projected - pixels, axis=1)
    for side in SIDES:
        idx = [i for i, s in enumerate(samples) if s.side == side]
        if idx:
            report[side] = {
                "count": len(idx),
                "mean_px": float(errors[idx].mean()),
                "max_px": float(errors[idx].max()),
                "mean_signed_residual_px": (projected[idx] - pixels[idx]).mean(axis=0).tolist(),
            }
    return report


def calibrate(
    samples: list[Sample],
    k_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    *,
    tool_offset_m: float,
    nominal_t_base_from_cam: np.ndarray | None,
    ransac_threshold_px: float = 3.0,
    per_arm: bool = False,
) -> dict[str, Any]:
    """Solve from samples and build the calibration document (JSON-compatible).

    With ``per_arm`` an additional extrinsic is solved from each arm's own
    samples; this absorbs an error in that arm's base-to-midpoint transform,
    which shows up as opposite signed residuals per arm in the joint solve.
    """
    if not samples:
        raise ValueError("no samples")
    points = np.asarray([s.point_base for s in samples], dtype=np.float64)
    pixels = np.asarray([s.pixel for s in samples], dtype=np.float64)
    result = solve_extrinsics(points, pixels, k_matrix, distortion, ransac_threshold_px=ransac_threshold_px)
    projected = project_points(points, k_matrix, result.t_cam_from_base)
    document: dict[str, Any] = {
        "schema": SCHEMA_CALIBRATION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frame_convention": "p_base = T_newbase_from_zed_optical @ p_cam; base is the dataset midpoint frame",
        "tool_offset_m": float(tool_offset_m),
        "camera": {
            "K": k_matrix.tolist(),
            "distortion": distortion.tolist(),
            "width": image_size[0],
            "height": image_size[1],
        },
        "T_newbase_from_zed_optical": result.t_base_from_cam.tolist(),
        "T_zed_optical_from_newbase": result.t_cam_from_base.tolist(),
        "inliers": list(result.inliers),
        "reprojection_px": list(result.reprojection_px),
        "leave_one_out_px": list(result.leave_one_out_px),
        "leave_one_out_xy_m": list(result.leave_one_out_xy_m),
        "stats": result.summary(),
        "per_side": per_side_report(samples, pixels, projected),
        "samples": [s.to_json() for s in samples],
    }
    if nominal_t_base_from_cam is not None:
        translation, rotation = transform_difference(nominal_t_base_from_cam, result.t_base_from_cam)
        document["nominal_comparison"] = {
            "translation_m": translation,
            "rotation_rad": rotation,
            "nominal_T_newbase_from_zed_optical": np.asarray(nominal_t_base_from_cam).tolist(),
        }
    if per_arm:
        document["per_arm"] = {}
        for side in SIDES:
            idx = [i for i, s in enumerate(samples) if s.side == side]
            entry: dict[str, Any] = {"count": len(idx), "sample_indices": idx}
            try:
                side_result = solve_extrinsics(
                    points[idx], pixels[idx], k_matrix, distortion, ransac_threshold_px=ransac_threshold_px
                )
                entry.update(
                    {
                        "T_newbase_from_zed_optical": side_result.t_base_from_cam.tolist(),
                        "inliers": [idx[i] for i in side_result.inliers],
                        "stats": side_result.summary(),
                    }
                )
                joint_t, joint_r = transform_difference(result.t_base_from_cam, side_result.t_base_from_cam)
                entry["difference_to_joint"] = {"translation_m": joint_t, "rotation_rad": joint_r}
            except ValueError as exc:
                entry["error"] = str(exc)
            document["per_arm"][side] = entry
    return document


def extrinsics_for_arm(document: dict[str, Any], side: str) -> np.ndarray:
    """Per-arm ``T_newbase_from_zed_optical`` when present and solved, else the joint one."""
    if side not in SIDES:
        raise ValueError("side must be left or right")
    entry = document.get("per_arm", {}).get(side, {})
    matrix = entry.get("T_newbase_from_zed_optical", document["T_newbase_from_zed_optical"])
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("calibration transform must be a finite 4x4")
    return value


def load_calibration(path: Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA_CALIBRATION:
        raise ValueError("expected a ZED PnP calibration document")
    matrix = np.asarray(document["T_newbase_from_zed_optical"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], (0, 0, 0, 1)):
        raise ValueError("calibration transform must be a finite homogeneous 4x4")
    invert_transform(matrix)  # raises on a non-rigid rotation
    return document


def load_samples(path: Path) -> tuple[list[Sample], dict[str, Any]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA_SAMPLES:
        raise ValueError("expected a ZED PnP samples document")
    return [Sample.from_json(v) for v in document["samples"]], document


def write_samples(path: Path, samples: list[Sample], camera: dict[str, Any], tool_offset_m: float) -> None:
    payload = {
        "schema": SCHEMA_SAMPLES,
        "tool_offset_m": tool_offset_m,
        "camera": camera,
        "samples": [s.to_json() for s in samples],
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def report_summary(document: dict[str, Any]) -> dict[str, Any]:
    payload = {k: document[k] for k in ("stats", "per_side") if k in document}
    if "nominal_comparison" in document:
        payload["nominal_comparison"] = {
            k: v
            for k, v in document["nominal_comparison"].items()
            if k != "nominal_T_newbase_from_zed_optical"
        }
    payload["inliers"] = document["inliers"]
    if "per_arm" in document:
        payload["per_arm"] = {
            side: {
                k: v for k, v in entry.items() if k not in ("T_newbase_from_zed_optical", "sample_indices")
            }
            for side, entry in document["per_arm"].items()
        }
    errors = document["reprojection_px"]
    worst = sorted(range(len(errors)), key=lambda i: -errors[i])[:3]
    payload["worst_samples"] = [
        {"index": i, "side": document["samples"][i]["side"], "reprojection_px": round(errors[i], 2)}
        for i in worst
    ]
    return payload


def print_report(document: dict[str, Any]) -> None:
    print(json.dumps(report_summary(document), indent=1), flush=True)


# --------------------------------------------------------------------------- session (front-end agnostic)


@dataclasses.dataclass
class Capture:
    side: str
    rgb: np.ndarray
    t_link0_ee: np.ndarray
    point: np.ndarray
    detection: BeadDetection | None
    stamp_ns: int


class CalibrationSession:
    """Sample bookkeeping shared by the stdin loop and the web front end.

    ``capture_fn(side)`` freezes one image/pose pair; ``live_rgb_fn()`` returns
    the newest frame; ``solve_fn(samples)`` returns a calibration document.
    """

    def __init__(
        self,
        *,
        capture_fn: Callable[[str], Capture],
        live_rgb_fn: Callable[[], np.ndarray],
        solve_fn: Callable[[list[Sample]], dict[str, Any]],
        samples: list[Sample],
        samples_path: Path,
        output_dir: Path,
        output: Path,
        camera_doc: dict[str, Any] | None,
        tool_offset_m: float,
    ):
        self.capture_fn = capture_fn
        self.live_rgb_fn = live_rgb_fn
        self.solve_fn = solve_fn
        self.samples = samples
        self.samples_path = samples_path
        self.output_dir = output_dir
        self.output = output
        self.camera_doc = camera_doc
        self.tool_offset_m = tool_offset_m
        self.pending: Capture | None = None
        self.lock = threading.Lock()

    def _persist(self) -> None:
        write_samples(self.samples_path, self.samples, self.camera_doc or {}, self.tool_offset_m)

    def state(self) -> dict[str, Any]:
        return {
            "samples": [
                {
                    "index": i,
                    "side": s.side,
                    "pixel": [round(s.pixel[0], 1), round(s.pixel[1], 1)],
                    "point_base": [round(x, 4) for x in s.point_base],
                    "manual": s.manual,
                }
                for i, s in enumerate(self.samples)
            ],
            "pending": None if self.pending is None else self.pending.side,
            "tool_offset_m": self.tool_offset_m,
            "camera_ready": self.camera_doc is not None,
        }

    def capture(self, side: str) -> dict[str, Any]:
        if side not in SIDES:
            raise ValueError("arm must be left or right")
        with self.lock:
            self.pending = self.capture_fn(side)
            detection = self.pending.detection
            return {
                "side": side,
                "point_base": [round(float(x), 4) for x in self.pending.point],
                "auto_pixel": None if detection is None else list(detection.pixel),
                "auto_area_px": None if detection is None else detection.area_px,
                "auto_circularity": None if detection is None else detection.circularity,
                "width": int(self.pending.rgb.shape[1]),
                "height": int(self.pending.rgb.shape[0]),
            }

    def accept(self, u: float, v: float, manual: bool) -> dict[str, Any]:
        with self.lock:
            pending = self.pending
            if pending is None:
                raise ValueError("capture a frame first")
            height, width = pending.rgb.shape[:2]
            if not (math.isfinite(u) and math.isfinite(v) and -0.5 <= u < width and -0.5 <= v < height):
                raise ValueError("pixel outside the image")
            frame_file = f"frame_{len(self.samples):02d}_{pending.side}.png"
            cv2 = _cv2()
            cv2.imwrite(str(self.output_dir / frame_file), cv2.cvtColor(pending.rgb, cv2.COLOR_RGB2BGR))
            area = 0.0 if pending.detection is None else pending.detection.area_px
            self.samples.append(
                Sample(
                    pending.side,
                    (float(u), float(v)),
                    tuple(float(x) for x in pending.point),
                    tuple(float(x) for x in pending.t_link0_ee.reshape(-1)),
                    pending.stamp_ns,
                    bool(manual),
                    float(area),
                    frame_file,
                )
            )
            self.pending = None
            self._persist()
            return self.state()

    def discard(self) -> dict[str, Any]:
        with self.lock:
            self.pending = None
            return self.state()

    def drop(self, index: int) -> dict[str, Any]:
        with self.lock:
            if not 0 <= index < len(self.samples):
                raise ValueError("sample index out of range")
            self.samples.pop(index)
            self._persist()
            return self.state()

    def solve(self, write: bool) -> dict[str, Any]:
        with self.lock:
            if len(self.samples) < 6:
                raise ValueError("need at least 6 samples spread over several heights")
            document = self.solve_fn(list(self.samples))
            if write:
                self.output.parent.mkdir(parents=True, exist_ok=True)
                self.output.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
            summary = report_summary(document)
            summary["written"] = str(self.output) if write else None
            return summary

    def live_jpeg(self) -> bytes:
        return encode_jpeg(self.live_rgb_fn(), 80)

    def pending_jpeg(self) -> bytes:
        with self.lock:
            if self.pending is None:
                raise ValueError("no frozen frame")
            return encode_jpeg(self.pending.rgb, 95)


# --------------------------------------------------------------------------- web front end

INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>ZED PnP calibration</title>
<style>
body{font-family:sans-serif;margin:12px;background:#1e1e1e;color:#ddd}
button,select,input{font-size:15px;margin:2px}
#wrap{position:relative;display:inline-block;margin-top:8px;border:1px solid #555}
#img{display:block;image-rendering:pixelated;cursor:crosshair}
.mark{position:absolute;pointer-events:none;width:0;height:0}
.mark:before,.mark:after{content:"";position:absolute;background:currentColor}
.mark:before{left:-12px;top:-1px;width:24px;height:2px}
.mark:after{left:-1px;top:-12px;width:2px;height:24px}
#pick{color:#ff0;display:none}#auto{color:#0f0;display:none}
#msg{min-height:1.4em;margin:6px 0;color:#fc6}
pre{background:#111;padding:8px;max-height:220px;overflow:auto}
</style></head><body>
<div>
 arm <select id="arm"><option>left</option><option>right</option></select>
 <button id="cap">Capture (freeze)</button>
 <button id="acc" disabled>Accept point</button>
 <button id="dis" disabled>Discard</button>
 zoom <input id="zoom" type="range" min="1" max="4" step="0.5" value="2">
 <button id="solve">Solve</button>
 <button id="save">Solve + write</button>
 drop # <input id="dropi" type="number" min="0" style="width:4em"><button id="drop">Drop</button>
</div>
<div id="msg">live view. Hold the arm still, then Capture; click the fingertip/marker centre; Accept.</div>
<div id="wrap"><img id="img"><div id="pick" class="mark"></div><div id="auto" class="mark"></div></div>
<pre id="samples"></pre><pre id="report"></pre>
<script>
const $=id=>document.getElementById(id);
let frozen=false, pick=null, auto=null, manual=true, live=null;
function msg(t){$('msg').textContent=t;}
async function post(path,body){const r=await fetch(path,{method:'POST',body:JSON.stringify(body||{})});
  const j=await r.json(); if(!r.ok) throw new Error(j.error||r.status); return j;}
function place(el,p){if(!p){el.style.display='none';return;}const img=$('img');
  el.style.left=((p[0]+0.5)/img.naturalWidth*img.clientWidth)+'px';
  el.style.top=((p[1]+0.5)/img.naturalHeight*img.clientHeight)+'px'; el.style.display='block';}
function redraw(){place($('pick'),pick);place($('auto'),auto);}
function setZoom(){const img=$('img'); if(img.naturalWidth) img.style.width=(img.naturalWidth*$('zoom').value)+'px'; redraw();}
function startLive(){frozen=false; pick=null; auto=null; redraw(); $('acc').disabled=true; $('dis').disabled=true;
  if(live) clearInterval(live); live=setInterval(()=>{ if(!frozen) $('img').src='/live.jpg?'+Date.now(); },500);}
$('img').onload=setZoom; $('zoom').oninput=setZoom;
$('img').onclick=e=>{ if(!frozen) return; const img=$('img'), r=img.getBoundingClientRect();
  pick=[(e.clientX-r.left)/r.width*img.naturalWidth-0.5,(e.clientY-r.top)/r.height*img.naturalHeight-0.5];
  manual=true; $('acc').disabled=false; redraw(); msg('pixel '+pick[0].toFixed(1)+', '+pick[1].toFixed(1)+' (manual)'); };
$('cap').onclick=async()=>{ try{ const j=await post('/capture',{arm:$('arm').value});
  frozen=true; clearInterval(live); live=null; $('img').src='/pending.jpg?'+Date.now(); $('dis').disabled=false;
  auto=j.auto_pixel; pick=j.auto_pixel; manual=false; $('acc').disabled=!pick;
  msg('frozen '+j.side+' base '+JSON.stringify(j.point_base)+(auto?' auto-detect at '+auto.map(x=>x.toFixed(1))+' (green); click to override':' no marker detected; click it'));
  setTimeout(redraw,200);}catch(e){msg('capture rejected: '+e.message);} };
$('acc').onclick=async()=>{ try{ const j=await post('/accept',{u:pick[0],v:pick[1],manual:manual});
  show(j); msg('sample '+(j.samples.length-1)+' accepted'); startLive(); }catch(e){msg(e.message);} };
$('dis').onclick=async()=>{ await post('/discard'); msg('discarded'); startLive(); };
$('drop').onclick=async()=>{ try{ show(await post('/drop',{index:parseInt($('dropi').value)})); }catch(e){msg(e.message);} };
async function solve(w){ try{ const j=await post('/solve',{write:w}); $('report').textContent=JSON.stringify(j,null,1);
  msg(w?'written '+j.written:'solved (not written)'); }catch(e){msg('solve failed: '+e.message);} }
$('solve').onclick=()=>solve(false); $('save').onclick=()=>solve(true);
function show(s){ $('samples').textContent=s.samples.map(x=>x.index+' '+x.side+' px '+x.pixel+' base '+x.point_base+(x.manual?' manual':' auto')).join('\\n')||'(no samples)'; }
fetch('/state').then(r=>r.json()).then(show); startLive();
</script></body></html>
"""


class CalibrationWebApp:
    """HTTP routing for the browser front end; ``handle`` is socket-free for tests."""

    def __init__(self, session: CalibrationSession):
        self.session = session

    @staticmethod
    def _json(status: int, payload: Any) -> tuple[int, str, bytes]:
        return status, "application/json", json.dumps(payload).encode("utf-8")

    def handle(self, method: str, path: str, body: bytes) -> tuple[int, str, bytes]:
        route = path.split("?", 1)[0]
        try:
            if method == "GET":
                if route == "/":
                    return 200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8")
                if route == "/state":
                    return self._json(200, self.session.state())
                if route == "/live.jpg":
                    return 200, "image/jpeg", self.session.live_jpeg()
                if route == "/pending.jpg":
                    return 200, "image/jpeg", self.session.pending_jpeg()
            elif method == "POST":
                payload = json.loads(body.decode("utf-8") or "{}") if body else {}
                if not isinstance(payload, dict):
                    raise ValueError("JSON object expected")
                if route == "/capture":
                    return self._json(200, self.session.capture(str(payload.get("arm", ""))))
                if route == "/accept":
                    accepted = self.session.accept(
                        float(payload["u"]), float(payload["v"]), bool(payload.get("manual", True))
                    )
                    return self._json(200, accepted)
                if route == "/discard":
                    return self._json(200, self.session.discard())
                if route == "/drop":
                    return self._json(200, self.session.drop(int(payload["index"])))
                if route == "/solve":
                    return self._json(200, self.session.solve(bool(payload.get("write", False))))
            return self._json(404, {"error": "not found"})
        except (KeyError, ValueError, TypeError, TimeoutError, RuntimeError) as exc:
            return self._json(400, {"error": str(exc)})


def serve_web(app: CalibrationWebApp, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            status, content_type, data = app.handle(method, self.path, body)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 - http.server API
            self._reply("GET")

        def do_POST(self):  # noqa: N802 - http.server API
            self._reply("POST")

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


# --------------------------------------------------------------------------- ROS capture


class _Latest:
    def __init__(self):
        self.lock = threading.Lock()
        self.image = None
        self.camera_info = None
        self.poses: dict[str, list[tuple[float, Any]]] = {side: [] for side in SIDES}

    def store_image(self, message):
        with self.lock:
            self.image = (time.monotonic(), message)

    def store_info(self, message):
        with self.lock:
            self.camera_info = message

    def store_pose(self, side, message):
        with self.lock:
            history = self.poses[side]
            history.append((time.monotonic(), message))
            del history[:-60]


def stationary_pose(
    history: list[tuple[float, Any]], *, now: float, window_s: float, tolerance_m: float
) -> Any:
    """Return the newest pose if the EE stayed within ``tolerance_m`` over the window."""
    if not history:
        raise TimeoutError("no current_pose received")
    newest_time, newest = history[-1]
    if now - newest_time > 0.5:
        raise TimeoutError("current_pose is stale")
    recent = [m for t, m in history if now - t <= window_s]
    if len(recent) < 2:
        raise TimeoutError("not enough pose history to confirm the arm is stationary")
    positions = np.asarray([[m.pose.position.x, m.pose.position.y, m.pose.position.z] for m in recent])
    if np.ptp(positions, axis=0).max() > tolerance_m:
        raise RuntimeError("arm is still moving; hold it still and try again")
    return newest


def hue_ranges_from_args(values: list[list[int]] | None) -> tuple[tuple[int, int], ...]:
    if not values:
        return ORANGE_HUE
    ranges = tuple((int(low), int(high)) for low, high in values)
    for low, high in ranges:
        if not 0 <= low < high <= 180:
            raise ValueError("--hue-range needs 0 <= LOW < HIGH <= 180 (OpenCV hue)")
    return ranges


def run(args) -> int:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    from .ros_utils import _stamp_ns, image_msg_to_rgb

    _cv2()
    hue_ranges = hue_ranges_from_args(args.hue_range)
    config = load_mapping(args.config)
    geometry = load_geometry_manifest(args.manifest)
    t_base_from_link0 = {
        "left": geometry.base_from_left_arm.astype(np.float64),
        "right": geometry.base_from_right_arm.astype(np.float64),
    }
    nominal = geometry.base_from_zed_optical.astype(np.float64)
    calibration = load_calibration(args.check) if args.check else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "samples.json"
    samples: list[Sample] = []
    if args.resume and samples_path.exists():
        samples, _ = load_samples(samples_path)
        print(f"resumed {len(samples)} samples from {samples_path}", flush=True)

    latest = _Latest()
    rclpy.init()
    node = rclpy.create_node("franka_duo_zed_pnp_calib")
    node.create_subscription(Image, config["topics"]["head"], latest.store_image, qos_profile_sensor_data)
    node.create_subscription(CameraInfo, args.camera_info_topic, latest.store_info, qos_profile_sensor_data)
    for side in SIDES:
        node.create_subscription(
            PoseStamped,
            config["topics"][f"{side}_pose"],
            lambda m, s=side: latest.store_pose(s, m),
            qos_profile_sensor_data,
        )
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    camera_cache: dict[str, Any] = {}

    def camera(timeout_s: float = 5.0):
        """Intrinsics from the first CameraInfo; cached afterwards."""
        if "k" not in camera_cache:
            deadline = time.monotonic() + timeout_s
            while latest.camera_info is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if latest.camera_info is None:
                raise TimeoutError(f"no CameraInfo on {args.camera_info_topic}")
            k_matrix, distortion, size = intrinsics_from_camera_info(latest.camera_info)
            camera_cache.update(k=k_matrix, distortion=distortion, size=size)
            camera_cache["doc"] = {
                "K": k_matrix.tolist(),
                "distortion": distortion.tolist(),
                "width": size[0],
                "height": size[1],
            }
            session.camera_doc = camera_cache["doc"]
        return camera_cache["k"], camera_cache["distortion"], camera_cache["size"]

    def live_rgb() -> np.ndarray:
        with latest.lock:
            image = latest.image
        if image is None or time.monotonic() - image[0] > 2.0:
            raise TimeoutError(f"no fresh ZED image on {config['topics']['head']}")
        return image_msg_to_rgb(image[1])

    def capture(side: str) -> Capture:
        camera(2.0)
        with latest.lock:
            image = latest.image
            history = list(latest.poses[side])
        if image is None or time.monotonic() - image[0] > 0.5:
            raise TimeoutError("no fresh ZED image")
        pose = stationary_pose(history, now=time.monotonic(), window_s=0.5, tolerance_m=0.001)
        rgb = image_msg_to_rgb(image[1])
        t_link0_ee = pose_to_transform(pose.pose).astype(np.float64)
        point = bead_point_in_base(t_link0_ee, t_base_from_link0[side], args.tool_offset_m)
        detection = None
        if args.auto_detect or calibration is not None:
            detection = detect_bead(
                rgb,
                hue_ranges=hue_ranges,
                min_saturation=args.min_saturation,
                min_value=args.min_value,
                min_area_px=args.min_area_px,
                min_circularity=args.min_circularity,
            )
        return Capture(side, rgb, t_link0_ee, point, detection, _stamp_ns(pose) or 0)

    def solve(current: list[Sample]) -> dict[str, Any]:
        k_matrix, distortion, size = camera()
        return calibrate(
            current,
            k_matrix,
            distortion,
            size,
            tool_offset_m=args.tool_offset_m,
            nominal_t_base_from_cam=nominal,
            ransac_threshold_px=args.ransac_px,
            per_arm=args.per_arm,
        )

    session = CalibrationSession(
        capture_fn=capture,
        live_rgb_fn=live_rgb,
        solve_fn=solve,
        samples=samples,
        samples_path=samples_path,
        output_dir=args.output_dir,
        output=args.output,
        camera_doc=None,
        tool_offset_m=args.tool_offset_m,
    )

    try:
        if calibration is not None:
            k_matrix, _distortion, _size = camera()
            k_cal = np.asarray(calibration["camera"]["K"], dtype=np.float64)
            if not np.allclose(k_cal, k_matrix, atol=1e-3):
                raise ValueError("live CameraInfo differs from the calibration's intrinsics")
            t_base_from_cam = np.asarray(calibration["T_newbase_from_zed_optical"], dtype=np.float64)
            frozen = capture(args.arm)
            if frozen.detection is None:
                raise RuntimeError("marker not detected; check lighting/HSV thresholds")
            predicted = project_points(frozen.point, k_matrix, invert_transform(t_base_from_cam))[0]
            hit = intersect_height(
                np.asarray(frozen.detection.pixel), k_matrix, t_base_from_cam, frozen.point[2]
            )
            report = {
                "arm": args.arm,
                "marker_base_m": frozen.point.tolist(),
                "detected_px": list(frozen.detection.pixel),
                "predicted_px": predicted.tolist(),
                "pixel_error_px": float(np.linalg.norm(predicted - frozen.detection.pixel)),
                "xy_error_at_true_height_m": float(np.linalg.norm(hit[:2] - frozen.point[:2])),
            }
            print(json.dumps(report, indent=1), flush=True)
            ok = report["pixel_error_px"] <= args.check_max_px
            verdict = "PASS" if ok else "FAIL (camera moved, base tilted, or marker offset changed)"
            print("CHECK", verdict, flush=True)
            return 0 if ok else 2

        if args.web:
            # Intrinsics are fetched lazily so the page is reachable before the ZED is up.
            try:
                camera(2.0)
            except TimeoutError as exc:
                print(f"warning: {exc}; will retry on capture", flush=True)
            server = serve_web(CalibrationWebApp(session), args.web_host, args.web)
            print(
                json.dumps(
                    {"web": f"http://{args.web_host}:{args.web}/", "tool_offset_m": args.tool_offset_m}
                ),
                flush=True,
            )
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
            return 0

        camera()
        print(json.dumps({"camera": session.camera_doc, "tool_offset_m": args.tool_offset_m}), flush=True)
        side = args.arm
        print(
            "Commands: Enter=capture (auto-detect only)  l/r=switch arm  d N=drop N  s=solve  q=solve+write+quit"
        )
        while True:
            print(f"[{side}] {len(session.samples)} samples > ", end="", flush=True)
            line = sys.stdin.readline()
            if not line:
                break
            command = line.strip().lower()
            try:
                if command in ("l", "r"):
                    side = "left" if command == "l" else "right"
                elif command.startswith("d "):
                    session.drop(int(command.split()[1]))
                elif command in ("s", "q"):
                    print(json.dumps(session.solve(write=command == "q"), indent=1), flush=True)
                    if command == "q":
                        break
                elif command:
                    print("unknown command", flush=True)
                else:
                    result = session.capture(side)
                    if result["auto_pixel"] is None:
                        session.discard()
                        print("marker not detected; use --web for manual clicks", flush=True)
                    else:
                        u, v = result["auto_pixel"]
                        session.accept(u, v, manual=False)
                        print(
                            f"sample {len(session.samples) - 1}: px {u:.1f},{v:.1f} base {result['point_base']}"
                        )
            except (ValueError, TimeoutError, RuntimeError) as exc:
                print(f"rejected: {exc}", flush=True)
        return 0
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        thread.join(timeout=2)
        node.destroy_node()


def solve_from_file(args) -> int:
    samples, document = load_samples(args.solve)
    geometry = load_geometry_manifest(args.manifest)
    camera = document["camera"]
    result = calibrate(
        samples,
        np.asarray(camera["K"], dtype=np.float64),
        np.asarray(camera["distortion"], dtype=np.float64),
        (int(camera["width"]), int(camera["height"])),
        tool_offset_m=float(document["tool_offset_m"]),
        nominal_t_base_from_cam=geometry.base_from_zed_optical.astype(np.float64),
        ransac_threshold_px=args.ransac_px,
        per_arm=args.per_arm,
    )
    print_report(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/tmr_rgb20d.yaml"))
    parser.add_argument(
        "--manifest", type=Path, required=True, help="dataset franka_duo_extras/derived_manifest.json"
    )
    parser.add_argument("--camera-info-topic", default="/head_camera/zed/rgb/color/rect/camera_info")
    parser.add_argument("--tool-offset-m", type=float, help="EE origin to marker centre along EE z")
    parser.add_argument("--arm", choices=SIDES, default="left", help="initial arm, or the arm for --check")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/zed_pnp"))
    parser.add_argument("--output", type=Path, default=Path("configs/zed_pnp_calibration.json"))
    parser.add_argument("--resume", action="store_true", help="continue from output-dir/samples.json")
    parser.add_argument("--solve", type=Path, help="re-solve from a samples.json without ROS")
    parser.add_argument("--check", type=Path, help="one-sample verification against a calibration")
    parser.add_argument("--check-max-px", type=float, default=5.0)
    parser.add_argument("--ransac-px", type=float, default=3.0)
    parser.add_argument("--per-arm", action="store_true", help="also solve one extrinsic per arm")
    parser.add_argument(
        "--hue-range",
        type=int,
        nargs=2,
        action="append",
        metavar=("LOW", "HIGH"),
        help="OpenCV hue range of the marker; repeatable. Default orange 5..28",
    )
    parser.add_argument("--min-saturation", type=int, default=100)
    parser.add_argument("--min-value", type=int, default=70)
    parser.add_argument("--min-area-px", type=float, default=20.0)
    parser.add_argument("--min-circularity", type=float, default=0.5)
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="colour-detect the marker (default: manual clicks in the web page only)",
    )
    parser.add_argument("--web", type=int, metavar="PORT", help="serve the browser front end on this port")
    parser.add_argument("--web-host", default="0.0.0.0")
    args = parser.parse_args(argv)
    if args.solve and args.check:
        parser.error("--solve and --check are exclusive")
    if args.solve:
        return solve_from_file(args)
    offset = args.tool_offset_m
    if offset is None or not math.isfinite(offset) or not 0 <= offset <= 0.5:
        parser.error("--tool-offset-m in [0, 0.5] is required for collection and --check")
    if args.web is not None and not 1 <= args.web <= 65535:
        parser.error("--web PORT must be 1..65535")
    if args.web is None and not args.check and not args.auto_detect:
        parser.error("collection needs --web PORT for manual clicks, or --auto-detect for the stdin loop")
    try:
        hue_ranges_from_args(args.hue_range)
    except ValueError as exc:
        parser.error(str(exc))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
