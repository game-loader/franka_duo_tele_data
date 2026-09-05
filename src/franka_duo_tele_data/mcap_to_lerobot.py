#!/usr/bin/env python3
"""Convert raw Franka Duo MCAP episodes to a LeRobot v3 dataset.

The recorder deliberately keeps ROS messages raw.  This module is the offline
boundary where those messages are decoded, synchronized and expressed in one
dataset frame.  The implementation follows the RGB/depth -> XYZ -> rigid
transform -> spatial sampling pipeline used by the RL-100/DP3 input path,
while keeping the coordinate math explicit and auditable in this file.

The converter intentionally does not use ROS.  ``rosbags`` supplies the ROS 2
CDR type system and ``AnyReader`` streams MCAP records without materialising a
bag in memory.  ``pyarrow`` writes the numeric part of the v3 dataset and
PyAV writes one MP4 stream per RGB camera.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
from collections import deque
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .action_spec import matrix_to_rot6d
from .pointcloud import adaptive_voxel_sample, depth_to_point_cloud, farthest_point_sample
from .ros_utils import depth_msg_to_meters, gripper_open_fraction, image_msg_to_rgb

# The raw TMR topic names are part of the capture contract.  Keeping them in
# one immutable mapping makes accidental source/relay substitutions visible.
DEFAULT_WORKSPACE_MIN = (0.4, -0.3, -0.3)
DEFAULT_WORKSPACE_MAX = (1.2, 0.3, 0.3)
WRIST_IMAGE_SIZE = (256, 256)
# The dual-arm Cartesian target has nine values per arm (XYZ + continuous
# rotation-6D).  The two gripper values are appended in the same order as the
# state fields and are taken from the *next* synchronized frame, exactly like
# the pose portion of the action.
EE_ACTION_DIM = 18
ACTION_DIM = 20
DEFAULT_TOPICS = {
    "head_rgb": "/head_camera/zed/rgb/color/rect/image",
    "head_depth": "/head_camera/zed/depth/depth_registered",
    "head_info": "/head_camera/zed/rgb/color/rect/camera_info",
    "wrist_left_rgb": "/wrist_camera_left/color/image_raw",
    "wrist_right_rgb": "/wrist_camera_right/color/image_raw",
    "left_pose": "/franka_duo_tele_data/rate100/left/franka_robot_state_broadcaster/current_pose",
    "right_pose": "/franka_duo_tele_data/rate100/right/franka_robot_state_broadcaster/current_pose",
    "left_joints": "/franka_duo_tele_data/rate100/left/franka_robot_state_broadcaster/measured_joint_states",
    "right_joints": "/franka_duo_tele_data/rate100/right/franka_robot_state_broadcaster/measured_joint_states",
    "left_gripper": "/franka_duo_tele_data/rate100/left/gripper/joint_states",
    "right_gripper": "/franka_duo_tele_data/rate100/right/gripper/joint_states",
    "episode_event": "/franka_duo_tele_data/episode_event",
}

EXPECTED_TYPES = {
    "head_rgb": "sensor_msgs/msg/Image",
    "head_depth": "sensor_msgs/msg/Image",
    "head_info": "sensor_msgs/msg/CameraInfo",
    "wrist_left_rgb": "sensor_msgs/msg/Image",
    "wrist_right_rgb": "sensor_msgs/msg/Image",
    "left_pose": "geometry_msgs/msg/PoseStamped",
    "right_pose": "geometry_msgs/msg/PoseStamped",
    "left_joints": "sensor_msgs/msg/JointState",
    "right_joints": "sensor_msgs/msg/JointState",
    "left_gripper": "sensor_msgs/msg/JointState",
    "right_gripper": "sensor_msgs/msg/JointState",
    "episode_event": "std_msgs/msg/String",
}


def _optional_dependency(name: str, import_name: str | None = None) -> Any:
    try:
        return __import__(import_name or name)
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError(
            f"{name} is required for MCAP post-processing; install the 'postprocess' extra"
        ) from exc


def _finite_vec(value: Sequence[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of shape ({size},), got {array.shape}")
    return array


def quaternion_to_matrix(quaternion: Sequence[float] | Any) -> np.ndarray:
    """Return a right-handed 3x3 matrix from an ``(x, y, z, w)`` quaternion."""

    if hasattr(quaternion, "x"):
        value = np.asarray([quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=np.float64)
    else:
        value = _finite_vec(quaternion, 4, "quaternion")
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion norm must be finite and non-zero")
    x, y, z, w = value / norm
    matrix = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return matrix.astype(np.float32)


def make_transform(
    translation: Sequence[float], rotation: Sequence[Sequence[float]] | np.ndarray
) -> np.ndarray:
    """Construct a homogeneous column-vector transform ``T_parent_child``."""

    position = _finite_vec(translation, 3, "translation")
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"rotation must be a finite 3x3 matrix, got {matrix.shape}")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=2e-4) or np.linalg.det(matrix) <= 0:
        raise ValueError("rotation must be a proper orthonormal matrix")
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = matrix.astype(np.float32)
    result[:3, 3] = position.astype(np.float32)
    return result


def compose_transform(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compose ``T_A_B @ T_B_C`` using the documented column-vector convention."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError("transforms must both have shape (4, 4)")
    result = a @ b
    if not np.isfinite(result).all() or not np.allclose(result[3], (0, 0, 0, 1), atol=1e-5):
        raise ValueError("composition produced an invalid homogeneous transform")
    return result.astype(np.float32)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid homogeneous transform."""

    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise ValueError("transform rotation must be orthonormal")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ value[:3, 3]
    return result.astype(np.float32)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to ``(N, 3)`` points."""

    value = np.asarray(points, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {value.shape}")
    matrix = np.asarray(transform, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    result = value.copy()
    result[:, :3] = value[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
    return np.ascontiguousarray(result)


def pose_to_transform(pose: Any) -> np.ndarray:
    """Decode a ROS ``geometry_msgs/Pose`` into ``T_frame_pose``."""

    return make_transform(
        (pose.position.x, pose.position.y, pose.position.z),
        quaternion_to_matrix(pose.orientation),
    )


def pose_vector(transform: np.ndarray) -> np.ndarray:
    """Flatten one pose as ``xyz + RL100 row-based 6D rotation`` (9 values)."""

    value = np.asarray(transform, dtype=np.float32)
    if value.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    return np.ascontiguousarray(np.concatenate((value[:3, 3], matrix_to_rot6d(value[:3, :3]))))


def resize_rgb(image: np.ndarray, size: tuple[int, int] = WRIST_IMAGE_SIZE) -> np.ndarray:
    """Resize an RGB HWC image without adding an image-processing dependency."""

    value = np.asarray(image, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {value.shape}")
    height, width = (int(size[0]), int(size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("resize target dimensions must be positive")
    if value.shape[:2] == (height, width):
        return np.ascontiguousarray(value)
    row_indices = np.rint(np.linspace(0, value.shape[0] - 1, height)).astype(np.int64)
    column_indices = np.rint(np.linspace(0, value.shape[1] - 1, width)).astype(np.int64)
    return np.ascontiguousarray(value[row_indices[:, None], column_indices[None, :], :])


def midpoint_base(
    left_world: np.ndarray, right_world: np.ndarray, rotation: np.ndarray | None = None
) -> np.ndarray:
    """Create a base at the midpoint of two arm-base origins.

    The default orientation is the USD root/world orientation.  The two arm
    bases have mirrored rotations, so averaging their quaternions would create
    an arbitrary frame; callers may provide a calibrated common orientation.
    """

    left = np.asarray(left_world, dtype=np.float64)
    right = np.asarray(right_world, dtype=np.float64)
    if left.shape != (4, 4) or right.shape != (4, 4):
        raise ValueError("arm transforms must have shape (4, 4)")
    orientation = np.eye(3, dtype=np.float32) if rotation is None else np.asarray(rotation, dtype=np.float32)
    return make_transform((left[:3, 3] + right[:3, 3]) / 2.0, orientation)


@dataclasses.dataclass(frozen=True)
class UsdGeometry:
    """Nominal fixed transforms extracted from the benchmark USD asset."""

    usd_path: str
    root_path: str
    left_arm_world: np.ndarray
    right_arm_world: np.ndarray
    zed_mount_world: np.ndarray
    new_base_world: np.ndarray
    base_from_left_arm: np.ndarray
    base_from_right_arm: np.ndarray
    base_from_zed_optical: np.ndarray
    left_prim: str
    right_prim: str
    mount_prim: str
    mount_to_optical: np.ndarray

    def manifest(self) -> dict[str, Any]:
        return {
            "usd_path": self.usd_path,
            "usd_root": self.root_path,
            "left_arm_prim": self.left_prim,
            "right_arm_prim": self.right_prim,
            "zed_mount_prim": self.mount_prim,
            "base_definition": "midpoint of left_fr3v2_link0 and right_fr3v2_link0 origins",
            "base_orientation": "USD root/world orientation (identity unless overridden)",
            "left_arm_world": self.left_arm_world.tolist(),
            "right_arm_world": self.right_arm_world.tolist(),
            "zed_mount_world": self.zed_mount_world.tolist(),
            "new_base_world": self.new_base_world.tolist(),
            "T_newbase_from_left_arm_base": self.base_from_left_arm.tolist(),
            "T_newbase_from_right_arm_base": self.base_from_right_arm.tolist(),
            "T_newbase_from_zed_optical": self.base_from_zed_optical.tolist(),
            "mount_to_optical": self.mount_to_optical.tolist(),
            "transform_convention": "p_A = T_A_from_B @ p_B; T_A_from_C = T_A_from_B @ T_B_from_C",
            "zed_external_calibration": "nominal USD head_camera_mounting_point plus ROS optical convention; override with --mount-to-optical",
        }


def _usd_to_numpy(gf_matrix: Any) -> np.ndarray:
    # USD Gf matrices use row-vector storage (translation in the last row).
    # Transpose to the column-vector convention used by this module.
    value = np.asarray(gf_matrix, dtype=np.float64).T
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("USD prim has an invalid transform")
    return value.astype(np.float32)


def _find_usd_prim(stage: Any, root: Any, names: Sequence[str]) -> Any:
    root_path = str(root.GetPath())
    # Search each alias in declared priority order.  This matters for the
    # camera mount: ``head_link`` is also present, but the more specific
    # ``head_camera_mounting_point`` is the intended optical attachment.
    for name in names:
        candidate = name.lower()
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(root_path) and prim.GetName().lower() == candidate:
                return prim
    raise ValueError(f"USD asset does not contain one of prims {list(names)!r}")


def load_usd_geometry(
    usd_path: Path,
    *,
    mount_to_optical: Sequence[float] | np.ndarray | None = None,
    base_rotation: Sequence[float] | np.ndarray | None = None,
) -> UsdGeometry:
    """Read Franka bases and the head-camera mounting point from benchmark USD."""

    _optional_dependency("usd-core", "pxr")
    from pxr import Usd, UsdGeom  # type: ignore[import-not-found]

    path = Path(usd_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"USD asset does not exist: {path}")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Unable to open USD asset: {path}")
    root = next(
        (p for p in stage.GetPseudoRoot().GetChildren() if p.GetName().startswith("mobile_fr3_duo")), None
    )
    if root is None:
        root = stage.GetPseudoRoot()
    left = _find_usd_prim(stage, root, ("left_fr3v2_link0", "left_base"))
    right = _find_usd_prim(stage, root, ("right_fr3v2_link0", "right_base"))
    mount = _find_usd_prim(stage, root, ("head_camera_mounting_point", "head_link"))
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_world = _usd_to_numpy(cache.GetLocalToWorldTransform(root))
    left_world = _usd_to_numpy(cache.GetLocalToWorldTransform(left))
    right_world = _usd_to_numpy(cache.GetLocalToWorldTransform(right))
    mount_world = _usd_to_numpy(cache.GetLocalToWorldTransform(mount))
    if base_rotation is None:
        base_rot = root_world[:3, :3]
    else:
        raw_rotation = np.asarray(base_rotation, dtype=np.float32)
        if raw_rotation.size != 9:
            raise ValueError("base_rotation must contain nine row-major values")
        base_rot = raw_rotation.reshape(3, 3)
    base_world = midpoint_base(left_world, right_world, base_rot)

    if mount_to_optical is None:
        # ROS camera_link -> camera optical frame.  The USD mount is treated as
        # the ZED body frame; a measured mount-to-camera transform can replace it.
        mount_optical = make_transform(
            (0, 0, 0),
            quaternion_to_matrix((0.5, -0.5, 0.5, -0.5)),
        )
    else:
        values = np.asarray(mount_to_optical, dtype=np.float32)
        if values.size != 16:
            raise ValueError("mount_to_optical must contain sixteen row-major values")
        mount_optical = values.reshape(4, 4)
        if not np.allclose(mount_optical[3], (0, 0, 0, 1), atol=1e-5):
            raise ValueError("mount_to_optical must be homogeneous")
        # Validate the rotation using the same rigid-transform checks.
        mount_optical = make_transform(mount_optical[:3, 3], mount_optical[:3, :3])

    base_from_world = invert_transform(base_world)
    return UsdGeometry(
        usd_path=str(path),
        root_path=str(root.GetPath()),
        left_arm_world=left_world,
        right_arm_world=right_world,
        zed_mount_world=mount_world,
        new_base_world=base_world,
        base_from_left_arm=compose_transform(base_from_world, left_world),
        base_from_right_arm=compose_transform(base_from_world, right_world),
        base_from_zed_optical=compose_transform(
            base_from_world, compose_transform(mount_world, mount_optical)
        ),
        left_prim=str(left.GetPath()),
        right_prim=str(right.GetPath()),
        mount_prim=str(mount.GetPath()),
        mount_to_optical=mount_optical,
    )


@dataclasses.dataclass(frozen=True)
class TimedMessage:
    stamp_ns: int
    receipt_ns: int
    message: Any


def message_stamp_ns(message: Any, receipt_ns: int) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(receipt_ns)
    try:
        value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return int(receipt_ns)
    return value if value > 0 else int(receipt_ns)


class TimedBuffer:
    def __init__(self, maxlen: int):
        if maxlen <= 0:
            raise ValueError("buffer maxlen must be positive")
        self.values: deque[TimedMessage] = deque(maxlen=maxlen)

    def append(self, message: Any, receipt_ns: int) -> None:
        self.values.append(TimedMessage(message_stamp_ns(message, receipt_ns), int(receipt_ns), message))

    def nearest(self, target_ns: int, tolerance_ns: int) -> TimedMessage | None:
        if not self.values:
            return None
        value = min(self.values, key=lambda item: (abs(item.stamp_ns - target_ns), -item.receipt_ns))
        return value if abs(value.stamp_ns - target_ns) <= tolerance_ns else None


class FixedRateGate:
    """Keep the first valid source frame on or after each output time grid."""

    def __init__(self, fps: int):
        if fps <= 0:
            raise ValueError("output fps must be positive")
        self.period_ns = int(round(1_000_000_000 / fps))
        self.next_target_ns: int | None = None

    def keep(self, stamp_ns: int) -> bool:
        stamp = int(stamp_ns)
        if self.next_target_ns is None:
            self.next_target_ns = stamp + self.period_ns
            return True
        if stamp < self.next_target_ns:
            return False
        while self.next_target_ns <= stamp:
            self.next_target_ns += self.period_ns
        return True


@dataclasses.dataclass
class SyncStats:
    head_seen: int = 0
    frames_ready: int = 0
    frames_written: int = 0
    dropped_missing_depth: int = 0
    dropped_missing_wrist: int = 0
    dropped_missing_state: int = 0
    dropped_missing_pose: int = 0
    dropped_missing_camera_info: int = 0
    dropped_invalid_payload: int = 0
    dropped_resampled: int = 0
    dropped_no_next_action: int = 0
    pending_overflow: int = 0

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class DerivedFrame:
    source_stamp_ns: int
    source_skew_ns: np.ndarray
    head_rgb: np.ndarray
    wrist_left_rgb: np.ndarray
    wrist_right_rgb: np.ndarray
    point_cloud: np.ndarray
    state: np.ndarray
    ee_pose: np.ndarray
    left_pose_frame_id: str | None = None
    right_pose_frame_id: str | None = None
    # Normalized open fractions for the two grippers.  This is kept separately
    # from ``state`` so action construction cannot accidentally use stale
    # state values when the representation evolves.
    gripper: np.ndarray | None = None


def _joint_positions(message: Any, side: str, expected: int = 7) -> np.ndarray:
    names = [str(item) for item in getattr(message, "name", ())]
    positions = list(getattr(message, "position", ()))
    if len(positions) < expected:
        raise ValueError(f"JointState contains {len(positions)} positions; expected at least {expected}")
    by_name = {name: float(value) for name, value in zip(names, positions, strict=False)}
    ordered = []
    for index in range(1, expected + 1):
        candidates = (
            f"{side}_fr3v2_joint{index}",
            f"{side}_fr3v2_1_joint{index}",
            f"{side}_fr3_joint{index}",
            f"{side}_joint{index}",
            f"{side}_panda_joint{index}",
        )
        match = next((by_name[name] for name in candidates if name in by_name), None)
        if match is None:
            # The TMR relay preserves source order.  Accept a nameless or
            # legacy seven-element message as a documented fallback.
            if len(names) != expected:
                raise ValueError(f"JointState is missing {side} joint {index}")
            match = float(positions[index - 1])
        ordered.append(match)
    values = np.asarray(ordered, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("JointState contains non-finite positions")
    return np.ascontiguousarray(values)


def _gripper_position(message: Any) -> float:
    values = np.asarray(list(getattr(message, "position", ())), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values[0]):
        raise ValueError("gripper JointState contains no finite position")
    return float(values[0])


def _camera_matrix(info: Any) -> np.ndarray:
    values = np.asarray(getattr(info, "k", ()), dtype=np.float32)
    if values.size != 9 or not np.isfinite(values).all():
        raise ValueError("ZED CameraInfo.k must contain nine finite values")
    return values.reshape(3, 3)


def _episode_dirs(input_root: Path) -> list[Path]:
    path = input_root.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"input root does not exist: {path}")
    if path.is_file() and path.suffix == ".mcap":
        return [path.parent]
    if any(path.glob("*.mcap")):
        return [path]
    episodes = sorted(item for item in path.iterdir() if item.is_dir() and any(item.glob("*.mcap")))
    if not episodes:
        raise FileNotFoundError(f"no episode directories containing MCAP found below {path}")
    return episodes


def _load_episode_manifest(episode_dir: Path) -> dict[str, Any] | None:
    path = episode_dir / "episode_manifest.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid episode manifest {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"episode manifest {path} must contain a JSON object")
    status = value.get("status")
    if status != "complete":
        raise ValueError(f"episode manifest {path} is not complete (status={status!r})")
    return dict(value)


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        av = _optional_dependency("av")
        self.container = av.open(str(path), mode="w")
        try:
            self.stream = self.container.add_stream("libx264", rate=Fraction(fps, 1))
            self.stream.options = {"crf": "18", "preset": "medium"}
        except Exception:
            self.stream = self.container.add_stream("mpeg4", rate=Fraction(fps, 1))
        self.stream.width = int(width)
        self.stream.height = int(height)
        self.stream.pix_fmt = "yuv420p"
        self.frames = 0

    def write(self, rgb: np.ndarray) -> None:
        value = np.asarray(rgb, dtype=np.uint8)
        if value.shape != (self.stream.height, self.stream.width, 3):
            raise ValueError(
                f"RGB frame shape {value.shape} does not match video {(self.stream.height, self.stream.width, 3)}"
            )
        av = _optional_dependency("av")
        frame = av.VideoFrame.from_ndarray(value, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


class _StatsAccumulator:
    def __init__(self, shape: tuple[int, ...], dtype: str):
        self.shape = shape
        self.dtype = dtype
        size = int(np.prod(shape))
        self.minimum = np.full(size, np.inf, dtype=np.float64)
        self.maximum = np.full(size, -np.inf, dtype=np.float64)
        self.total = np.zeros(size, dtype=np.float64)
        self.square = np.zeros(size, dtype=np.float64)
        self.count = 0

    def update(self, value: Any) -> None:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size != self.minimum.size or not np.isfinite(array).all():
            return
        self.minimum = np.minimum(self.minimum, array)
        self.maximum = np.maximum(self.maximum, array)
        self.total += array
        self.square += array * array
        self.count += 1

    def finish(self) -> dict[str, Any]:
        if self.count == 0:
            zeros = np.zeros(self.shape, dtype=np.float32).tolist()
            return {"min": zeros, "max": zeros, "mean": zeros, "std": zeros, "count": [0]}
        mean = self.total / self.count
        std = np.sqrt(np.maximum(self.square / self.count - mean * mean, 0))
        return {
            "min": self.minimum.reshape(self.shape).astype(np.float32).tolist(),
            "max": self.maximum.reshape(self.shape).astype(np.float32).tolist(),
            "mean": mean.reshape(self.shape).astype(np.float32).tolist(),
            "std": std.reshape(self.shape).astype(np.float32).tolist(),
            "count": [int(self.count)],
        }


def _arrow_array(values: list[Any], shape: tuple[int, ...], dtype: str, pa: Any) -> Any:
    if dtype == "float32":
        arrow_scalar = pa.float32()
    elif dtype == "int64":
        arrow_scalar = pa.int64()
    else:
        raise ValueError(f"unsupported parquet dtype {dtype}")
    if shape == (1,):
        return pa.array([np.asarray(item).reshape(-1)[0].item() for item in values], type=arrow_scalar)
    item_type = arrow_scalar
    for size in reversed(shape):
        item_type = pa.list_(item_type, list_size=size)
    normalized = [np.asarray(item).reshape(shape).tolist() for item in values]
    return pa.array(normalized, type=item_type)


class LeRobotV3Writer:
    """Small dependency-light writer for the public LeRobot v3 layout."""

    def __init__(self, root: Path, fps: int, task: str, num_points: int, channels: int):
        self.root = root
        self.fps = fps
        self.task = task
        self.num_points = num_points
        self.channels = channels
        self.data_writer = None
        self.data_path: Path | None = None
        self.video_writers: dict[str, VideoWriter] = {}
        self.video_shapes: dict[str, tuple[int, int, int]] = {}
        self.features: dict[str, dict[str, Any]] | None = None
        self.data_rows: list[dict[str, Any]] = []
        self.episode_metadata: list[dict[str, Any]] = []
        self.total_frames = 0
        self.total_video_frames = 0
        self.accumulators: dict[str, _StatsAccumulator] = {}
        self.video_accumulators: dict[str, _StatsAccumulator] = {}
        self.root.mkdir(parents=True, exist_ok=True)

    def _initialize(self, frame: DerivedFrame) -> None:
        if self.features is not None:
            return
        image_shapes = {
            "observation.images.head": tuple(frame.head_rgb.shape),
            "observation.images.wrist_left": tuple(frame.wrist_left_rgb.shape),
            "observation.images.wrist_right": tuple(frame.wrist_right_rgb.shape),
        }
        if any(len(shape) != 3 or shape[2] != 3 for shape in image_shapes.values()):
            raise ValueError(f"RGB frames must be HWC with three channels, got {image_shapes}")
        vector_names = [
            "left_x",
            "left_y",
            "left_z",
            "left_rot6d_row0_x",
            "left_rot6d_row0_y",
            "left_rot6d_row0_z",
            "left_rot6d_row1_x",
            "left_rot6d_row1_y",
            "left_rot6d_row1_z",
            "right_x",
            "right_y",
            "right_z",
            "right_rot6d_row0_x",
            "right_rot6d_row0_y",
            "right_rot6d_row0_z",
            "right_rot6d_row1_x",
            "right_rot6d_row1_y",
            "right_rot6d_row1_z",
        ]
        state_names = [
            *(f"left_joint_{i}" for i in range(1, 8)),
            *(f"right_joint_{i}" for i in range(1, 8)),
            "left_gripper_open_fraction",
            "right_gripper_open_fraction",
        ]
        self.features = {
            "observation.state": {"dtype": "float32", "shape": [16], "names": state_names},
            "observation.ee_pose": {"dtype": "float32", "shape": [18], "names": vector_names},
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": [*vector_names, "left_gripper_open_fraction", "right_gripper_open_fraction"],
            },
            "observation.point_cloud": {
                "dtype": "float32",
                "shape": [self.num_points, self.channels],
                "names": ["x", "y", "z", "r", "g", "b"][: self.channels],
            },
            "observation.source_timestamp_ns": {"dtype": "int64", "shape": [1], "names": None},
            "observation.sync_skew_ns": {
                "dtype": "int64",
                "shape": [9],
                "names": [
                    "depth",
                    "wrist_left",
                    "wrist_right",
                    "left_pose",
                    "right_pose",
                    "left_joints",
                    "right_joints",
                    "left_gripper",
                    "right_gripper",
                ],
            },
        }
        for key, shape in image_shapes.items():
            self.features[key] = {
                "dtype": "video",
                "shape": list(shape),
                "names": ["height", "width", "channels"],
                "info": {
                    "video.is_depth_map": False,
                    "video.height": shape[0],
                    "video.width": shape[1],
                    "video.channels": shape[2],
                    "video.fps": self.fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                },
            }
        self.accumulators = {
            key: _StatsAccumulator(tuple(value["shape"]), value["dtype"])
            for key, value in self.features.items()
            if value["dtype"] != "video"
        }
        self.video_accumulators = {key: _StatsAccumulator((3, 1, 1), "float32") for key in image_shapes}
        import pyarrow as pa
        import pyarrow.parquet as pq

        fields = [
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
        for key in (
            "observation.state",
            "observation.ee_pose",
            "action",
            "observation.point_cloud",
            "observation.source_timestamp_ns",
            "observation.sync_skew_ns",
        ):
            spec = self.features[key]
            if tuple(spec["shape"]) == (1,):
                arrow_type = pa.float32() if spec["dtype"] == "float32" else pa.int64()
            else:
                item_type = pa.float32() if spec["dtype"] == "float32" else pa.int64()
                for size in reversed(spec["shape"]):
                    item_type = pa.list_(item_type, list_size=size)
                arrow_type = item_type
            fields.append(pa.field(key, arrow_type))
        self.data_path = self.root / "data" / "chunk-000" / "file-000.parquet"
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_writer = pq.ParquetWriter(self.data_path, pa.schema(fields), compression="zstd")

    def _ensure_video(self, key: str, rgb: np.ndarray) -> VideoWriter:
        shape = tuple(np.asarray(rgb).shape)
        writer = self.video_writers.get(key)
        if writer is None:
            self.video_shapes[key] = shape  # type: ignore[assignment]
            path = self.root / "videos" / key / "chunk-000" / "file-000.mp4"
            writer = VideoWriter(path, shape[1], shape[0], self.fps)
            self.video_writers[key] = writer
        return writer

    def add_frame(
        self, frame: DerivedFrame, action: np.ndarray, episode_index: int, frame_index: int
    ) -> None:
        self._initialize(frame)
        assert self.features is not None and self.data_writer is not None
        if frame.ee_pose.shape != (EE_ACTION_DIM,) or np.asarray(action).shape != (ACTION_DIM,):
            raise ValueError(
                "ee_pose must have shape (18,) and action must have shape (20,) "
                "(dual-arm xyz + rot6d + two grippers)"
            )
        if frame.point_cloud.shape != (self.num_points, self.channels):
            raise ValueError(
                f"point_cloud must have shape {(self.num_points, self.channels)}, got {frame.point_cloud.shape}"
            )
        images = {
            "observation.images.head": frame.head_rgb,
            "observation.images.wrist_left": frame.wrist_left_rgb,
            "observation.images.wrist_right": frame.wrist_right_rgb,
        }
        for key, image in images.items():
            self._ensure_video(key, image).write(image)
            pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[:, :, :, None]
            self.video_accumulators[key].update(pixels.mean(axis=(1, 2, 3)).reshape(3, 1, 1) / 255.0)
        row = {
            "timestamp": np.float32(frame_index / self.fps),
            "frame_index": int(frame_index),
            "episode_index": int(episode_index),
            "index": int(self.total_frames),
            "task_index": 0,
            "observation.state": frame.state,
            "observation.ee_pose": frame.ee_pose,
            "action": action,
            "observation.point_cloud": frame.point_cloud,
            "observation.source_timestamp_ns": np.int64(frame.source_stamp_ns),
            "observation.sync_skew_ns": frame.source_skew_ns,
        }
        self.data_rows.append(row)
        for key, value in row.items():
            if key in self.accumulators:
                self.accumulators[key].update(value)
        self.total_frames += 1
        self.total_video_frames += 1
        if len(self.data_rows) >= 64:
            self._flush_rows()

    def _flush_rows(self) -> None:
        if not self.data_rows:
            return
        import pyarrow as pa

        assert self.features is not None and self.data_writer is not None
        columns = []
        names = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
        columns.extend(
            [
                pa.array(
                    [row[name] for row in self.data_rows],
                    type=pa.float32() if name == "timestamp" else pa.int64(),
                )
                for name in names
            ]
        )
        for key in (
            "observation.state",
            "observation.ee_pose",
            "action",
            "observation.point_cloud",
            "observation.source_timestamp_ns",
            "observation.sync_skew_ns",
        ):
            spec = self.features[key]
            columns.append(
                _arrow_array([row[key] for row in self.data_rows], tuple(spec["shape"]), spec["dtype"], pa)
            )
        self.data_writer.write_table(pa.Table.from_arrays(columns, schema=self.data_writer.schema))
        self.data_rows.clear()

    def finish_episode(self, episode_index: int, original_episode: str, stats: SyncStats) -> None:
        self._flush_rows()
        length = int(stats.frames_written)
        if length == 0:
            return
        start = sum(int(item["length"]) for item in self.episode_metadata)
        end = start + length
        end_time = end / self.fps
        entry = {
            "episode_index": episode_index,
            "tasks": [self.task],
            "length": length,
            # LeRobot uses these metadata indices to locate this row in the
            # consolidated episode parquet file.  They are independent from
            # the data/video chunk indices below.
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": start,
            "dataset_to_index": end,
            "original_episode": original_episode,
            "sync_stats": stats.as_dict(),
        }
        for key in self.video_writers:
            entry[f"videos/{key}/chunk_index"] = 0
            entry[f"videos/{key}/file_index"] = 0
            entry[f"videos/{key}/from_timestamp"] = start / self.fps
            entry[f"videos/{key}/to_timestamp"] = end_time
        self.episode_metadata.append(entry)

    def finalize(self) -> None:
        self._flush_rows()
        if self.data_writer is not None:
            self.data_writer.close()
        if self.features is None:
            raise ValueError("no synchronized frames were written")
        for key, writer in self.video_writers.items():
            # Persist the stream's actual values, including the fallback codec
            # selected when libx264 is unavailable on the host.
            stream = writer.stream
            info = self.features[key]["info"]
            rate = stream.base_rate
            info.update(
                {
                    "video.height": int(stream.height),
                    "video.width": int(stream.width),
                    "video.channels": 3,
                    "video.fps": int(rate) if rate is not None else self.fps,
                    "video.codec": str(stream.codec.canonical_name),
                    "video.pix_fmt": str(stream.pix_fmt),
                }
            )
        for writer in self.video_writers.values():
            writer.close()
        import pyarrow as pa
        import pyarrow.parquet as pq

        meta = self.root / "meta"
        (meta / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        info = {
            "codebase_version": "v3.0",
            "fps": self.fps,
            "robot_type": "franka_duo",
            "total_episodes": len(self.episode_metadata),
            "total_frames": self.total_frames,
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 500,
            "video_files_size_in_mb": 2000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": self.features,
            "splits": {"train": f"0:{len(self.episode_metadata)}"},
        }
        (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        import pandas as pd

        tasks = pd.DataFrame(
            {"task_index": [0]},
            index=pd.Index([self.task], name="task"),
        )
        tasks.to_parquet(meta / "tasks.parquet", compression="zstd")
        if self.episode_metadata:
            # Nested sync_stats are useful provenance but are not part of the
            # canonical reader contract; keep them as a JSON string column.
            rows = []
            for item in self.episode_metadata:
                row = dict(item)
                row["sync_stats"] = json.dumps(row["sync_stats"], sort_keys=True)
                rows.append(row)
            pq.write_table(
                pa.Table.from_pylist(rows),
                meta / "episodes" / "chunk-000" / "file-000.parquet",
                compression="zstd",
            )
        stats = {key: accumulator.finish() for key, accumulator in self.accumulators.items()}
        stats.update({key: accumulator.finish() for key, accumulator in self.video_accumulators.items()})
        (meta / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


class EpisodeConverter:
    def __init__(
        self,
        geometry: UsdGeometry,
        writer: LeRobotV3Writer,
        *,
        topics: Mapping[str, str] = DEFAULT_TOPICS,
        num_points: int = 2048,
        channels: int = 3,
        sampling: str = "adaptive",
        seed: int = 0,
        fps_candidate_limit: int = 4096,
        rgb_tolerance_ms: float = 45.0,
        depth_tolerance_ms: float = 16.0,
        state_tolerance_ms: float = 50.0,
        min_depth: float = 0.05,
        max_depth: float = 5.0,
        workspace_min: Sequence[float] = DEFAULT_WORKSPACE_MIN,
        workspace_max: Sequence[float] = DEFAULT_WORKSPACE_MAX,
        gripper_closed: float = 0.8,
        gripper_open: float = 0.0,
    ):
        if channels != 3:
            raise ValueError("channels must be 3 (XYZ-only point cloud)")
        if sampling not in ("adaptive", "random", "fps"):
            raise ValueError("sampling must be adaptive, random or fps")
        if num_points <= 0 or fps_candidate_limit < num_points:
            raise ValueError("num_points must be positive and fps_candidate_limit >= num_points")
        self.geometry = geometry
        self.writer = writer
        self.topics = dict(topics)
        self.num_points = num_points
        self.channels = channels
        self.sampling = sampling
        self.seed = seed
        self.fps_candidate_limit = fps_candidate_limit
        self.tolerances = {
            "rgb": int(rgb_tolerance_ms * 1_000_000),
            "depth": int(depth_tolerance_ms * 1_000_000),
            "state": int(state_tolerance_ms * 1_000_000),
        }
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.workspace_min = _finite_vec(workspace_min, 3, "workspace_min").astype(np.float32)
        self.workspace_max = _finite_vec(workspace_max, 3, "workspace_max").astype(np.float32)
        if not np.all(self.workspace_min < self.workspace_max):
            raise ValueError("workspace_min must be strictly less than workspace_max")
        self.gripper_closed = gripper_closed
        self.gripper_open = gripper_open

    def _connections(self, reader: Any) -> list[Any]:
        by_topic: dict[str, list[Any]] = {}
        for connection in reader.connections:
            by_topic.setdefault(connection.topic, []).append(connection)
        selected = []
        missing = []
        for key, topic in self.topics.items():
            if key == "episode_event" and topic not in by_topic:
                continue
            candidates = by_topic.get(topic, [])
            if not candidates:
                missing.append(topic)
                continue
            expected = EXPECTED_TYPES[key]
            message_types = {connection.msgtype for connection in candidates}
            if message_types != {expected} or len(candidates) != 1:
                details = sorted(message_types) or ["<none>"]
                raise ValueError(
                    f"{topic} has ambiguous connections ({len(candidates)} channels, types={details}); "
                    f"expected exactly one {expected} channel"
                )
            connection = candidates[0]
            selected.append(connection)
        if missing:
            raise ValueError(f"MCAP is missing required topics: {missing}")
        return selected

    def _build_frame(
        self,
        reader: Any,
        target: TimedMessage,
        buffers: Mapping[str, TimedBuffer],
        camera_info: Any,
        frame_index: int,
        stats: SyncStats,
    ) -> DerivedFrame | None:
        depth = buffers["head_depth"].nearest(target.stamp_ns, self.tolerances["depth"])
        wrist_left = buffers["wrist_left_rgb"].nearest(target.stamp_ns, self.tolerances["rgb"])
        wrist_right = buffers["wrist_right_rgb"].nearest(target.stamp_ns, self.tolerances["rgb"])
        left_pose = buffers["left_pose"].nearest(target.stamp_ns, self.tolerances["state"])
        right_pose = buffers["right_pose"].nearest(target.stamp_ns, self.tolerances["state"])
        left_joints = buffers["left_joints"].nearest(target.stamp_ns, self.tolerances["state"])
        right_joints = buffers["right_joints"].nearest(target.stamp_ns, self.tolerances["state"])
        left_gripper = buffers["left_gripper"].nearest(target.stamp_ns, self.tolerances["state"])
        right_gripper = buffers["right_gripper"].nearest(target.stamp_ns, self.tolerances["state"])
        if depth is None:
            stats.dropped_missing_depth += 1
            return None
        if wrist_left is None or wrist_right is None:
            stats.dropped_missing_wrist += 1
            return None
        if left_pose is None or right_pose is None:
            stats.dropped_missing_pose += 1
            return None
        if any(item is None for item in (left_joints, right_joints, left_gripper, right_gripper)):
            stats.dropped_missing_state += 1
            return None
        if camera_info is None:
            stats.dropped_missing_camera_info += 1
            return None
        try:
            head_rgb = image_msg_to_rgb(target.message)
            left_rgb = resize_rgb(image_msg_to_rgb(wrist_left.message))
            right_rgb = resize_rgb(image_msg_to_rgb(wrist_right.message))
            # Keep the ZED 32FC1 meters payload at float32 precision for
            # deprojection; live evaluation intentionally defaults to f16.
            depth_m = depth_msg_to_meters(depth.message, 0.001, dtype=np.float32)
            matrix = _camera_matrix(camera_info)
            if depth_m.shape != head_rgb.shape[:2]:
                raise ValueError(f"head RGB/depth shapes differ: {head_rgb.shape} and {depth_m.shape}")
            info_width = int(getattr(camera_info, "width", depth_m.shape[1]))
            info_height = int(getattr(camera_info, "height", depth_m.shape[0]))
            if (info_height, info_width) != depth_m.shape:
                raise ValueError(
                    f"ZED CameraInfo dimensions {(info_height, info_width)} do not match depth {depth_m.shape}"
                )
            all_points = depth_to_point_cloud(
                depth_m,
                matrix,
                extrinsics=self.geometry.base_from_zed_optical,
                workspace_min=self.workspace_min,
                workspace_max=self.workspace_max,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                num_points=None,
            )
            if self.sampling == "adaptive":
                points = adaptive_voxel_sample(
                    all_points,
                    self.num_points,
                    seed=self.seed + frame_index,
                )
            elif self.sampling == "fps":
                points = farthest_point_sample(
                    all_points,
                    self.num_points,
                    seed=self.seed + frame_index,
                    candidate_limit=self.fps_candidate_limit,
                )
            else:
                rng = np.random.default_rng(self.seed + frame_index)
                if all_points.shape[0] >= self.num_points:
                    points = all_points[rng.choice(all_points.shape[0], self.num_points, replace=False)]
                else:
                    pad = rng.choice(all_points.shape[0], self.num_points - all_points.shape[0], replace=True)
                    points = np.concatenate((all_points, all_points[pad]), axis=0)
            left_pose_transform = compose_transform(
                self.geometry.base_from_left_arm, pose_to_transform(left_pose.message.pose)
            )
            right_pose_transform = compose_transform(
                self.geometry.base_from_right_arm, pose_to_transform(right_pose.message.pose)
            )
            state = np.concatenate(
                (
                    _joint_positions(left_joints.message, "left"),
                    _joint_positions(right_joints.message, "right"),
                    np.asarray(
                        [
                            gripper_open_fraction(
                                self._gripper_position(left_gripper.message),
                                closed_position=self.gripper_closed,
                                open_position=self.gripper_open,
                            ),
                            gripper_open_fraction(
                                self._gripper_position(right_gripper.message),
                                closed_position=self.gripper_closed,
                                open_position=self.gripper_open,
                            ),
                        ],
                        dtype=np.float32,
                    ),
                )
            ).astype(np.float32)
        except (TypeError, ValueError, OverflowError):
            stats.dropped_invalid_payload += 1
            return None
        source_values = {
            "depth": depth,
            "wrist_left": wrist_left,
            "wrist_right": wrist_right,
            "left_pose": left_pose,
            "right_pose": right_pose,
            "left_joints": left_joints,
            "right_joints": right_joints,
            "left_gripper": left_gripper,
            "right_gripper": right_gripper,
        }
        skew = np.asarray(
            [item.stamp_ns - target.stamp_ns for item in source_values.values()], dtype=np.int64
        )
        stats.frames_ready += 1
        return DerivedFrame(
            source_stamp_ns=target.stamp_ns,
            source_skew_ns=skew,
            head_rgb=head_rgb,
            wrist_left_rgb=left_rgb,
            wrist_right_rgb=right_rgb,
            point_cloud=np.ascontiguousarray(points, dtype=np.float32),
            state=state,
            ee_pose=np.concatenate(
                (pose_vector(left_pose_transform), pose_vector(right_pose_transform))
            ).astype(np.float32),
            gripper=state[-2:].copy(),
            left_pose_frame_id=str(getattr(getattr(left_pose.message, "header", None), "frame_id", "")),
            right_pose_frame_id=str(getattr(getattr(right_pose.message, "header", None), "frame_id", "")),
        )

    def _gripper_position(self, message: Any) -> float:
        value = _gripper_position(message)
        # Keep calibration local to the conversion and make the endpoint choice
        # explicit in the manifest generated by ``convert``.
        if not math.isfinite(self.gripper_closed) or not math.isfinite(self.gripper_open):
            raise ValueError("gripper calibration endpoints must be finite")
        return value

    def convert_episode(
        self, episode_dir: Path, output_episode_index: int
    ) -> tuple[SyncStats, dict[str, Any]]:
        _optional_dependency("rosbags.highlevel", "rosbags.highlevel")
        from rosbags.highlevel import AnyReader  # type: ignore[import-not-found]

        sidecar = _load_episode_manifest(episode_dir)
        stats = SyncStats()
        buffers = {
            # Image payloads are large; a few hundred milliseconds is enough
            # to cover the configured timestamp tolerances without retaining a
            # substantial fraction of a multi-GB bag.
            "head_depth": TimedBuffer(16),
            "wrist_left_rgb": TimedBuffer(16),
            "wrist_right_rgb": TimedBuffer(16),
            "left_pose": TimedBuffer(256),
            "right_pose": TimedBuffer(256),
            "left_joints": TimedBuffer(256),
            "right_joints": TimedBuffer(256),
            "left_gripper": TimedBuffer(256),
            "right_gripper": TimedBuffer(256),
        }
        pending: deque[TimedMessage] = deque(maxlen=8)
        camera_info = None
        previous: DerivedFrame | None = None
        event_records: list[dict[str, Any]] = []
        pose_frame_ids = {"left": set[str](), "right": set[str]()}
        output_gate = FixedRateGate(self.writer.fps)
        frame_index = 0
        with AnyReader([episode_dir]) as reader:
            connections = self._connections(reader)
            event_topic = self.topics["episode_event"]
            event_topic_present = any(connection.topic == event_topic for connection in connections)
            topic_to_key = {topic: key for key, topic in self.topics.items()}
            flush_ns = max(self.tolerances.values()) + 10_000_000

            def flush_ready(watermark_ns: int, force: bool = False) -> None:
                nonlocal previous, frame_index
                while pending and (force or watermark_ns >= pending[0].receipt_ns + flush_ns):
                    target = pending.popleft()
                    current = self._build_frame(reader, target, buffers, camera_info, frame_index, stats)
                    if current is None:
                        frame_index += 1
                        continue
                    if not output_gate.keep(current.source_stamp_ns):
                        stats.dropped_resampled += 1
                        frame_index += 1
                        continue
                    if current.left_pose_frame_id:
                        pose_frame_ids["left"].add(current.left_pose_frame_id)
                    if current.right_pose_frame_id:
                        pose_frame_ids["right"].add(current.right_pose_frame_id)
                    if previous is not None:
                        # The action for frame ``previous`` is the next valid
                        # synchronized frame's target.  Extend the historical
                        # 18D pose target with the next frame's two normalized
                        # gripper open fractions.
                        if current.gripper is None or np.asarray(current.gripper).shape != (2,):
                            raise ValueError("next synchronized frame is missing its two gripper values")
                        next_action = np.concatenate((current.ee_pose, current.gripper)).astype(np.float32)
                        self.writer.add_frame(
                            previous, next_action, output_episode_index, stats.frames_written
                        )
                        stats.frames_written += 1
                    frame_index += 1
                    previous = current

            for connection, receipt_ns, raw in reader.messages(connections=connections):
                key = topic_to_key[connection.topic]
                message = reader.deserialize(raw, connection.msgtype)
                if key == "head_info":
                    camera_info = message
                elif key == "episode_event":
                    payload = getattr(message, "data", "")
                    if isinstance(payload, (str, bytes, bytearray)):
                        try:
                            value = json.loads(payload)
                        except json.JSONDecodeError:
                            value = None
                        if isinstance(value, Mapping):
                            event_records.append(dict(value))
                elif key == "head_rgb":
                    stats.head_seen += 1
                    if len(pending) == pending.maxlen:
                        stats.pending_overflow += 1
                    pending.append(
                        TimedMessage(message_stamp_ns(message, receipt_ns), int(receipt_ns), message)
                    )
                elif key in buffers:
                    buffers[key].append(message, receipt_ns)
                flush_ready(int(receipt_ns))
            flush_ready(0, force=True)
        if previous is not None:
            stats.dropped_no_next_action += 1
        if event_topic_present and not any(item.get("event") == "start" for item in event_records):
            raise ValueError(f"MCAP episode {episode_dir} has episode_event topic but no start event")
        self.writer.finish_episode(output_episode_index, episode_dir.name, stats)
        provenance = {
            "event_topic_present": event_topic_present,
            "event_count": len(event_records),
            "events": [item.get("event") for item in event_records if isinstance(item.get("event"), str)],
            "pose_header_frame_ids": {side: sorted(values) for side, values in pose_frame_ids.items()},
        }
        if sidecar is not None:
            provenance.update(
                {
                    "manifest_status": sidecar.get("status"),
                    "manifest_outcome": sidecar.get("outcome"),
                    "manifest_episode_index": sidecar.get("episode_index"),
                    "manifest_reward": sidecar.get("reward"),
                    "manifest_metadata_events": sidecar.get("metadata_events_recorded"),
                }
            )
        return stats, {"episode": episode_dir.name, "stats": stats.as_dict(), "provenance": provenance}


def _floats(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument(
        "--channels", type=int, choices=(3,), default=3, help="point-cloud channels; XYZ-only"
    )
    parser.add_argument("--sampling", choices=("adaptive", "fps", "random"), default="adaptive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps-candidate-limit", type=int, default=4096)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--rgb-tolerance-ms", type=float, default=45.0)
    parser.add_argument("--depth-tolerance-ms", type=float, default=16.0)
    parser.add_argument("--state-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument(
        "--workspace-min",
        type=_floats,
        default=DEFAULT_WORKSPACE_MIN,
        help="base-frame point-cloud lower bound x,y,z (meters)",
    )
    parser.add_argument(
        "--workspace-max",
        type=_floats,
        default=DEFAULT_WORKSPACE_MAX,
        help="base-frame point-cloud upper bound x,y,z (meters)",
    )
    parser.add_argument("--gripper-closed", type=float, default=0.8)
    parser.add_argument("--gripper-open", type=float, default=0.0)
    parser.add_argument("--task", default="franka duo manipulation")
    parser.add_argument(
        "--mount-to-optical", type=_floats, default=None, help="nominal mount->optical 4x4 override"
    )
    parser.add_argument("--base-rotation", type=_floats, default=None, help="new base rotation 3x3 override")
    parser.add_argument("--force", action="store_true")
    return parser


def convert(args: argparse.Namespace) -> dict[str, Any]:
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"output already exists (use --force): {args.output}")
        shutil.rmtree(args.output)
    if args.mount_to_optical is not None and len(args.mount_to_optical) != 16:
        raise ValueError("--mount-to-optical must contain sixteen values")
    if args.base_rotation is not None and len(args.base_rotation) != 9:
        raise ValueError("--base-rotation must contain nine values")
    if len(args.workspace_min) != 3 or len(args.workspace_max) != 3:
        raise ValueError("--workspace-min and --workspace-max must each contain three values")
    if args.channels != 3:
        raise ValueError("this converter stores XYZ-only point clouds; --channels must be 3")
    geometry = load_usd_geometry(
        args.usd,
        mount_to_optical=args.mount_to_optical,
        base_rotation=args.base_rotation,
    )
    writer = LeRobotV3Writer(
        args.output.expanduser().resolve(), args.fps, args.task, args.num_points, args.channels
    )
    converter = EpisodeConverter(
        geometry,
        writer,
        num_points=args.num_points,
        channels=args.channels,
        sampling=args.sampling,
        seed=args.seed,
        fps_candidate_limit=args.fps_candidate_limit,
        rgb_tolerance_ms=args.rgb_tolerance_ms,
        depth_tolerance_ms=args.depth_tolerance_ms,
        state_tolerance_ms=args.state_tolerance_ms,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        workspace_min=args.workspace_min,
        workspace_max=args.workspace_max,
        gripper_closed=args.gripper_closed,
        gripper_open=args.gripper_open,
    )
    episode_reports = []
    output_episode_index = 0
    for episode_dir in _episode_dirs(args.input_root):
        stats, report = converter.convert_episode(episode_dir, output_episode_index)
        episode_reports.append(report)
        print(
            f"{episode_dir.name}: heads={stats.head_seen} written={stats.frames_written} drops={stats.as_dict()}"
        )
        if stats.frames_written > 0:
            output_episode_index += 1
    writer.finalize()
    manifest = {
        "schema": "franka_duo_tele_data.mcap_to_lerobot.v2",
        "source_root": str(args.input_root.expanduser().resolve()),
        "output": str(args.output.expanduser().resolve()),
        "fps": args.fps,
        "task": args.task,
        "pointcloud": {
            "num_points": args.num_points,
            "channels": args.channels,
            "sampling": args.sampling,
            "seed": args.seed,
            "candidate_limit": args.fps_candidate_limit,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "workspace_min": list(args.workspace_min),
            "workspace_max": list(args.workspace_max),
            "input": "ZED registered depth deprojected with CameraInfo.k and transformed into midpoint base",
            "output": "XYZ-only base-frame points; adaptive voxel representatives and deterministic thinning",
        },
        "images": {
            "wrist_left": {"resize": [256, 256], "channels": 3},
            "wrist_right": {"resize": [256, 256], "channels": 3},
            "head": {"resize": None, "channels": 3},
        },
        "state": "16D measured state: left 7 joints, right 7 joints, left/right actual gripper open fraction",
        "observation_ee_pose": "18D: left xyz+rot6d_rows followed by right xyz+rot6d_rows, relative to midpoint base",
        "pose_rotation_representation": "rot6d_rows: first two rows of each 3x3 rotation matrix flattened row-major",
        "pose_frame_assumption": "left/right current_pose payloads are respectively relative to their arm base links; static USD link0-to-midpoint transforms are applied",
        # Keep the human-readable legacy field and expose the machine-readable
        # action contract separately for real-robot bundle exporters.
        "action": (
            "20D next valid synchronized frame target after 15Hz resampling: "
            "left/right xyz+rot6d_rows followed by normalized left/right gripper "
            "open fractions."
        ),
        "action_dim": ACTION_DIM,
        "action_spec": {
            "dimension": ACTION_DIM,
            "ee_dimension": 9,
            "ee_rotation": "rot6d_rows",
            "layout": {
                "left_ee": [0, 9],
                "right_ee": [9, 18],
                "left_gripper": 18,
                "right_gripper": 19,
            },
            "ee_format": "xyz + continuous rot6d (first two rotation-matrix rows flattened row-major)",
            "gripper_range": [0.0, 1.0],
        },
        "sync": {
            "anchor": "ZED RGB header stamp",
            "zed_stream": "head RGB and registered depth are matched to the same ZED RGB anchor; output is fixed 15Hz by default",
            "output_rate_sampling": "keep first valid synchronized frame on each fixed --fps grid",
            "rgb_tolerance_ms": args.rgb_tolerance_ms,
            "depth_tolerance_ms": args.depth_tolerance_ms,
            "state_tolerance_ms": args.state_tolerance_ms,
            "drop_last_frame_without_next_action": True,
        },
        "gripper_calibration": {"closed_position": args.gripper_closed, "open_position": args.gripper_open},
        "coordinate_transforms": geometry.manifest(),
        "episodes": episode_reports,
    }
    (args.output.expanduser().resolve() / "meta" / "derived_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    convert(args)
    print(f"Wrote LeRobot v3 dataset: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
