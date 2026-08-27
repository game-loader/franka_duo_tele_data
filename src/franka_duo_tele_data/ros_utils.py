"""ROS message decoding used only by live policy evaluation.

MCAP capture does not call these helpers: it records serialized ROS messages
unchanged and defers synchronization and conversion to offline processing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SPINE_JOINT = "franka_spine_vertical_joint"
GRIPPER_CLOSED_POSITION = 0.8


def _candidate_joint_names(side: str, index: int) -> tuple[str, ...]:
    side = side.lower()
    return (
        f"{side}_fr3v2_joint{index}",
        f"{side}_fr3v2_1_joint{index}",
        f"{side}_fr3_joint{index}",
        f"{side}_joint{index}",
        f"{side}_panda_joint{index}",
    )


def _resolve(mapping: Mapping[str, float], names: Sequence[str], default: float = math.nan) -> float:
    for name in names:
        value = mapping.get(name)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return default


def _arm_joints(mapping: Mapping[str, float], side: str) -> np.ndarray:
    values = np.asarray(
        [_resolve(mapping, _candidate_joint_names(side, index)) for index in range(1, 8)],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        missing = [index + 1 for index, value in enumerate(values) if not math.isfinite(float(value))]
        raise ValueError(f"Missing {side} arm joints {missing}")
    return values


def _gripper_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_right_finger_joint",
        f"{side}_fr3v2_finger_joint1",
        f"{side}_franka_finger_joint1",
        f"{side}_panda_finger_joint1",
        f"{side}_finger_joint1",
        f"{side}_gripper_width",
        f"{side}_gripper",
        f"{side}_hand_joint",
    )


def gripper_open_fraction(
    position: float,
    closed_position: float = GRIPPER_CLOSED_POSITION,
    open_position: float = 0.0,
) -> float:
    if not math.isfinite(float(position)):
        return math.nan
    if not math.isfinite(float(closed_position)) or not math.isfinite(float(open_position)):
        raise ValueError("gripper calibration endpoints must be finite")
    if math.isclose(float(closed_position), float(open_position), abs_tol=1e-12):
        raise ValueError("gripper calibration endpoints must differ")
    fraction = (float(position) - float(closed_position)) / (float(open_position) - float(closed_position))
    return float(np.clip(fraction, 0.0, 1.0))


def _joint_map(message: Any) -> dict[str, float]:
    names = list(getattr(message, "name", ()))
    positions = list(getattr(message, "position", ()))
    return {str(name): float(position) for name, position in zip(names, positions, strict=False)}


def build_state(
    measured: Mapping[str, float],
    *,
    closed_rad: float = GRIPPER_CLOSED_POSITION,
    open_rad: float = 0.0,
    include_spine: bool = True,
) -> np.ndarray:
    """Build an optional policy state from an aggregate semantic JointState."""

    left = _arm_joints(measured, "left")
    right = _arm_joints(measured, "right")
    tail = [
        gripper_open_fraction(_resolve(measured, _gripper_names("left")), closed_rad, open_rad),
        gripper_open_fraction(_resolve(measured, _gripper_names("right")), closed_rad, open_rad),
    ]
    if include_spine:
        tail.append(_resolve(measured, (SPINE_JOINT,)))
    values = np.concatenate((left, right, np.asarray(tail, dtype=np.float32))).astype(np.float32)
    expected = 17 if include_spine else 16
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise ValueError("Joint state topic is missing a finite value for every required channel")
    return values


def _stamp_ns(message: Any) -> int | None:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return None
    try:
        value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def _image_rows(message: Any, dtype: np.dtype) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    itemsize = np.dtype(dtype).itemsize
    if step % itemsize:
        raise ValueError("Image step is not aligned to the pixel dtype")
    row_values = step // itemsize
    byte_order = ">" if bool(getattr(message, "is_bigendian", False)) else "<"
    message_dtype = np.dtype(dtype).newbyteorder(byte_order)
    raw = np.frombuffer(message.data, dtype=message_dtype)
    if raw.size < height * row_values:
        raise ValueError("Image message data is shorter than height*step")
    return raw[: height * row_values].reshape(height, row_values)[:, :width]


def _yuv422_to_rgb(message: Any, *, uyvy: bool) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    if width % 2:
        raise ValueError("YUV422 images must have an even width")
    row_bytes = width * 2
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if step < row_bytes or raw.size < height * step:
        raise ValueError("YUV422 message step/data are inconsistent with width")
    packed = raw[: height * step].reshape(height, step)[:, :row_bytes].reshape(height, width // 2, 4)
    if uyvy:
        u, y0, v, y1 = (packed[..., index].astype(np.int32) for index in range(4))
    else:
        y0, u, y1, v = (packed[..., index].astype(np.int32) for index in range(4))
    y = np.empty((height, width), dtype=np.int32)
    y[:, 0::2] = y0
    y[:, 1::2] = y1
    u = np.repeat(u, 2, axis=1) - 128
    v = np.repeat(v, 2, axis=1) - 128
    c = np.maximum(y - 16, 0)
    red = (298 * c + 409 * v + 128) >> 8
    green = (298 * c - 100 * u - 208 * v + 128) >> 8
    blue = (298 * c + 516 * u + 128) >> 8
    rgb = np.stack((red, green, blue), axis=-1)
    return np.ascontiguousarray(np.clip(rgb, 0, 255), dtype=np.uint8)


def image_msg_to_rgb(message: Any, expected_shape: tuple[int, int, int] | None = None) -> np.ndarray:
    """Decode common ROS image encodings into contiguous HWC RGB uint8."""

    encoding = str(getattr(message, "encoding", "")).lower()
    if encoding in {"yuyv", "yuy2", "yuv422_yuy2"}:
        result = _yuv422_to_rgb(message, uyvy=False)
    elif encoding in {"uyvy", "yuv422"}:
        result = _yuv422_to_rgb(message, uyvy=True)
    elif encoding in {"mono8", "8uc1"}:
        result = np.repeat(_image_rows(message, np.uint8)[:, :, None], 3, axis=2)
    elif encoding in {"rgb8", "bgr8", "rgba8", "bgra8"}:
        channels = 4 if "a8" in encoding else 3
        height = int(message.height)
        width = int(message.width)
        step = int(message.step)
        row_bytes = width * channels
        raw = np.frombuffer(message.data, dtype=np.uint8)
        if step < row_bytes or raw.size < height * step:
            raise ValueError("RGB message step/data are inconsistent with width and encoding")
        result = raw[: height * step].reshape(height, step)[:, :row_bytes].reshape(height, width, channels)
        result = result[:, :, :3]
        if encoding in {"bgr8", "bgra8"}:
            result = result[:, :, ::-1]
    else:
        raise ValueError(f"Unsupported RGB image encoding {encoding!r}")
    if expected_shape is not None and tuple(result.shape) != tuple(expected_shape):
        raise ValueError(f"RGB shape {result.shape} does not match configured {expected_shape}")
    return np.ascontiguousarray(result, dtype=np.uint8)


def depth_msg_to_meters(
    message: Any,
    depth_scale: float = 0.001,
    *,
    dtype: np.dtype | type = np.float16,
) -> np.ndarray:
    """Decode registered ROS depth into a contiguous meter image.

    Live evaluation defaults to ``float16`` to keep the callback cache small.
    Offline point-cloud conversion can request ``float32`` so a ``32FC1`` ZED
    payload is not quantized before deprojection.
    """

    encoding = str(getattr(message, "encoding", "")).lower()
    if encoding in {"32fc1", "32fc"}:
        values = _image_rows(message, np.float32).astype(np.float32, copy=False)
    elif encoding in {"16uc1", "mono16"}:
        values = _image_rows(message, np.uint16).astype(np.float32) * float(depth_scale)
    else:
        raise ValueError(f"Unsupported depth encoding {encoding!r}")
    output_dtype = np.dtype(dtype)
    if output_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError("depth dtype must be float16 or float32")
    return np.ascontiguousarray(values.astype(output_dtype.newbyteorder("<")))
