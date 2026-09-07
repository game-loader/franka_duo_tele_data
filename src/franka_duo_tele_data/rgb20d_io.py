"""RGB-only policy input with the same geometry and encoding as MCAP conversion."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .action_spec import FrankaDuoActionSpec
from .franka_duo_eval_io import EvalObservationCache, TimedValue, _nearest
from .mcap_to_lerobot import (
    _binary_gripper_state,
    _gripper_position,
    compose_transform,
    invert_transform,
    pose_to_transform,
    pose_vector,
)
from .ros_utils import _stamp_ns, image_msg_to_rgb

CAMERAS = ("head", "wrist_left", "wrist_right")
CAMERA_MAP = {
    f"observation.images.{name}": f"observation.images.camera{i + 1}" for i, name in enumerate(CAMERAS)
}


@dataclass(frozen=True)
class RGB20DObservation:
    stamp_ns: int
    state: np.ndarray
    images: dict[str, np.ndarray]
    source_stamps_ns: dict[str, int]
    arrival_ns: int

    def policy_input(self) -> dict[str, np.ndarray]:
        # Transport boundary: uint8 RGB HWC and float32 state, no resizing or
        # normalization. The server's policy preprocessor owns CHW conversion.
        return {
            "observation.state": self.state.copy(),
            **{CAMERA_MAP[f"observation.images.{key}"]: value for key, value in self.images.items()},
        }


class RGB20DContract:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.manifest = json.loads((self.root / "franka_duo_extras/derived_manifest.json").read_text())
        if self.manifest.get("schema") != "franka_duo_tele_data.mcap_to_lerobot.rgb20d.v1":
            raise ValueError("expected a versioned RGB20D conversion manifest")
        self.fps = int(self.info["fps"])
        if self.fps != 30:
            raise ValueError("RGB20D execution requires the dataset's 30 Hz contract")
        for key in ("observation.state", "action"):
            if self.info["features"][key]["shape"] != [20]:
                raise ValueError(f"{key} must be 20D")
        self.image_shapes = {}
        for key in CAMERAS:
            feature = self.info["features"][f"observation.images.{key}"]
            shape = tuple(feature["shape"])
            if feature["dtype"] != "video" or len(shape) != 3 or shape[-1] != 3:
                raise ValueError(f"{key} must be an HWC RGB video")
            self.image_shapes[key] = shape
        self.transforms = {}
        for side in ("left", "right"):
            matrix = np.asarray(
                self.manifest["coordinate_transforms"][f"T_newbase_from_{side}_arm_base"],
                dtype=np.float32,
            )
            invert_transform(matrix)
            if not np.allclose(matrix[3], (0, 0, 0, 1)) or not np.isclose(
                np.linalg.det(matrix[:3, :3]), 1, atol=2e-4
            ):
                raise ValueError(f"{side} arm transform must be rigid and right handed")
            self.transforms[side] = matrix
        self.action_spec = FrankaDuoActionSpec.from_manifest(self.manifest)
        self.gripper = self.manifest["gripper_calibration"]
        if self.gripper.get("encoding") != "0=closed, 1=open":
            raise ValueError("unsupported gripper polarity")

    def state_from_messages(self, selected: dict[str, Any]) -> np.ndarray:
        poses = [
            pose_vector(
                compose_transform(self.transforms[side], pose_to_transform(selected[f"{side}_pose"].pose))
            )
            for side in ("left", "right")
        ]
        grippers = [
            _binary_gripper_state(
                _gripper_position(selected[f"{side}_gripper"]),
                closed_position=self.gripper["closed_position"],
                open_position=self.gripper["open_position"],
                threshold=self.gripper["threshold"],
            )
            for side in ("left", "right")
        ]
        state = np.concatenate((*poses, grippers)).astype(np.float32)
        self.action_spec.validate(state)
        return state

    def validate_action(self, action: np.ndarray) -> np.ndarray:
        result = self.action_spec.validate(action)
        if not np.isin(result[18:], (0.0, 1.0)).all():
            raise ValueError("RGB20D action grippers must be binary 0=closed, 1=open")
        return result


class RGB20DCache(EvalObservationCache):
    """Keep ROS receipt time for headerless grippers, matching raw rosbag2."""

    def _store(self, target, message: Any) -> None:
        with self._condition:
            target.append(TimedValue(message, time.monotonic_ns(), _stamp_ns(message) or time.time_ns()))
            self._revision += 1
            self._condition.notify_all()


class RGB20DReader:
    def __init__(self, cache: RGB20DCache, contract: RGB20DContract, *, max_age_ms: float = 200):
        self.cache = cache
        self.contract = contract
        self.last_stamp = -1
        self.max_age_ns = int(max_age_ms * 1e6)
        sync = contract.manifest["sync"]
        self.rgb_tolerance_ns = int(sync["rgb_tolerance_ms"] * 1e6)
        self.state_tolerance_ns = int(sync["state_tolerance_ms"] * 1e6)
        self.settle_ns = max(self.rgb_tolerance_ns, self.state_tolerance_ns) + 10_000_000
        if self.max_age_ns <= self.settle_ns:
            raise ValueError("max_age_ms must exceed the synchronization lookahead")

    def next(self, timeout_s: float = 1.0) -> RGB20DObservation:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snapshot = self.cache.snapshot()
            now = time.monotonic_ns()
            # Wait the same lookahead as offline conversion, then take the
            # newest complete head frame to avoid accumulating camera latency.
            heads = [
                h
                for h in snapshot["images"]["head"]
                if h.stamp_ns > self.last_stamp and self.settle_ns <= now - h.arrival_ns <= self.max_age_ns
            ]
            if heads:
                head = max(heads, key=lambda h: h.stamp_ns)
                selected = {"head": head}
                for key in CAMERAS[1:]:
                    selected[key] = _nearest(snapshot["images"][key], head.stamp_ns, self.rgb_tolerance_ns)
                for side in ("left", "right"):
                    for kind, suffix in (("pose", "pose"), ("gripper", "gripper_states")):
                        selected[f"{side}_{kind}"] = _nearest(
                            snapshot[f"{side}_{suffix}"], head.stamp_ns, self.state_tolerance_ns
                        )
                if all(v is not None and now - v.arrival_ns <= self.max_age_ns for v in selected.values()):
                    images = {
                        key: image_msg_to_rgb(
                            selected[key].message, expected_shape=self.contract.image_shapes[key]
                        )
                        for key in CAMERAS
                    }
                    state = self.contract.state_from_messages({k: v.message for k, v in selected.items()})
                    self.last_stamp = head.stamp_ns
                    return RGB20DObservation(
                        head.stamp_ns,
                        state,
                        images,
                        {key: value.stamp_ns for key, value in selected.items()},
                        min(value.arrival_ns for value in selected.values()),
                    )
            self.cache.wait_for_update(snapshot["revision"], min(0.005, max(0, deadline - time.monotonic())))
        raise TimeoutError(
            "no fresh synchronized head/wrists/EE poses/grippers for RGB20D input; "
            + self.input_diagnostics()
        )

    def input_diagnostics(self) -> str:
        """Describe missing/stale streams without changing matching or freshness limits."""
        snapshot = self.cache.snapshot()
        now = time.monotonic_ns()
        streams = {key: snapshot["images"][key] for key in CAMERAS}
        for side in ("left", "right"):
            streams[f"{side}_pose"] = snapshot[f"{side}_pose"]
            streams[f"{side}_gripper"] = snapshot[f"{side}_gripper_states"]
        details: dict[str, Any] = {
            "missing_inputs": [key for key, values in streams.items() if not values],
            "latest_arrival_age_ms": {
                key: round((now - values[-1].arrival_ns) / 1e6, 1)
                for key, values in streams.items()
                if values
            },
        }
        heads = [
            value
            for value in streams["head"]
            if value.stamp_ns > self.last_stamp
            and self.settle_ns <= now - value.arrival_ns <= self.max_age_ns
        ]
        if heads:
            head = max(heads, key=lambda value: value.stamp_ns)
            details["nearest_to_head_ms"] = {
                key: round(
                    (
                        min(values, key=lambda value: abs(value.stamp_ns - head.stamp_ns)).stamp_ns
                        - head.stamp_ns
                    )
                    / 1e6,
                    1,
                )
                for key, values in streams.items()
                if values
            }
        return json.dumps(details, sort_keys=True)


class RobotStateReader:
    """Fresh robot feedback for recorded-action diagnostics without cameras."""

    def __init__(self, cache: RGB20DCache, contract: RGB20DContract, *, max_age_ms: float = 200):
        self.cache, self.contract = cache, contract
        self.max_age_ns = int(max_age_ms * 1e6)
        self.tolerance_ns = int(contract.manifest["sync"]["state_tolerance_ms"] * 1e6)

    def next(self, timeout_s: float = 1.0) -> RGB20DObservation:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snapshot = self.cache.snapshot()
            now = time.monotonic_ns()
            poses = [snapshot[f"{side}_pose"] for side in ("left", "right")]
            if all(poses):
                stamp = min(items[-1].stamp_ns for items in poses)
                selected = {
                    f"{side}_{kind}": _nearest(snapshot[f"{side}_{suffix}"], stamp, self.tolerance_ns)
                    for side in ("left", "right")
                    for kind, suffix in (("pose", "pose"), ("gripper", "gripper_states"))
                }
                if all(v is not None and now - v.arrival_ns <= self.max_age_ns for v in selected.values()):
                    state = self.contract.state_from_messages({k: v.message for k, v in selected.items()})
                    return RGB20DObservation(
                        stamp,
                        state,
                        {},
                        {key: value.stamp_ns for key, value in selected.items()},
                        min(value.arrival_ns for value in selected.values()),
                    )
            self.cache.wait_for_update(snapshot["revision"], min(0.005, max(0, deadline - time.monotonic())))
        raise TimeoutError("no fresh paired EE poses/grippers for recorded-action replay")
