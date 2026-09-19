"""Labs inference contract and measured observations; never used by raw capture."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labs_action_delta import CONVENTION, DELTA_NAMES, DELTA_REPRESENTATION, delta14_to_absolute20
from .labs_kinematics import URDFFK, joint_positions, pose_vector, vector_pose
from .labs_mcap_to_lerobot import CAMERAS, STATE_NAMES, TASK, binary_gripper, numeric_topics, validate_knuckle
from .ros_utils import _stamp_ns, image_msg_to_rgb

SIDES = ("left", "right")
COMMAND_TOPIC = "/franka_duo/labs/action"
STATUS_TOPIC = "/franka_duo/labs/status"
RETURN_SCHEMA = "labs_fr3_episode_joint_return_v2"
COMMAND_SCHEMA = "labs_fr3_link8_absolute20_command_v1"
STATUS_SCHEMA = "labs_fr3_relay_status_v1"
CHUNK_REFERENCE = "request_observation"
FR3_REPRESENTATION = "franka_fr3_duo_link8_delta14_v1"
GRIPPER_POSTPROCESS = "binary_open_ge_0.5_v1"
TOPICS = {**CAMERAS, **{k: v for k, v in numeric_topics().items() if not k.endswith("target")}}


class LabsContract:
    def __init__(self, config: Path, dataset: Path | None = None):
        self.config = Path(config)
        self.fk = {s: URDFFK(self.config / f"{s}.urdf", s) for s in SIDES}
        self.model_hashes = {
            s: hashlib.sha256((self.config / f"{s}.urdf").read_bytes()).hexdigest() for s in SIDES
        }
        self.task = TASK
        if dataset is not None:
            info = json.loads((dataset / "meta/info.json").read_text())
            manifest = json.loads((dataset / "meta/conversion_manifest.json").read_text())
            if (
                info.get("codebase_version") != "v3.0"
                or info.get("fps") != 30
                or info["features"]["observation.state"].get("shape") != [34]
                or info["features"]["observation.state"].get("names") != STATE_NAMES
                or info["features"]["action"].get("shape") != [14]
                or info["features"]["action"].get("names") != DELTA_NAMES
                or manifest.get("schema") != DELTA_REPRESENTATION
                or manifest.get("delta_convention") != CONVENTION
                or manifest.get("tool_offset_m") != 0
                or manifest.get("frames") != {s: f"{s}_fr3_link0" for s in SIDES}
                or manifest.get("tips") != {s: f"{s}_fr3_link8" for s in SIDES}
            ):
                raise ValueError("Expected Labs link0/link8 state34 delta14 dataset contract")
            for camera in CAMERAS:
                if info["features"][f"observation.images.{camera}"].get("shape") != [480, 640, 3]:
                    raise ValueError("Labs images must be 640x480 RGB")
            self.task = manifest["task"]

    def state(self, joints, grippers):
        poses = [pose_vector(self.fk[s](joints[s])) for s in SIDES]
        return self.validate_state(np.concatenate((*poses, grippers, *(joints[s] for s in SIDES))))

    def validate_state(self, state):
        value = np.asarray(state, dtype=np.float64)
        if value.shape != (34,) or not np.isfinite(value).all():
            raise ValueError("Labs state must be finite float32[34]")
        if not np.isin(value[18:20], [0, 1]).all():
            raise ValueError("Labs measured grippers must be binary closed=0/open=1")
        for s, pose_start, q_start in (("left", 0, 20), ("right", 9, 27)):
            q = value[q_start : q_start + 7]
            bounds = np.asarray(self.fk[s].bounds)
            if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
                raise ValueError(f"{s} joints outside saved URDF bounds")
            expected = pose_vector(self.fk[s](q))
            if not np.allclose(value[pose_start : pose_start + 9], expected, atol=2e-6, rtol=0):
                raise ValueError(f"{s} state pose must be FK of its measured joints in link0/link8")
        return value.astype(np.float32)


def episode_start(dataset: Path, episode: int, contract):
    """Read the actual measured first frame, never a desired/action label."""
    import pyarrow.parquet as pq

    if episode < 0:
        raise ValueError("Episode index must be nonnegative")
    found = []
    for path in sorted((dataset / "data").rglob("*.parquet")):
        table = pq.read_table(
            path,
            columns=["observation.state"],
            filters=[("episode_index", "=", episode), ("frame_index", "=", 0)],
        )
        found.extend(table["observation.state"].to_pylist())
    if len(found) != 1:
        raise ValueError(f"Expected exactly one episode {episode}, frame 0; found {len(found)}")
    return contract.validate_state(found[0])


def start_identity(state, episode, contract):
    """Content identity portable across client/relay dataset mount paths."""
    value = contract.validate_state(state)
    payload = {
        "episode_index": episode,
        "frame_index": 0,
        "state34": value.tolist(),
        "model_hashes": contract.model_hashes,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_response(result):
    if not isinstance(result, dict) or result.get("action_representation") not in (
        DELTA_REPRESENTATION,
        FR3_REPRESENTATION,
    ):
        raise ValueError(f"Server must declare action_representation={DELTA_REPRESENTATION}")
    if result.get("normalized") is not False:
        raise ValueError("Server must explicitly return normalized=false physical actions")
    actions = np.asarray(result.get("actions"), dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 14 or not 1 <= len(actions) <= 256:
        raise ValueError("Expected [H,14] actions, 1 <= H <= 256")
    if not np.isfinite(actions).all():
        raise ValueError("Nonfinite actions")
    # Binary-label regression can slightly overshoot an endpoint. Retain the
    # raw reply and reject gross scale errors before applying the label boundary.
    if np.any((actions[:, 12:] < -0.1) | (actions[:, 12:] > 1.1)):
        raise ValueError("Gripper prediction exceeds [-0.1,1.1] guard")
    result = actions.copy()
    result[:, 12:] = (actions[:, 12:] >= 0.5).astype(float)
    return result


def reconstruct_chunk(result, state, *, max_delta_m=0.25, max_delta_rad=1.0):
    """The server rebases ALL rows to this request's measured observation.

    This is the explicitly agreed inference-chunk contract. The dataset's
    individual rows still reference their own states; do not change its labels.
    """
    delta = validate_response(result)
    for offset in (0, 6):
        if np.any(np.linalg.norm(delta[:, offset : offset + 3], axis=1) > max_delta_m):
            raise ValueError("Labs translation delta exceeds limit")
        if np.any(np.linalg.norm(delta[:, offset + 3 : offset + 6], axis=1) > max_delta_rad):
            raise ValueError("Labs rotation delta exceeds limit")
    references = np.broadcast_to(state, (len(delta), 34))
    return delta14_to_absolute20(delta, references)


def validate_target(target):
    value = np.asarray(target, dtype=np.float64)
    if value.shape != (20,) or not np.isfinite(value).all():
        raise ValueError("Expected finite absolute20 target")
    for offset in (0, 9):
        vector_pose(value[offset : offset + 9])
    if not np.isin(value[18:], [0, 1]).all():
        raise ValueError("Labs target grippers must be binary")
    return value


@dataclass
class Sample:
    stamp_ns: int
    received_ns: int
    value: object


@dataclass
class LabsObservation:
    state: np.ndarray
    images: dict
    stamp_ns: int
    source_stamps_ns: dict


def resize_image(message):
    import av

    # Same libswscale direct resize as labs_mcap_to_lerobot; no crop or letterbox.
    rgb = image_msg_to_rgb(message)
    return (
        av.VideoFrame.from_ndarray(rgb, format="rgb24")
        .reformat(width=640, height=480)
        .to_ndarray(format="rgb24")
    )


class LabsObservationCache:
    def __init__(self, contract, *, max_age_ms=200):
        self.contract = contract
        self.max_age_ns = int(max_age_ms * 1e6)
        if self.max_age_ns <= 60_000_000:
            raise ValueError("Input age must allow 60 ms synchronization lookahead")
        self.buffers = {k: deque(maxlen=12 if k in CAMERAS else 1200) for k in TOPICS}
        self.condition = threading.Condition()
        self.last_stamp = 0
        self.error = None

    def store(self, key, message):
        try:
            stamp = _stamp_ns(message)
            if stamp is None:
                raise ValueError(f"{key} needs a nonzero header stamp")
            if key in CAMERAS:
                value = message  # Decode selected images only, outside the ROS callback.
            elif key.endswith("_q"):
                value = joint_positions(message.name, message.position, key.split("_")[0])
            else:
                names = list(message.name)
                indices = [
                    i for i, name in enumerate(names) if name.endswith("robotiq_85_left_knuckle_joint")
                ]
                if len(indices) != 1 or len(message.position) != len(names):
                    raise ValueError("Expected one named Robotiq knuckle")
                value = float(message.position[indices[0]])
                validate_knuckle(value)
            with self.condition:
                self.buffers[key].append(Sample(stamp, time.monotonic_ns(), value))
                self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def select(self, *, after_ns=0, now_ns=None, monotonic_ns=None):
        now = time.time_ns() if now_ns is None else now_ns
        mono = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
        with self.condition:
            if self.error is not None:
                raise ValueError(f"Invalid live Labs observation: {self.error}") from self.error
            history = {k: list(v) for k, v in self.buffers.items()}

        def fresh(sample):
            return (
                0 <= now - sample.stamp_ns <= self.max_age_ns
                and 0 <= mono - sample.received_ns <= self.max_age_ns
            )

        for head in reversed(history["head"]):
            if head.stamp_ns <= max(after_ns, self.last_stamp) or not fresh(head):
                continue
            if now - head.stamp_ns < 60_000_000:
                continue
            selected = {"head": head}
            for key, values in history.items():
                if key == "head":
                    continue
                tolerance = 45_000_000 if key in CAMERAS else 50_000_000
                candidates = [v for v in values if fresh(v) and abs(v.stamp_ns - head.stamp_ns) <= tolerance]
                if not candidates:
                    break
                selected[key] = min(candidates, key=lambda v: abs(v.stamp_ns - head.stamp_ns))
            if len(selected) == len(TOPICS):
                self.last_stamp = head.stamp_ns
                return selected
        return None

    def next(self, *, after_ns=0, timeout_s=3):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            selected = self.select(after_ns=after_ns)
            if selected is not None:
                joints = {s: selected[f"{s}_q"].value for s in SIDES}
                grippers = [binary_gripper(selected[f"{s}_gripper"].value) for s in SIDES]
                return LabsObservation(
                    self.contract.state(joints, grippers),
                    {k: resize_image(selected[k].value) for k in CAMERAS},
                    selected["head"].stamp_ns,
                    {k: v.stamp_ns for k, v in selected.items()},
                )
            with self.condition:
                self.condition.wait(timeout=min(0.02, max(0, deadline - time.monotonic())))
        raise TimeoutError("No fresh synchronized Labs head/wrists/measured joints/grippers")
