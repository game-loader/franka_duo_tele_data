"""Client side of the site joint servo: absolute-step chunk payloads and request pacing.

ROS-free so the message layout and pacing rules can be unit tested.  The servo
reads the absolute step of row zero from ``layout.data_offset`` and publishes
a JSON status; this module builds the former and parses the latter.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

STATUS_SCHEMA = "franka_duo_joint_servo_status_v1"


@dataclass(frozen=True)
class ServoStatus:
    started: bool
    fault: bool
    fault_reason: str
    step: float
    holding: bool
    last_step: int
    playback_speed: float
    tracking_error_rad: float
    chunks: int
    action_rate_hz: float = 30.0
    commit_lead_steps: int = 3
    blend_steps: int = 0
    blend_mode: str = ""
    last_chunk_start_step: int = -1


def parse_status(text: str) -> ServoStatus:
    value = json.loads(text)
    if value.get("schema") != STATUS_SCHEMA:
        raise ValueError("unexpected joint servo status schema")
    return ServoStatus(
        started=bool(value["started"]),
        fault=bool(value["fault"]),
        fault_reason=str(value.get("fault_reason", "")),
        step=float(value["step"]),
        holding=bool(value["holding"]),
        last_step=int(value["last_step"]),
        playback_speed=float(value["playback_speed"]),
        tracking_error_rad=float(value["tracking_error_rad"]),
        chunks=int(value["chunks"]),
        action_rate_hz=float(value.get("action_rate_hz", 30.0)),
        commit_lead_steps=int(value.get("commit_lead_steps", 3)),
        blend_steps=int(value.get("blend_steps", 0)),
        blend_mode=str(value.get("blend_mode", "")),
        last_chunk_start_step=int(value.get("last_chunk_start_step", -1)),
    )


def chunk_payload(actions: np.ndarray, start_step: int) -> tuple[list[float], list[int], int]:
    """Return (data, [rows, 20], data_offset) for a Float32MultiArray chunk."""
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 20 or array.shape[0] < 1:
        raise ValueError("chunk must have shape [rows>=1, 20]")
    if not np.isfinite(array).all():
        raise ValueError("chunk contains non-finite values")
    if start_step < 0:
        raise ValueError("start_step must be nonnegative")
    return array.reshape(-1).tolist(), [int(array.shape[0]), 20], int(start_step)


class ChunkPacer:
    """Decide when to request the next chunk and where it starts.

    Mirrors a real policy: the next chunk starts at the step after the servo's
    current step, overlapping the running plan; the servo blends the overlap.
    Requests are monotonic and never exceed the total step count.
    """

    def __init__(self, horizon: int, total_steps: int):
        if horizon < 2 or total_steps < 1:
            raise ValueError("horizon >= 2 and total_steps >= 1 required")
        self.horizon = horizon
        self.total_steps = total_steps
        self.last_start = -1

    def next_start(self, status: ServoStatus | None) -> int | None:
        """Return the start step of the chunk to request now, or None."""
        if status is None or not status.started:
            return 0 if self.last_start < 0 else None
        if status.fault:
            return None
        # The last chunk already reaches the end of the recording.
        if self.last_start >= 0 and self.last_start + self.horizon >= self.total_steps:
            return None
        remaining = status.last_step - status.step
        if remaining > self.horizon // 2:
            return None
        start = max(int(math.ceil(status.step)) + 1, self.last_start + 1)
        if start >= self.total_steps:
            return None
        return start

    def record(self, start: int) -> None:
        if start <= self.last_start:
            raise ValueError("chunk requests must be monotonic")
        self.last_start = start

    def finished(self, status: ServoStatus | None) -> bool:
        return (
            status is not None
            and status.started
            and status.holding
            and status.step >= self.total_steps - 1
            and status.last_step >= self.total_steps - 1
        )
