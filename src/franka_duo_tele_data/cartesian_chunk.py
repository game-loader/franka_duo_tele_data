"""Absolute-step chunk buffering; Cartesian servo state lives in the site controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action_spec import rot6d_to_matrix
from .rgb20d_io import RGB20DContract, RGB20DObservation


def pose_distance(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions, angles = [], []
    for offset in (0, 9):
        positions.append(float(np.linalg.norm(first[offset : offset + 3] - second[offset : offset + 3])))
        delta = rot6d_to_matrix(first[offset + 3 : offset + 9]).T @ rot6d_to_matrix(
            second[offset + 3 : offset + 9]
        )
        angles.append(float(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
    return np.array(positions), np.array(angles)


@dataclass(frozen=True)
class ChunkRequest:
    start_step: int
    observation: RGB20DObservation


@dataclass(frozen=True)
class ActionChunk:
    start_step: int
    observation_stamp_ns: int
    actions: np.ndarray


class CartesianChunkBuffer:
    """Trim late rows and replace only future targets; never rewind at chunk boundaries."""

    def __init__(
        self,
        contract: RGB20DContract,
        *,
        max_horizon: int = 128,
        max_step_m: float = 0.04,
        max_step_rad: float = 0.35,
    ):
        self.contract = contract
        self.max_horizon = max_horizon
        self.max_step_m = max_step_m
        self.max_step_rad = max_step_rad
        self.cursor = 0
        self.last_start = -1
        self.last_stamp = -1
        self.last_action = None
        self.targets: dict[int, np.ndarray] = {}

    @property
    def remaining(self) -> int:
        return len(self.targets)

    def submit(self, chunk: ActionChunk) -> int:
        actions = np.asarray(chunk.actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 20 or not 0 < len(actions) <= self.max_horizon:
            raise ValueError("action chunk must have shape [1..max_horizon, 20]")
        if (
            chunk.start_step < 0
            or chunk.start_step <= self.last_start
            or chunk.observation_stamp_ns < self.last_stamp
        ):
            raise ValueError("out-of-order action chunk")
        if chunk.start_step > self.cursor:
            raise ValueError("chunk starts after the current control step")
        valid = [self.contract.validate_action(row) for row in actions]
        skipped = self.cursor - chunk.start_step
        if skipped >= len(valid):
            raise TimeoutError("entire action chunk expired during inference")
        previous = self.last_action
        for row in valid[skipped:]:
            if previous is not None:
                position, angle = pose_distance(previous, row)
                if np.any(position > self.max_step_m) or np.any(angle > self.max_step_rad):
                    raise ValueError(
                        f"Cartesian target jump: meters={position.tolist()}, radians={angle.tolist()}"
                    )
            previous = row
        # Validate the complete replacement before touching the active buffer.
        self.targets = {chunk.start_step + i: row for i, row in enumerate(valid) if i >= skipped}
        self.last_start = chunk.start_step
        self.last_stamp = chunk.observation_stamp_ns
        return skipped

    def pop(self) -> np.ndarray:
        if self.cursor not in self.targets:
            raise TimeoutError("action buffer underrun; stop publication and let the controller brake")
        row = self.targets.pop(self.cursor)
        self.last_action = row
        self.cursor += 1
        return row.copy()


def check_tracking(state: np.ndarray, target: np.ndarray, *, max_m: float, max_rad: float) -> None:
    distance, angle = pose_distance(state, target)
    if np.any(distance > max_m) or np.any(angle > max_rad):
        raise ValueError(
            f"live pose too far from target: meters={distance.tolist()}, radians={angle.tolist()}"
        )
