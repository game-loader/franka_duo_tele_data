"""Versioned FastWAM delta14 client on the SmolVLA binary WebSocket transport."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from .action_spec import FrankaDuoActionSpec, matrix_to_rot6d, rot6d_to_matrix
from .smolvla_client import SmolVLAClient

ACTION_REPRESENTATION = "franka_duo_midpoint_delta14_v1"
TASK = "pick cup and bowl"
HORIZON = 32


def validate_response(result: dict) -> np.ndarray:
    if result.get("action_representation") != ACTION_REPRESENTATION:
        raise ValueError(f"expected action_representation={ACTION_REPRESENTATION}")
    if result.get("normalized") is not False:
        raise ValueError("delta14 response must explicitly declare normalized=False")
    actions = np.asarray(result.get("actions"), dtype=np.float64)
    if actions.shape != (HORIZON, 14) or not np.isfinite(actions).all():
        raise ValueError(f"delta14 actions must be finite [32,14], got {actions.shape}")
    if np.any((actions[:, 12:] < 0) | (actions[:, 12:] > 1)):
        raise ValueError("delta14 gripper openness must be in [0,1]")
    return actions


class Delta14Client(SmolVLAClient):
    action_representation = ACTION_REPRESENTATION

    async def infer(self, state: list[float], images: dict[str, bytes], task=TASK) -> dict:
        if task != TASK:
            raise ValueError(f"delta14 server currently supports only {TASK!r}")
        FrankaDuoActionSpec().validate(state, clip=False)
        if set(images) != {"head", "wrist_left", "wrist_right"} or any(
            not isinstance(value, bytes) or not value for value in images.values()
        ):
            raise ValueError("provide JPEG/PNG bytes for head, wrist_left and wrist_right")
        result = await super().infer(state, images, task)
        validate_response(result)
        return result


def reconstruct_absolute20(result: dict, state: np.ndarray) -> np.ndarray:
    """Integrate ALL rows from this request's observation before any time trimming.

    Translation is in midpoint-base axes. Rotation increments left-multiply;
    grippers are absolute, with no threshold, normalization or sign correction.
    """
    from scipy.spatial.transform import Rotation

    delta = validate_response(result)
    previous = FrankaDuoActionSpec().validate(state, clip=False).astype(np.float64)
    absolute = np.empty((HORIZON, 20), dtype=np.float64)
    for source, target in ((0, 0), (6, 9)):
        absolute[:, target : target + 3] = previous[target : target + 3] + np.cumsum(
            delta[:, source : source + 3], axis=0
        )
        rotation = rot6d_to_matrix(previous[target + 3 : target + 9]).astype(np.float64)
        increments = Rotation.from_rotvec(delta[:, source + 3 : source + 6]).as_matrix()
        for index, increment in enumerate(increments):
            rotation = increment @ rotation
            absolute[index, target + 3 : target + 9] = matrix_to_rot6d(rotation)
    absolute[:, 18:] = delta[:, 12:]
    if not np.isfinite(absolute).all():
        raise ValueError("delta14 accumulation overflow")
    return absolute


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8081/infer")
    parser.add_argument("--state-json", type=Path, required=True)
    for camera in ("head", "wrist-left", "wrist-right"):
        parser.add_argument(f"--{camera}", type=Path, required=True)
    parser.add_argument("--task", choices=(TASK,), default=TASK)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    state = json.loads(args.state_json.read_text())
    images = {key: getattr(args, key).read_bytes() for key in ("head", "wrist_left", "wrist_right")}

    async def session():
        async with Delta14Client(args.url) as client:
            for _ in range(args.repeat):
                print(json.dumps(await client.infer(state, images, args.task), allow_nan=False))

    asyncio.run(session())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
