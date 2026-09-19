"""Stream versioned delta14 predictions through the site's absolute20 joint servo.

Dry-run by default, with local JSONL diagnostics and no MCAP recording.
Each chunk is integrated from its request observation in midpoint-base axes.
Wait for all 32 steps and servo hold before observing and requesting again.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config_io import load_mapping
from .delta14_client import TASK, Delta14Client, reconstruct_absolute20, validate_response
from .smolvla_once import sanitize_chunk
from .smolvla_stream import build_parser, parse_stream_args, run as run_stream


def prepare_delta14_chunk(result, contract, state, args, config):
    delta = validate_response(result)
    # Check the supplied increments too: Exp(2*pi) must not bypass rotation limits.
    for offset in (0, 6):
        if np.any(np.linalg.norm(delta[:, offset : offset + 3], axis=1) > config["max_target_step_m"]):
            raise ValueError("delta14 translation increment exceeds max_target_step_m")
        if np.any(np.linalg.norm(delta[:, offset + 3 : offset + 6], axis=1) > config["max_target_step_rad"]):
            raise ValueError("delta14 rotation increment exceeds max_target_step_rad")
    absolute = reconstruct_absolute20(result, state)
    return sanitize_chunk(
        absolute,
        contract,
        state,
        max_step_m=config["max_target_step_m"],
        max_step_rad=config["max_target_step_rad"],
        first_offset_m=args.max_first_offset_m,
        first_offset_rad=args.max_first_offset_rad,
        binary_grippers=False,
    )


def run(args) -> int:
    if args.publish != args.enable_robot:
        raise ValueError("robot publication requires both --publish and --enable-robot")
    config = load_mapping(args.config)
    if config["joint_servo_chunk_topic"] != "/franka_duo/joint_servo/action_chunk":
        raise ValueError("delta14 output must target the site-owned joint servo chunk relay")
    return run_stream(args, client_type=Delta14Client, prepare_chunk=prepare_delta14_chunk, sequential=True)


def main(argv=None) -> int:
    parser = build_parser(__doc__)
    parser.set_defaults(
        url="ws://127.0.0.1:8081/infer",
        output=Path("outputs/delta14_stream.jsonl"),
    )
    args = parse_stream_args(parser, argv)
    if args.task != TASK:
        parser.error(f"delta14 server currently supports only {TASK!r}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
