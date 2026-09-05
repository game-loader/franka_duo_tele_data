#!/usr/bin/env python3
"""Export a LeRobot ``pretrained_model`` checkpoint for Franka Duo eval.

The exported directory is self-contained: it contains the LeRobot policy
artifacts plus an explicit manifest describing the 34D observation, 2048-point
cloud preprocessing, 20D action contract, and optional base-to-link0
transforms.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .action_spec import ACTION_DIM, FrankaDuoActionSpec, action_spec_manifest
from .franka_duo_eval_io import PointCloudConfig


def _floats(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--channels", type=int, choices=(3, 6), default=3)
    parser.add_argument("--sampling", choices=("adaptive", "fps"), default="adaptive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workspace-min", type=_floats, default=None)
    parser.add_argument("--workspace-max", type=_floats, default=None)
    parser.add_argument("--extrinsics", type=_floats, default=None)
    parser.add_argument("--left-link0-from-base", type=_floats, default=None)
    parser.add_argument("--right-link0-from-base", type=_floats, default=None)
    parser.add_argument(
        "--derived-manifest",
        type=Path,
        default=None,
        help="Optional mcap_to_lerobot derived_manifest.json supplying calibrated transforms",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def build_manifest(args: argparse.Namespace) -> dict:
    coordinate_transforms = None
    if args.derived_manifest is not None:
        source = args.derived_manifest.expanduser().resolve()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid derived manifest {source}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("coordinate_transforms"), dict):
            raise ValueError(f"derived manifest lacks coordinate_transforms: {source}")
        coordinate_transforms = payload["coordinate_transforms"]
    pointcloud = PointCloudConfig(
        num_points=args.num_points,
        channels=args.channels,
        sampling=args.sampling,
        seed=args.seed,
        workspace_min=tuple(args.workspace_min) if args.workspace_min is not None else None,
        workspace_max=tuple(args.workspace_max) if args.workspace_max is not None else None,
        extrinsics=tuple(args.extrinsics) if args.extrinsics is not None else None,
    )
    for name, value, size in (
        ("--workspace-min", args.workspace_min, 3),
        ("--workspace-max", args.workspace_max, 3),
        ("--extrinsics", args.extrinsics, 16),
        ("--left-link0-from-base", args.left_link0_from_base, 16),
        ("--right-link0-from-base", args.right_link0_from_base, 16),
    ):
        if value is not None and len(value) != size:
            raise ValueError(f"{name} must contain {size} values")
    left_link0_from_base = tuple(args.left_link0_from_base) if args.left_link0_from_base is not None else None
    right_link0_from_base = (
        tuple(args.right_link0_from_base) if args.right_link0_from_base is not None else None
    )
    if coordinate_transforms is not None:
        for key, target in (
            ("left_link0_from_base", "T_newbase_from_left_arm_base"),
            ("right_link0_from_base", "T_newbase_from_right_arm_base"),
        ):
            value = coordinate_transforms.get(target)
            if value is None:
                continue
            matrix = np.asarray(value, dtype=np.float32).reshape(4, 4)
            inverse = np.eye(4, dtype=np.float32)
            inverse[:3, :3] = matrix[:3, :3].T
            inverse[:3, 3] = -inverse[:3, :3] @ matrix[:3, 3]
            inferred = tuple(float(x) for x in inverse.reshape(-1))
            if key == "left_link0_from_base" and left_link0_from_base is None:
                left_link0_from_base = inferred
            if key == "right_link0_from_base" and right_link0_from_base is None:
                right_link0_from_base = inferred
    spec = FrankaDuoActionSpec(
        dimension=ACTION_DIM,
        workspace_min=tuple(args.workspace_min) if args.workspace_min is not None else None,
        workspace_max=tuple(args.workspace_max) if args.workspace_max is not None else None,
        left_link0_from_base=left_link0_from_base,
        right_link0_from_base=right_link0_from_base,
    )
    manifest = {
        "manifest_version": 1,
        "backend": "lerobot",
        "policy_dir": "pretrained_model",
        "action_dim": ACTION_DIM,
        "action_spec": action_spec_manifest(spec),
        "pointcloud": {
            "num_points": pointcloud.num_points,
            "channels": pointcloud.channels,
            "sampling": pointcloud.sampling,
            "seed": pointcloud.seed,
            "workspace_min": list(pointcloud.workspace_min) if pointcloud.workspace_min else None,
            "workspace_max": list(pointcloud.workspace_max) if pointcloud.workspace_max else None,
            "extrinsics": list(pointcloud.extrinsics) if pointcloud.extrinsics else None,
        },
        "inputs": {
            "point_cloud_key": "observation.point_cloud",
            "state_key": "observation.state",
            "image_keys": {
                "observation.images.wrist_left": "wrist_left",
                "observation.images.wrist_right": "wrist_right",
            },
        },
    }
    if coordinate_transforms is not None:
        manifest["coordinate_transforms"] = coordinate_transforms
    if args.extrinsics is None and coordinate_transforms is None:
        raise ValueError(
            "Provide --derived-manifest or --extrinsics so eval uses the exact "
            "mcap_to_lerobot ZED-to-base transform."
        )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    required = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"LeRobot checkpoint is missing {missing}: {checkpoint}")
    if output.exists():
        if not args.force:
            raise FileExistsError(f"output already exists (use --force to replace): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copytree(checkpoint, output / "pretrained_model")
    (output / "manifest.json").write_text(json.dumps(build_manifest(args), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote LeRobot Franka Duo eval bundle: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
