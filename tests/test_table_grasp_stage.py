"""ROS-free tests for stage pose loading and the impedance transfer chunk."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from franka_duo_tele_data.action_spec import FrankaDuoActionSpec, rot6d_to_matrix  # noqa: E402
from franka_duo_tele_data.config_io import load_mapping  # noqa: E402
from franka_duo_tele_data.grasp_cup_bowl import ARM_SLICE, DATASET_GRASP_ROT6D  # noqa: E402
from franka_duo_tele_data.table_grasp_stage import (  # noqa: E402
    STAGE_NAMES,
    ApproachLimits,
    StagePose,
    build_transfer_rows,
    load_stage_poses,
    prepare,
    require_pose,
)

REPO_POSES = "configs/grasp_stage_poses.json"


def pose_action(left_xyz, right_xyz) -> list[float]:
    return [
        *left_xyz,
        *DATASET_GRASP_ROT6D["left"],
        *right_xyz,
        *DATASET_GRASP_ROT6D["right"],
    ]


def document(**overrides) -> dict:
    poses = {
        name: {"action": None, "grippers": [1.0, 1.0]} for name in STAGE_NAMES
    }
    poses["camera_clear"]["action"] = pose_action((0.30, 0.34, 0.05), (0.30, -0.34, 0.05))
    poses["grasp_ready"]["action"] = pose_action((0.40, 0.20, -0.05), (0.40, -0.10, -0.05))
    value = {
        "schema": "franka_duo_grasp_stage_poses_v1",
        "spine_height_m": 0.468,
        "poses": poses,
        "approach": {"step_m": 0.01, "step_rad": 0.08, "settle_rows": 3},
    }
    value.update(overrides)
    return value


def write(tmp_path, value):
    path = tmp_path / "poses.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_repo_pose_file_has_a_taught_stow_and_optional_rest():
    """travel_stow is captured; the other two stay optional until taught."""
    poses, limits, spine_height_m = load_stage_poses(REPO_POSES)
    assert spine_height_m == 0.468
    assert set(poses) == set(STAGE_NAMES)
    assert limits.step_m > 0 and limits.step_rad > 0
    stow = require_pose(poses, "travel_stow")
    assert stow.action.shape == (18,)
    assert np.isfinite(stow.action).all()
    # Both arms must sit inside the workspace the contract enforces, or the
    # transfer chunk would be rejected at runtime.
    config = load_mapping(Path("configs/tmr_rgb20d.yaml"))
    low = np.asarray(config["workspace_min"], dtype=float)
    high = np.asarray(config["workspace_max"], dtype=float)
    for offset in (0, 9):
        xyz = stow.action[offset : offset + 3].astype(float)
        assert np.all(xyz >= low) and np.all(xyz <= high), f"stow xyz {xyz} outside workspace"
    for name in ("camera_clear", "grasp_ready"):
        assert poses[name] is None
        with pytest.raises(ValueError, match="has not been taught"):
            require_pose(poses, name)


def test_prepare_needs_only_the_stow_pose_for_a_stow_run(tmp_path):
    """Teaching travel_stow alone must be enough to stow the arms."""
    only_stow = document()
    only_stow["poses"]["travel_stow"]["action"] = pose_action((0.28, 0.30, 0.02), (0.28, -0.30, 0.02))
    only_stow["poses"]["camera_clear"]["action"] = None
    only_stow["poses"]["grasp_ready"]["action"] = None
    poses, _limits, _height = load_stage_poses(write(tmp_path, only_stow))
    assert isinstance(require_pose(poses, "travel_stow"), StagePose)
    assert poses["camera_clear"] is None and poses["grasp_ready"] is None


def test_load_rejects_wrong_schema_and_malformed_poses(tmp_path):
    with pytest.raises(ValueError, match="stage pose document"):
        load_stage_poses(write(tmp_path, document(schema="something_else")))

    bad = document()
    bad["poses"]["camera_clear"]["action"] = [0.0] * 17
    with pytest.raises(ValueError, match="finite 18D"):
        load_stage_poses(write(tmp_path, bad))

    bad = document()
    bad["poses"]["camera_clear"]["action"] = [float("nan")] * 18
    with pytest.raises(ValueError, match="finite 18D"):
        load_stage_poses(write(tmp_path, bad))

    bad = document()
    bad["poses"]["grasp_ready"]["grippers"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="grippers"):
        load_stage_poses(write(tmp_path, bad))

    bad = document(approach={"step_m": 0.0})
    with pytest.raises(ValueError, match="approach limits"):
        load_stage_poses(write(tmp_path, bad))


def test_load_keeps_taught_poses_and_returns_untaught_as_none(tmp_path):
    poses, _limits, _height = load_stage_poses(write(tmp_path, document()))
    assert poses["travel_stow"] is None
    assert isinstance(require_pose(poses, "camera_clear"), StagePose)
    assert require_pose(poses, "grasp_ready").grippers == (1.0, 1.0)


def test_prepare_validates_before_touching_ros(tmp_path):
    """Config, pose and gate errors must surface without a ROS installation."""
    poses_path = write(tmp_path, document())

    def args(**overrides):
        values = {
            "stage_poses": poses_path,
            "pose_only": None,
            "spine_target_m": 0.468,
            "spine_velocity": 0.05,
            "publish": False,
            "enable_robot": False,
            "config": "configs/tmr_rgb20d.yaml",
            "calibration": "outputs/zed_pnp/zed_pnp_calibration.json",
            "arm": "right",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    # Both robot gates are required together.
    with pytest.raises(ValueError, match="both --publish and --enable-robot"):
        prepare(args(publish=True))

    # A stage height that disagrees with the taught poses is refused.
    with pytest.raises(ValueError, match="taught at spine"):
        prepare(args(spine_target_m=0.700))

    # An untaught posture is refused when it is the one being asked for.
    with pytest.raises(ValueError, match="has not been taught"):
        prepare(args(pose_only="travel_stow"))


def state_at(left_xyz, right_xyz) -> np.ndarray:
    state = np.zeros(20, np.float32)
    state[0:3] = left_xyz
    state[3:9] = DATASET_GRASP_ROT6D["left"]
    state[9:12] = right_xyz
    state[12:18] = DATASET_GRASP_ROT6D["right"]
    state[18:] = 1.0
    return state


def assert_reaches(row: np.ndarray, pose: StagePose) -> None:
    """Endpoint check.

    Positions must land exactly.  Rotations are compared as a rotation angle at
    1e-3 rad: the stored rot6d constants are rounded to four decimals and are
    not exactly orthonormal, so interpolation renormalizes them and returns
    values up to ~5e-5 away componentwise.
    """
    for offset in (0, 9):
        assert np.allclose(row[offset : offset + 3], pose.action[offset : offset + 3], atol=1e-6)
        delta = (
            rot6d_to_matrix(row[offset + 3 : offset + 9]).T
            @ rot6d_to_matrix(pose.action[offset + 3 : offset + 9])
        )
        assert math.acos(min(1.0, (np.trace(delta) - 1) / 2)) <= 1e-3


def test_transfer_rows_move_both_arms_within_step_limits():
    state = state_at((0.30, 0.30, 0.00), (0.30, -0.30, 0.00))
    pose = StagePose(
        "grasp_ready",
        np.asarray(pose_action((0.45, 0.18, -0.08), (0.42, -0.12, -0.06)), np.float32),
        (1.0, 0.0),
    )
    limits = ApproachLimits(step_m=0.01, step_rad=0.08, settle_rows=4)
    rows = build_transfer_rows(state, pose, limits)

    assert rows.shape[1] == 20
    spec = FrankaDuoActionSpec()
    previous = state
    for row in rows:
        spec.validate(row)
        for offset in (0, 9):
            assert np.linalg.norm(row[offset : offset + 3] - previous[offset : offset + 3]) <= 0.0101
            delta = (
                rot6d_to_matrix(previous[offset + 3 : offset + 9]).T
                @ rot6d_to_matrix(row[offset + 3 : offset + 9])
            )
            assert math.acos(min(1.0, (np.trace(delta) - 1) / 2)) <= 0.081
        previous = row

    # Both arms land on the commanded pose, and the grippers follow it.
    assert_reaches(rows[-1], pose)
    assert rows[-1, 18] == 1.0 and rows[-1, 19] == 0.0
    # The settle rows hold the endpoint rather than drifting past it.
    for row in rows[-limits.settle_rows :]:
        assert_reaches(row, pose)


def test_transfer_rows_hold_the_shorter_arm_while_the_other_finishes():
    """A far arm and a near arm must arrive together, not leave a jump behind."""
    state = state_at((0.30, 0.30, 0.00), (0.40, -0.10, 0.00))
    pose = StagePose(
        "camera_clear",
        np.asarray(pose_action((0.50, 0.30, 0.00), (0.41, -0.10, 0.00)), np.float32),
        (1.0, 1.0),
    )
    rows = build_transfer_rows(state, pose, ApproachLimits(settle_rows=0))
    # Left travels 0.20 m at 0.01 m per row; right only 0.01 m.
    assert len(rows) == 20
    right = rows[:, ARM_SLICE["right"]][:, :3]
    # The right arm reaches its target early and then holds it.
    assert np.allclose(right[0], (0.41, -0.10, 0.00), atol=1e-6)
    assert np.allclose(right[-1], (0.41, -0.10, 0.00), atol=1e-6)
    assert_reaches(rows[-1], pose)


def test_transfer_rows_to_the_current_pose_are_a_hold():
    state = state_at((0.35, 0.25, -0.05), (0.35, -0.15, -0.05))
    pose = StagePose("camera_clear", state[:18].copy(), (1.0, 1.0))
    rows = build_transfer_rows(state, pose, ApproachLimits(settle_rows=2))
    assert len(rows) >= 1
    for row in rows:
        assert_reaches(row, pose)
