import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from franka_duo_tele_data import return_to_start as returner
from franka_duo_tele_data.action_spec import FrankaDuoActionSpec, matrix_to_rot6d, rot6d_to_matrix
from franka_duo_tele_data.cartesian_chunk import pose_distance
from franka_duo_tele_data.joint_servo_client import ChunkPacer, ServoStatus


@pytest.fixture
def target_context():
    payload = json.loads((returner.ROOT / "configs/dataset_initial_state.json").read_text())
    spec = FrankaDuoActionSpec.from_manifest(
        {
            "action_spec": {"dimension": 20, "ee_dimension": 9, "ee_rotation": "rot6d_columns"},
            "coordinate_transforms": payload["coordinate_transforms"],
        }
    )
    contract = SimpleNamespace(
        fps=30,
        action_spec=spec,
        gripper=payload["gripper_calibration"],
        transforms={
            side: np.array(payload["coordinate_transforms"][f"T_newbase_from_{side}_arm_base"])
            for side in ("left", "right")
        },
    )
    config = {"workspace_min": [0.25, -0.30, -0.27], "workspace_max": [0.56, 0.50, 0.35]}
    return payload, contract, config


def test_saved_target_is_archived_first_state(target_context):
    payload, contract, config = target_context
    target, _, _ = returner.load_target(
        returner.ROOT / "configs/dataset_initial_state.json", contract, config
    )
    np.testing.assert_allclose(target[:3], [0.2867118418, 0.19295156, -0.0773802623])
    np.testing.assert_allclose(target[9:12], [0.3474262357, -0.2121326327, -0.0907858685])
    assert payload["archive_sha256"] == "e0db659ded0e1ccd2bf7afd2500716a555176b4589ff8057bc99c2bf8d8f2786"


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_field", "action"),
        ("episode_index", 1),
        ("frame_index", 1),
        ("timestamp", 0.1),
        ("fps", 10),
    ],
)
def test_wrong_source_rejected(tmp_path, target_context, key, value):
    payload, contract, config = target_context
    payload[key] = value
    path = tmp_path / "target.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        returner.load_target(path, contract, config)


@pytest.mark.parametrize("change", ["transform", "calibration", "workspace", "nonfinite"])
def test_incompatible_target_rejected(tmp_path, target_context, change):
    payload, contract, config = target_context
    if change == "transform":
        payload["coordinate_transforms"]["T_newbase_from_right_arm_base"][0][3] += 0.001
    elif change == "calibration":
        contract.gripper = {**contract.gripper, "closed_position": 1.0}
    elif change == "workspace":
        payload["state"][9] = 0.8
    else:
        payload["state"][0] = float("nan")
    path = tmp_path / "target.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        returner.load_target(path, contract, config)


@pytest.mark.parametrize("hz", [3, 9, 30])
def test_return_geometry_speed_and_gripper_retention(target_context, hz):
    payload, contract, _ = target_context
    target = np.array(payload["state"])
    state = target.copy()
    state[:3] += [0.12, -0.04, 0.03]
    state[9:12] += [-0.12, 0.03, 0.09]  # outside the policy workspace is allowed at start
    state[3:9] = matrix_to_rot6d(Rotation.from_rotvec([0.5, 0.1, 0.2]).as_matrix())
    state[18:] = [0.35, 0.72]
    rows = returner.plan_return(state, target, contract.action_spec, hz)
    np.testing.assert_allclose(rows[-1, :18], target[:18], atol=1e-7)
    np.testing.assert_allclose(rows[:, 18:], np.tile(state[18:], (len(rows), 1)))
    np.testing.assert_array_equal(target, payload["state"])  # caller's archived state stays intact
    for offset in (0, 9):
        xyz = rows[:, offset : offset + 3]
        assert np.all(xyz >= np.minimum(state[offset : offset + 3], target[offset : offset + 3]) - 1e-7)
        assert np.all(xyz <= np.maximum(state[offset : offset + 3], target[offset : offset + 3]) + 1e-7)
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        distance, rotation = pose_distance(a, b)
        assert max(distance) <= 0.003 + 1e-7
        assert max(rotation) <= 0.012 + 1e-7
        assert max(distance) * hz <= 0.025 + 1e-6
        # Trace/arccos on float32 matrices loses precision at small angles;
        # independently measure physical angular speed with quaternion deltas.
        for offset in (3, 12):
            first = Rotation.from_matrix(rot6d_to_matrix(a[offset : offset + 6]).astype(float))
            second = Rotation.from_matrix(rot6d_to_matrix(b[offset : offset + 6]).astype(float))
            assert (first.inv() * second).magnitude() * hz <= 0.10 + 1e-5


def test_gripper_conversion_is_continuous_and_validates_feedback(target_context):
    _, contract, _ = target_context
    assert returner.physical_opening(0.0, contract.gripper) == 1
    assert returner.physical_opening(0.8, contract.gripper) == 0
    assert returner.physical_opening(0.3, contract.gripper) == pytest.approx(0.625)
    for position in (float("nan"), float("inf"), -0.1, 0.9):
        with pytest.raises(ValueError):
            returner.physical_opening(position, contract.gripper)


@pytest.mark.parametrize("flag", ["--publish", "--enable-robot"])
def test_partial_gates_fail_before_ros_or_dataset(monkeypatch, flag):
    monkeypatch.setattr(returner, "run", lambda _: pytest.fail("must reject before run"))
    with pytest.raises(SystemExit) as exc:
        returner.main([flag])
    assert exc.value.code == 2


def test_default_dry_run_and_direct_run_gate(monkeypatch):
    args = returner.build_parser().parse_args([])
    assert not args.publish and not args.enable_robot
    args.publish = True
    monkeypatch.setattr(returner, "load_mapping", lambda _: pytest.fail("must reject before config"))
    with pytest.raises(ValueError, match="both"):
        returner.run(args)


def test_restarted_timeline_and_ack_reject_skipped_rows():
    status = ServoStatus(
        started=True,
        fault=False,
        fault_reason="",
        step=1234.25,
        holding=True,
        last_step=1100,
        playback_speed=0.3,
        tracking_error_rad=0.01,
        chunks=15,
    )
    base = returner.start_step(status)
    assert base == 1244
    assert returner.start_step(dataclasses.replace(status, started=False)) == 0
    pacer = ChunkPacer(32, 64)
    relative = dataclasses.replace(
        status, started=False, step=status.step - base, last_step=status.last_step - base
    )
    assert pacer.next_start(relative) == 0
    assert not pacer.finished(relative)
    ack = dataclasses.replace(status, chunks=16, last_chunk_start_step=base, last_step=base + 31)
    returner.check_ack(ack, 15, base, 32)
    for bad in (
        dataclasses.replace(ack, chunks=17),
        dataclasses.replace(ack, last_chunk_start_step=base + 1),
        dataclasses.replace(ack, last_step=base + 30),
    ):
        with pytest.raises(RuntimeError, match="acknowledgement"):
            returner.check_ack(bad, 15, base, 32)
    done = dataclasses.replace(relative, started=True, last_step=63, step=63)
    assert pacer.finished(done)
    assert not pacer.finished(dataclasses.replace(done, holding=False))


def test_shared_lock_blocks_run_and_releases_after_failure(tmp_path, target_context, monkeypatch):
    import fcntl

    payload, contract, config = target_context
    config.update(
        joint_servo_chunk_topic="/franka_duo/joint_servo/action_chunk",
        joint_servo_status_topic="/franka_duo/joint_servo/status",
    )
    target_file = Path(returner.ROOT / "configs/dataset_initial_state.json")
    monkeypatch.setattr(returner, "ROOT", tmp_path)
    monkeypatch.setattr(returner, "load_mapping", lambda _: config)
    monkeypatch.setattr(returner, "RGB20DContract", lambda _: contract)
    calls = []

    def fake_runtime(*args):
        calls.append(args[0].publish)
        raise RuntimeError("injected read failure")

    monkeypatch.setattr(returner, "run_ros", fake_runtime)
    args = returner.build_parser().parse_args(["--target", str(target_file)])
    lock_path = tmp_path / "log/servo_start/.smolvla_loop.lock"
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another policy"):
            returner.run(args)
    assert calls == []
    for _ in range(2):
        with pytest.raises(RuntimeError, match="injected"):
            returner.run(args)
    assert calls == [False, False]
