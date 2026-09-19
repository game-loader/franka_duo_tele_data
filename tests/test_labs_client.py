from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from franka_duo_tele_data.action_spec import rot6d_to_matrix
from franka_duo_tele_data.labs_action_delta import DELTA_REPRESENTATION, absolute20_to_delta14
from franka_duo_tele_data.labs_client import LabsClient, command_for, parse_args, require_relay
from franka_duo_tele_data.labs_inference import (
    CAMERAS,
    CHUNK_REFERENCE,
    STATUS_SCHEMA,
    TOPICS,
    LabsContract,
    LabsObservationCache,
    Sample,
    reconstruct_chunk,
    validate_response,
)
from franka_duo_tele_data.labs_relay import admit_command, build_plan


@pytest.fixture
def contract():
    return LabsContract(Path(__file__).resolve().parents[1] / "configs/labs_fr3_31")


def state34(contract):
    return contract.state(dict.fromkeys(("left", "right"), [0, -0.4, 0, -1.8, 0, 1.5, 0]), [1, 0])


def response(rows=32):
    actions = np.zeros((rows, 14))
    actions[:, 0] = 0.01
    actions[:, 6] = -0.02
    actions[:, 3:6] = [0.03, -0.02, 0.04]
    actions[:, 12:] = [0.25, 0.75]
    return {
        "action_representation": DELTA_REPRESENTATION,
        "normalized": False,
        "chunk_reference": CHUNK_REFERENCE,
        "actions": actions.tolist(),
        "request_id": "req",
    }


def test_entire_chunk_uses_one_observation_never_cumsum(contract):
    state = state34(contract)
    targets = reconstruct_chunk(response(), state)
    assert targets.shape == (32, 20)
    np.testing.assert_allclose(targets[:, 0], state[0] + 0.01, atol=1e-7)
    np.testing.assert_allclose(targets[:, 9], state[9] - 0.02, atol=1e-7)
    expected = Rotation.from_rotvec([0.03, -0.02, 0.04]).as_matrix() @ rot6d_to_matrix(state[3:9])
    np.testing.assert_allclose(rot6d_to_matrix(targets[-1, 3:9]), expected, atol=2e-7)
    np.testing.assert_array_equal(targets[:, 18:], np.tile([0, 1], (32, 1)))
    np.testing.assert_allclose(
        absolute20_to_delta14(targets, np.tile(state, (32, 1))), validate_response(response()), atol=2e-7
    )


def test_binary_gripper_regression_overshoot_preserves_raw_reply():
    result = response()
    result["actions"][0][12:] = [1.054, -0.04]
    result["actions"][1][12:] = [0.4999, 0.5]
    canonical = validate_response(result)
    np.testing.assert_array_equal(canonical[:2, 12:], [[1, 0], [0, 1]])
    assert result["actions"][0][12:] == [1.054, -0.04]
    result["actions"][0][12] = 1.2
    with pytest.raises(ValueError, match="Gripper"):
        validate_response(result)


@pytest.mark.parametrize(
    "change",
    [
        {"normalized": True},
        {"normalized": 0},
        {"action_representation": "franka_duo_midpoint_delta14_v1"},
        {"actions": np.zeros((32, 20)).tolist()},
        {"actions": []},
        {"actions": np.full((32, 14), np.nan).tolist()},
        {"actions": np.full((32, 14), 2).tolist()},
    ],
)
def test_bad_response_rejected(change):
    with pytest.raises(ValueError):
        validate_response({**response(), **change})


def test_chunk_reference_and_extent_guards(contract):
    # Server temporal prose is retained as provenance; the user-confirmed client
    # convention remains request-observation-relative for the whole chunk.
    np.testing.assert_array_equal(
        reconstruct_chunk({**response(), "chunk_reference": "previous_action"}, state34(contract)),
        reconstruct_chunk(response(), state34(contract)),
    )
    result = response()
    result["actions"][-1][0] = 0.4
    with pytest.raises(ValueError, match="translation"):
        reconstruct_chunk(result, state34(contract))


def test_state_checks_fk_joint_bounds_and_gripper(contract):
    state = state34(contract)
    for index, value in [(0, 0.01), (18, 0.5), (23, 10)]:
        bad = state.copy()
        bad[index] += value
        with pytest.raises(ValueError):
            contract.validate_state(bad)


def test_both_gates_and_capture_only(tmp_path):
    for flags in (["--publish"], ["--enable-robot"], ["--capture-only", "--publish", "--enable-robot"]):
        with pytest.raises(SystemExit):
            parse_args(["--url", "ws://localhost/infer", *flags])
    assert not parse_args(["--capture-only"]).publish
    with pytest.raises(SystemExit):
        parse_args([])


def png():
    output = io.BytesIO()
    Image.fromarray(np.full((480, 640, 3), 64, dtype=np.uint8)).save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize(
    "profile,health_model", [("labs", None), ("c23", "FastWAM-FR3-C23"), ("c23", "wrong")]
)
def test_real_websocket_reuses_transport_state34_and_all_rows(contract, profile, health_model):
    import msgpack
    from aiohttp import web

    async def scenario():
        requests, connections = [], []

        async def serve(request):
            ws = web.WebSocketResponse(protocols=("fastwam.msgpack.v1",))
            await ws.prepare(request)
            connections.append(ws)
            async for message in ws:
                request = msgpack.unpackb(message.data, raw=False)
                requests.append(request)
                result = {**response(), "request_id": request["request_id"]}
                await ws.send_bytes(msgpack.packb(result, use_bin_type=True))
            return ws

        app = web.Application()
        app.router.add_get("/infer", serve)
        from franka_duo_tele_data.labs_action_delta import DELTA_NAMES
        from franka_duo_tele_data.labs_mcap_to_lerobot import STATE_NAMES

        async def health(_request):
            return web.json_response(
                {
                    "ready": True,
                    "model": health_model,
                    "variant": "c23",
                    "busy": False,
                    "normalized": False,
                    "action_representation": DELTA_REPRESENTATION,
                    "state_dim": 34,
                    "action_dim": 14,
                    "horizon": 32,
                    "action_rate_hz": 30,
                    "state_layout": STATE_NAMES,
                    "action_layout": DELTA_NAMES,
                    "task": contract.task,
                }
            )

        app.router.add_get("/health", health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            images = dict.fromkeys(CAMERAS, png())
            if profile == "c23" and health_model == "wrong":
                with pytest.raises(ValueError, match="C23 profile"):
                    async with LabsClient(f"ws://127.0.0.1:{port}/infer", contract, server_profile=profile):
                        pytest.fail("Wrong model accepted")
                assert not requests
                return
            async with LabsClient(f"ws://127.0.0.1:{port}/infer", contract, server_profile=profile) as client:
                for _ in range(2):
                    result = await client.infer(state34(contract), images)
                    assert reconstruct_chunk(result, state34(contract)).shape == (32, 20)
            assert len(connections) == 1
            assert requests[0]["task"] == contract.task
            np.testing.assert_array_equal(requests[0]["state"], state34(contract))
            assert requests[0]["images"] == images
            assert requests[0]["request_id"] != requests[1]["request_id"]
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_cache_matches_header_times_not_arrival_or_joint_array_order(contract):
    cache = LabsObservationCache(contract)
    state = state34(contract)
    stamp = time.time_ns() - 80_000_000
    names = [f"left_fr3_joint{i}" for i in range(7, 0, -1)]
    msg = SimpleNamespace(
        name=names,
        position=state[20:27][::-1],
        header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp // 10**9, nanosec=stamp % 10**9)),
    )
    cache.store("left_q", msg)
    np.testing.assert_array_equal(cache.buffers["left_q"][-1].value, state[20:27])
    mono = time.monotonic_ns()
    for key in TOPICS:
        if key != "left_q":
            cache.buffers[key].append(Sample(stamp, mono, key))
    cache.buffers["wrist_left"].append(Sample(stamp + 46_000_000, mono, "wrong"))
    selected = cache.select()
    assert selected["wrist_left"].value == "wrist_left"
    assert cache.select() is None  # Do not reuse an observation.
    cache.last_stamp = 0
    cache.buffers["head"][0].received_ns -= 300_000_000
    assert cache.select() is None  # Replayed recent header with stale receipt.


def plan_command(contract):
    initial = state34(contract)
    rows = []
    qs = []
    for i in range(1, 5):
        q = initial[20:].copy()
        q[[0, 7]] += 0.00005 * i
        qs.append(q)
        rows.append(contract.state({"left": q[:7], "right": q[7:]}, [0, 1])[:20])
    result = response(4)
    result["actions"] = absolute20_to_delta14(np.asarray(rows), np.tile(initial, (4, 1))).tolist()
    observation = SimpleNamespace(state=initial, stamp_ns=time.time_ns())
    command = command_for(result, observation, contract, speed=1.0)

    class Solver:
        def __init__(self, side):
            self.index = 0
            self.side = side

        def solve(self, pose, seed):
            q = qs[self.index][0:7] if self.side == "left" else qs[self.index][7:14]
            self.index += 1
            return {"success": True, "joint_positions": q}

    return command, {s: Solver(s) for s in ("left", "right")}, qs


def test_full_chunk_plan_endpoint_continuity_and_speed_limits(contract):
    command, solvers, qs = plan_command(contract)
    initial = state34(contract)[20:]
    admit_command(command, contract, initial)
    plan = build_plan(command, contract, solvers, initial)
    assert all(s.index == 4 for s in solvers.values())
    np.testing.assert_allclose(plan.sample(0)[0], initial)
    np.testing.assert_allclose(plan.sample(plan.duration)[0], qs[-1])
    assert plan.sample(plan.duration)[2]
    np.testing.assert_allclose(plan.joints[1:], qs)
    assert plan.duration >= sum(plan.durations)
    times = np.linspace(0, plan.duration, 10000)
    positions = np.asarray([plan.sample(t)[0] for t in times])
    velocity = np.diff(positions, axis=0) / (times[1] - times[0])
    acceleration = np.diff(velocity, axis=0) / (times[1] - times[0])
    assert np.max(np.abs(velocity)) <= 0.8001
    assert np.max(np.abs(acceleration)) <= 2.001


def test_last_row_ik_failure_rejects_whole_plan(contract):
    command, solvers, _ = plan_command(contract)
    original = solvers["right"].solve

    def fail_last(pose, seed):
        if solvers["right"].index == 3:
            return {"success": False, "reason": "test"}
        return original(pose, seed)

    solvers["right"].solve = fail_last
    with pytest.raises(ValueError, match="IK failed"):
        build_plan(command, contract, solvers, state34(contract)[20:])


def test_stale_moved_and_wrong_model_commands_rejected(contract):
    command, _, _ = plan_command(contract)
    q = state34(contract)[20:]
    for change in ({"created_ns": 0}, {"model_hashes": {}}, {"chunk_reference": "cumulative"}):
        with pytest.raises(ValueError):
            admit_command({**command, **change}, contract, q)
    with pytest.raises(ValueError, match="moved"):
        admit_command(command, contract, q + 0.03)


def test_relay_requires_ack_and_final_hold(contract):
    status = {
        "schema": STATUS_SCHEMA,
        "model_hashes": contract.model_hashes,
        "enabled": True,
        "ready": True,
        "fault": "",
        "phase": "holding",
        "command_id": "old",
    }
    assert not require_relay(status, contract, command_id="new")
    assert not require_relay({**status, "phase": "executing"}, contract, command_id="old")
    assert require_relay(status, contract, command_id="old")
    with pytest.raises(RuntimeError, match="fault"):
        require_relay({**status, "fault": "tracking"}, contract)


def test_return_exact_recorded_joints_without_ik_and_bounded_motion(contract):
    from franka_duo_tele_data.labs_client import start_command
    from franka_duo_tele_data.labs_relay import build_return_plan, return_at_goal

    target = state34(contract)
    initial = target[20:].astype(float).copy()
    initial[0] += 0.7
    initial[8] -= 0.4
    command = start_command(target, contract, 7)
    admit_command(command, contract, initial)
    plan = build_return_plan(command, contract, initial, np.zeros(14), target, 7)
    assert plan.preserve_grippers
    np.testing.assert_array_equal(plan.sample(plan.duration)[0], target[20:])
    distance = np.abs(target[20:] - initial)
    assert np.max(1.875 * distance / plan.duration) <= 0.25
    assert np.max(10 / np.sqrt(3) * distance / plan.duration**2) <= 0.5
    assert command["execution_rate_hz"] == 9
    assert plan.diagnostics["velocity_limit_rad_s"] == pytest.approx(0.075)
    assert plan.diagnostics["acceleration_limit_rad_s2"] == pytest.approx(0.045)
    assert not return_at_goal(plan, initial, np.zeros(14))
    assert not return_at_goal(plan, target[20:], np.ones(14) * 0.03)
    assert return_at_goal(plan, target[20:], np.zeros(14))
    with pytest.raises(ValueError, match="separate"):
        build_plan(command, contract, {}, initial)
    for change in ({"start_episode": 8}, {"start_identity": "wrong"}):
        with pytest.raises(ValueError, match="does not match"):
            build_return_plan({**command, **change}, contract, initial, np.zeros(14), target, 7)
    for velocity in ([], np.full(14, np.nan), np.full(14, 0.03)):
        with pytest.raises(ValueError, match="stationary"):
            build_return_plan(command, contract, initial, velocity, target, 7)


def test_return_time_scaling_preserves_path_and_rejects_invalid_or_legacy_commands(contract):
    from franka_duo_tele_data.labs_client import start_command
    from franka_duo_tele_data.labs_relay import build_return_plan

    target = state34(contract)
    initial = target[20:].astype(float).copy()
    initial[0] += 0.7
    fast_command = start_command(target, contract, 0, speed=1.0)
    slow_command = start_command(target, contract, 0)
    fast = build_return_plan(fast_command, contract, initial, np.zeros(14), target, 0)
    slow = build_return_plan(slow_command, contract, initial, np.zeros(14), target, 0)
    assert slow.duration == pytest.approx(fast.duration / 0.3)
    for fraction in np.linspace(0, 1, 101):
        q = slow.sample(fraction * slow.duration)[0]
        np.testing.assert_allclose(q, fast.sample(fraction * fast.duration)[0])
        assert np.all(q >= np.minimum(initial, target[20:]) - 1e-10)
        assert np.all(q <= np.maximum(initial, target[20:]) + 1e-10)
    assert slow.diagnostics["max_velocity_rad_s"] == pytest.approx(
        fast.diagnostics["max_velocity_rad_s"] * 0.3
    )
    assert slow.diagnostics["max_acceleration_rad_s2"] == pytest.approx(
        fast.diagnostics["max_acceleration_rad_s2"] * 0.3**2
    )
    for speed in (None, 0, -0.1, 1.1, np.nan, np.inf, True, "0.3"):
        with pytest.raises(ValueError, match="speed"):
            start_command(target, contract, 0, speed=speed)
        with pytest.raises(ValueError, match="speed"):
            build_return_plan({**slow_command, "speed": speed}, contract, initial, np.zeros(14), target, 0)
    with pytest.raises(ValueError, match="execution_rate_hz"):
        build_return_plan({**slow_command, "execution_rate_hz": 30}, contract, initial, np.zeros(14), target, 0)
    legacy = {**slow_command, "schema": "labs_fr3_episode_joint_return_v1"}
    with pytest.raises(ValueError, match="Incompatible"):
        admit_command(legacy, contract, initial)
    with pytest.raises(ValueError, match="does not match"):
        build_return_plan(legacy, contract, initial, np.zeros(14), target, 0)
    with pytest.raises(ValueError, match="more than 90"):
        build_return_plan(
            start_command(target, contract, 0, speed=0.01), contract, initial, np.zeros(14), target, 0
        )


def test_episode_start_requires_unique_frame_zero(tmp_path, contract):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from franka_duo_tele_data.labs_inference import episode_start

    data = tmp_path / "data"
    data.mkdir()
    start = state34(contract)
    other = contract.state(dict.fromkeys(("left", "right"), [0.3, -0.4, 0, -1.8, 0, 1.5, 0]), [1, 0])
    table = pa.table(
        {
            "episode_index": [2, 1, 2],
            "frame_index": [1, 0, 0],
            "observation.state": [other.tolist(), other.tolist(), start.tolist()],
        }
    )
    pq.write_table(table, data / "part.parquet")
    np.testing.assert_array_equal(episode_start(tmp_path, 2, contract), start)
    with pytest.raises(ValueError, match="found 0"):
        episode_start(tmp_path, 3, contract)
    pq.write_table(table, data / "duplicate.parquet")
    with pytest.raises(ValueError, match="found 2"):
        episode_start(tmp_path, 2, contract)


def test_startup_ack_failure_blocks_inference_and_dry_run_does_not_publish(contract):
    from franka_duo_tele_data.labs_client import return_before_inference, start_command

    command = start_command(state34(contract), contract, 0)

    async def scenario(fail=False, dry=False):
        events = []

        async def wait(command_id=None, **kwargs):
            if command_id is not None:
                if fail:
                    raise TimeoutError("not settled")
                events.append("settled")
            return {"completed_start_identity": command["start_identity"]}

        async def session():
            await return_before_inference(
                command,
                publish=None if dry else lambda _: events.append("return"),
                wait_hold=wait,
                record=lambda _: None,
            )
            for _ in range(3):
                events.append("infer")

        if fail:
            with pytest.raises(TimeoutError):
                await session()
            assert events == ["return"]
        else:
            await session()
            assert events == (["infer"] * 3 if dry else ["return", "settled"] + ["infer"] * 3)

    asyncio.run(scenario())
    asyncio.run(scenario(fail=True))
    asyncio.run(scenario(dry=True))


def test_policy_continuous_tracker_accepts_previously_rejected_step(contract):
    command, solvers, _ = plan_command(contract)
    command["speed"] = 1
    initial = state34(contract)[20:].copy()
    initial[0] -= 0.005
    plan = build_plan(command, contract, solvers, initial)
    assert plan.diagnostics["tracking"] == "continuous_ruckig_v2"
    assert plan.diagnostics["max_acceleration_rad_s2"] <= 2.000001
    assert plan.diagnostics["max_velocity_rad_s"] <= 0.800001
    assert plan.duration >= len(command["targets"]) / 30


def test_return_settle_requires_continuous_dwell():
    from franka_duo_tele_data.labs_relay import settle_update

    since, done = settle_update(None, True, 1, 0.5)
    assert not done
    since, done = settle_update(since, True, 1.49, 0.5)
    assert not done
    since, done = settle_update(since, False, 1.5, 0.5)
    assert since is None and not done
    since, done = settle_update(since, True, 2, 0.5)
    assert not done
    _, done = settle_update(since, True, 2.5, 0.5)
    assert done


def test_operation_flags_and_publication_gates(tmp_path):
    base = ["--dataset", str(tmp_path)]
    restore = parse_args([*base, "--restore"])
    assert restore.restore and not restore.infer and restore.url is None
    assert restore.speed == 0.3
    infer = parse_args([*base, "--infer", "--url", "ws://localhost/infer"])
    assert infer.infer and not infer.restore
    assert infer.speed == 0.3
    both = parse_args([*base, "--restore", "--infer", "--url", "ws://localhost/infer"])
    assert both.restore and both.infer
    for invalid in (
        ["--restore", "--publish"],
        ["--restore", "--enable-robot"],
        ["--restore", "--capture-only"],
        ["--infer"],
    ):
        with pytest.raises(SystemExit):
            parse_args([*base, *invalid])


@pytest.mark.parametrize("restore,infer", [(True, False), (False, True), (True, True)])
def test_run_operation_routing_without_robot_or_network(monkeypatch, tmp_path, contract, restore, infer):
    import json
    import sys
    from types import ModuleType

    from franka_duo_tele_data import labs_client as module

    calls = []
    measured = state34(contract)
    monkeypatch.setattr(module, "LabsContract", lambda *_: contract)

    def read_start(*_):
        calls.append("read_episode")
        return measured

    monkeypatch.setattr(module, "episode_start", read_start)
    monkeypatch.setattr(module, "launch_relay", lambda *_: pytest.fail("dry-run launched relay"))

    class Cache:
        def __init__(self, *_):
            pass

        def store(self, *_):
            pass

        def next(self, **_):
            return SimpleNamespace(state=measured, images={}, stamp_ns=time.time_ns(), source_stamps_ns={})

    class Client:
        def __init__(self, *_, **kwargs):
            self.health = {}

        async def __aenter__(self):
            calls.append("connect")
            return self

        async def __aexit__(self, *_):
            pass

        async def infer(self, *_):
            calls.append("infer")
            return response()

    monkeypatch.setattr(module, "LabsObservationCache", Cache)
    monkeypatch.setattr(module, "LabsClient", Client)
    monkeypatch.setattr(module, "encode_images", lambda _: {})
    node = SimpleNamespace(create_subscription=lambda *_: None, destroy_node=lambda: None)
    ros = ModuleType("rclpy")
    ros.init = ros.shutdown = lambda: None
    ros.spin = lambda _: None
    ros.create_node = lambda _: node
    qos = ModuleType("rclpy.qos")
    qos.qos_profile_sensor_data = object()
    sensor = ModuleType("sensor_msgs.msg")
    sensor.Image = sensor.JointState = object
    std = ModuleType("std_msgs.msg")
    std.String = object
    for name, value in [
        ("rclpy", ros),
        ("rclpy.qos", qos),
        ("sensor_msgs.msg", sensor),
        ("std_msgs.msg", std),
    ]:
        monkeypatch.setitem(sys.modules, name, value)
    output = tmp_path / "result"
    flags = (["--restore"] if restore else []) + (["--infer"] if infer else [])
    args = parse_args(
        [
            "--dataset",
            str(tmp_path),
            "--url",
            "ws://unused/infer",
            "--output",
            str(output),
            "--max-chunks",
            "2",
            "--manage-relay",
            *flags,
        ]
    )
    assert module.run(args) == 0
    events = [json.loads(line)["event"] for line in (output / "trace.jsonl").read_text().splitlines()]
    assert events.count("return_to_start") == int(restore)
    assert calls.count("read_episode") == int(restore)
    assert calls.count("connect") == int(infer)
    assert calls.count("infer") == (2 if infer else 0)
    assert "published" not in events and "completed" not in events


def test_activation_hold_is_fixed_and_preserves_grippers(contract):
    from franka_duo_tele_data.labs_relay import build_hold_plan, controller_set_ready

    q = state34(contract)[20:]
    plan = build_hold_plan(contract, q, np.zeros(14))
    assert plan.preserve_grippers
    for elapsed in (0, 0.5, 1, 100):
        np.testing.assert_array_equal(plan.sample(elapsed)[0], q)
    with pytest.raises(ValueError, match="stationary"):
        build_hold_plan(contract, q, np.ones(14))
    with pytest.raises(ValueError, match="in-bounds"):
        build_hold_plan(contract, q + 100, np.zeros(14))
    broadcasters = {"joint_state_broadcaster", "franka_robot_state_broadcaster"}
    assert controller_set_ready(broadcasters, allow_inactive=True)
    assert not controller_set_ready(broadcasters)
    assert controller_set_ready(broadcasters | {"joint_follower_controller"})
    assert not controller_set_ready(broadcasters | {"gravity_compensation_controller"}, allow_inactive=True)


def test_return_accepts_authorized_position_tolerance_but_requires_stationary(contract):
    from franka_duo_tele_data.labs_relay import build_hold_plan, return_at_goal

    q = state34(contract)[20:].astype(float)
    plan = build_hold_plan(contract, q, np.zeros(14))
    offset = np.zeros(14)
    offset[6] = 0.0499
    assert return_at_goal(plan, q + offset, np.zeros(14))
    assert not return_at_goal(plan, q + offset, np.full(14, 0.021))
    assert not return_at_goal(plan, q + offset, np.zeros(14), position_tolerance=0.01)
    offset[6] = 0.0501
    assert not return_at_goal(plan, q + offset, np.zeros(14))


def test_continuous_tracker_carries_derivatives_across_steps_and_brakes_at_end(contract):
    from franka_duo_tele_data.labs_tracking import track_chunk

    q = state34(contract)[20:].astype(float)
    qs = np.tile(q, (33, 1))
    qs[:, 0] += np.linspace(0, 0.18, 33)
    qs[:, 8] += 0.01 * np.sin(np.linspace(0, 3 * np.pi, 33))
    bounds = np.concatenate([contract.fk[s].bounds for s in ("left", "right")])
    plan = track_chunk(qs, np.ones((32, 2)), 1 / 30, bounds)
    assert plan.duration >= 32 / 30
    # Every retarget carries position, velocity AND acceleration forward.
    for left, right in zip(plan.segments[:-1], plan.segments[1:], strict=True):
        np.testing.assert_allclose(left.at_time(plan.period), right.at_time(0), atol=1e-10)
    velocities = [np.max(np.abs(plan.kinematics(i / 30)[1])) for i in range(2, 31)]
    assert min(velocities) > 0.01  # no artificial stop at any internal policy row
    dt = 0.0005
    samples = np.asarray([plan.kinematics(t) for t in np.arange(0, plan.duration + dt, dt)])
    assert np.max(np.abs(samples[:, 1])) <= 0.500001
    assert np.max(np.abs(samples[:, 2])) <= 1.000001
    assert np.max(np.abs(np.diff(samples[:, 2], axis=0) / dt)) <= 10.0001
    end_q, end_v, end_a = plan.kinematics(plan.duration)
    np.testing.assert_allclose(end_q, qs[-1], atol=1e-8)
    np.testing.assert_array_equal(end_v, np.zeros(14))
    np.testing.assert_array_equal(end_a, np.zeros(14))
    assert plan.sample(plan.duration)[2]
    assert not plan.sample(0.5)[2]


def test_zero_target_velocity_removes_reversals_on_monotonic_reference():
    from franka_duo_tele_data.labs_tracking import track_chunk

    joints = np.zeros((33, 14))
    joints[:, 0] = np.linspace(0, 0.3, 33)
    bounds = np.tile([-3.0, 3.0], (14, 1))
    plans = [
        track_chunk(
            joints, np.ones((32, 2)), 1 / 9, bounds,
            max_velocity=0.8, max_acceleration=2, max_jerk=20,
            max_reference_lag=0.3, target_velocity_weight=weight,
        )
        for weight in (0, 0.5, 1)
    ]
    travel = []
    for weight, plan in zip((0, 0.5, 1), plans, strict=True):
        assert plan.diagnostics["target_velocity_weight"] == weight
        samples = np.asarray([plan.kinematics(t)[0][0] for t in np.arange(0, plan.duration, 0.001)])
        travel.append(np.abs(np.diff(np.r_[samples, 0.3])).sum())
        np.testing.assert_array_equal(plan.joints, joints)
    assert travel[0] == pytest.approx(0.3, abs=1e-8)
    assert travel[2] > travel[0] + 0.005  # old tangent term induces reversals
    assert travel[0] < travel[1] < travel[2]


@pytest.mark.parametrize("weight", [-0.01, 1.01, float("nan"), float("inf")])
def test_tracker_rejects_invalid_target_velocity_weight(weight):
    from franka_duo_tele_data.labs_tracking import track_chunk

    with pytest.raises(ValueError, match="Invalid"):
        track_chunk(
            np.zeros((2, 14)), np.ones((1, 2)), 1 / 9,
            np.tile([-3.0, 3.0], (14, 1)), target_velocity_weight=weight,
        )


def test_continuous_tracker_rejects_excessive_lag_and_joint_bounds(contract):
    from franka_duo_tele_data.labs_tracking import track_chunk

    q = state34(contract)[20:].astype(float)
    bounds = np.concatenate([contract.fk[s].bounds for s in ("left", "right")])
    qs = np.tile(q, (33, 1))
    qs[1:, 0] += 0.3
    with pytest.raises(ValueError, match="lag"):
        track_chunk(qs, np.ones((32, 2)), 1 / 30, bounds)
    qs[2, 0] = bounds[0, 1] + 0.01
    with pytest.raises(ValueError, match="Invalid"):
        track_chunk(qs, np.ones((32, 2)), 1 / 30, bounds)


def test_policy_starts_from_last_command_without_reset_to_measured(contract):
    command, solvers, _ = plan_command(contract)
    measured = state34(contract)[20:].astype(float)
    held = measured.copy()
    held[0] += 0.025  # stationary impedance tracking error is not a new command
    plan = build_plan(command, contract, solvers, measured, commanded_start=held)
    np.testing.assert_array_equal(plan.sample(0)[0], held)


def test_c23_playback_uses_nine_hz_without_changing_source_contract(contract):
    state = state34(contract)
    observation = SimpleNamespace(state=state, stamp_ns=time.time_ns())
    slow = command_for(response(), observation, contract)
    normal = command_for(response(), observation, contract, speed=1.0)
    assert slow["action_rate_hz"] == 30
    assert slow["execution_rate_hz"] == 9
    np.testing.assert_array_equal(slow["targets"], normal["targets"])
    command, solvers, _ = plan_command(contract)
    command["speed"] = 0.3
    plan = build_plan(command, contract, solvers, state[20:])
    np.testing.assert_allclose(plan.durations, 1 / 9)
    assert plan.diagnostics["reference_duration_s"] == pytest.approx(len(command["targets"]) / 9)


def test_recorded_episode_uses_each_rows_state_and_actions(tmp_path, contract):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from franka_duo_tele_data.labs_episode import load_episode

    q = state34(contract)[20:].astype(float)
    states, targets = [], []
    for index in range(4):
        current = q.copy()
        current[0] += index * 0.015
        state = contract.state({"left": current[:7], "right": current[7:]}, [1, 0])
        target_q = current.copy()
        target_q[0] += 0.004
        target = contract.state({"left": target_q[:7], "right": target_q[7:]}, [0, 1])[:20]
        states.append(state)
        targets.append(target)
    actions = absolute20_to_delta14(np.asarray(targets), np.asarray(states))
    data = tmp_path / "data"
    data.mkdir()
    order = [2, 0, 3, 1]
    table = pa.table(
        {
            "episode_index": [0] * 4,
            "frame_index": order,
            "timestamp": [i / 30 for i in order],
            "observation.state": [states[i].tolist() for i in order],
            "action": [actions[i].tolist() for i in order],
        }
    )
    pq.write_table(table, data / "one.parquet")
    episode = load_episode(tmp_path, 0, contract)
    np.testing.assert_allclose(episode.targets, targets, atol=2e-7)
    # Replaying recorded measured poses would lose the actual action offset.
    assert not np.allclose(episode.targets[:, :18], episode.states[:, :18])
    with pytest.raises(ValueError, match="unique consecutive"):
        load_episode(tmp_path, 1, contract)
    pq.write_table(table, data / "duplicate.parquet")
    with pytest.raises(ValueError, match="unique consecutive"):
        load_episode(tmp_path, 0, contract)


def test_recorded_replay_has_separate_admission_and_initial_pose_gate(contract):
    from franka_duo_tele_data.labs_episode import REPLAY_SCHEMA, Episode
    from franka_duo_tele_data.labs_relay import build_episode_plan

    q = state34(contract)[20:]
    command = {
        "schema": REPLAY_SCHEMA,
        "command_id": "replay",
        "created_ns": time.time_ns(),
        "model_hashes": contract.model_hashes,
        "reference_joints": q.tolist(),
    }
    admit_command(command, contract, q)
    with pytest.raises(ValueError, match="moved"):
        admit_command(command, contract, q + 0.03)
    episode = Episode(
        0, np.array([state34(contract)]), np.zeros((1, 14)), np.array([state34(contract)[:20]]), "hash"
    )
    with pytest.raises(ValueError, match="Return to"):
        build_episode_plan(episode, contract, {}, q + 0.06, q)


@pytest.mark.parametrize("speed", [1.0, 0.3], ids=["explicit_30hz", "default_9hz"])
def test_policy_and_replay_share_site_dynamics(contract, speed):
    from franka_duo_tele_data.labs_episode import Episode
    from franka_duo_tele_data.labs_relay import build_episode_plan

    command, policy_solvers, _ = plan_command(contract)
    command["speed"] = speed
    state = state34(contract)
    held = state[20:].astype(float).copy()
    held[0] -= 0.005
    policy = build_plan(command, contract, policy_solvers, state[20:], commanded_start=held)
    states = np.tile(state, (len(command["targets"]), 1))
    targets = np.asarray(command["targets"])
    episode = Episode(0, states, absolute20_to_delta14(targets, states), targets, "test")
    _, replay_solvers, _ = plan_command(contract)
    replay = build_episode_plan(episode, contract, replay_solvers, state[20:], held, 30 * speed)

    assert policy.diagnostics == replay.diagnostics
    assert policy.diagnostics["velocity_limit_rad_s"] == 0.8
    assert policy.diagnostics["acceleration_limit_rad_s2"] == 2.0
    assert policy.diagnostics["max_jerk_rad_s3"] == 20.0
    assert policy.diagnostics["reference_lag_limit_rad"] == 0.3
    assert policy.diagnostics["target_velocity_weight"] == 0.0
    np.testing.assert_allclose(policy.durations, 1 / (30 * speed))
    for elapsed in np.linspace(0, policy.duration, 101):
        np.testing.assert_allclose(policy.kinematics(elapsed), replay.kinematics(elapsed))


def test_custom_tracker_dynamics_are_bounded(contract):
    from franka_duo_tele_data.labs_tracking import track_chunk

    q = state34(contract)[20:].astype(float)
    qs = np.tile(q, (33, 1))
    qs[:, 0] += np.linspace(0, 0.3, 33)
    bounds = np.concatenate([contract.fk[s].bounds for s in ("left", "right")])
    slower = track_chunk(qs, np.ones((32, 2)), 1 / 9, bounds)
    faster = track_chunk(
        qs,
        np.ones((32, 2)),
        1 / 9,
        bounds,
        max_reference_lag=0.3,
        max_velocity=0.8,
        max_acceleration=2,
        max_jerk=20,
    )
    assert slower.diagnostics["velocity_limit_rad_s"] == 0.5
    assert slower.diagnostics["reference_lag_limit_rad"] == 0.15
    assert faster.diagnostics["velocity_limit_rad_s"] == 0.8
    assert faster.diagnostics["max_velocity_rad_s"] <= 0.800001
    assert faster.diagnostics["max_acceleration_rad_s2"] <= 2.000001
    times = np.arange(0, faster.duration, 0.001)
    samples = np.asarray([faster.kinematics(t) for t in times])
    assert np.max(np.abs(np.diff(samples[:, 2], axis=0) / 0.001)) <= 20.0001
    assert np.all(samples[:, 0] >= bounds[:, 0]) and np.all(samples[:, 0] <= bounds[:, 1])
    invalid = qs.copy()
    invalid[1, 0] = bounds[0, 1] + 0.001
    with pytest.raises(ValueError, match="Invalid"):
        track_chunk(
            invalid,
            np.ones((32, 2)),
            1 / 9,
            bounds,
            max_reference_lag=0.3,
            max_velocity=0.8,
            max_acceleration=2,
            max_jerk=20,
        )
