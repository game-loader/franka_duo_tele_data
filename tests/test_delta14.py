"""Protocol, frame composition and ROS-free stream execution checks."""

from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import test_rgb20d_runtime

from franka_duo_tele_data import delta14_stream, smolvla_stream
from franka_duo_tele_data.action_spec import matrix_to_rot6d, rot6d_to_matrix
from franka_duo_tele_data.delta14_client import (
    ACTION_REPRESENTATION,
    Delta14Client,
    reconstruct_absolute20,
    validate_response,
)
from franka_duo_tele_data.joint_servo_client import ServoStatus

contract = test_rgb20d_runtime.contract
Rotation = pytest.importorskip("scipy.spatial.transform").Rotation


def state20():
    state = np.zeros(20)
    state[:3] = [0.4, 0.1, 0.1]
    state[9:12] = [0.4, -0.1, 0.1]
    state[3:9] = state[12:18] = matrix_to_rot6d(Rotation.from_euler("x", 45, degrees=True).as_matrix())
    return state


def response():
    actions = np.zeros((32, 14))
    actions[:, 0] = 0.001
    actions[:, 7] = -0.002
    actions[:, 12:] = [0.25, 0.75]
    return {
        "actions": actions.tolist(),
        "normalized": False,
        "action_representation": ACTION_REPRESENTATION,
        "request_id": "test",
    }


def test_accumulation_uses_base_axes_and_absolute_grippers():
    state = state20()
    before = state.copy()
    result = response()
    absolute = reconstruct_absolute20(result, state)
    np.testing.assert_allclose(absolute[-1, :3], state[:3] + [0.032, 0, 0])
    np.testing.assert_allclose(absolute[-1, 9:12], state[9:12] + [0, -0.064, 0])
    np.testing.assert_allclose(absolute[:, 18:], np.tile([0.25, 0.75], (32, 1)))
    np.testing.assert_array_equal(state, before)
    # An expired prefix still contributes to later absolute targets.
    np.testing.assert_allclose(absolute[10, 0], state[0] + 0.011)
    other = state.copy()
    other[0] += 0.1
    assert reconstruct_absolute20(result, other)[0, 0] == pytest.approx(other[0] + 0.001)


def test_noncommuting_rotations_are_left_multiplied_without_sign_flip():
    state = state20()
    result = response()
    result["actions"][0][3:6] = [0, 0, 0.1]
    result["actions"][1][3:6] = [0, 0.2, 0]
    result["actions"][0][9:12] = [-0.1, 0, 0]
    absolute = reconstruct_absolute20(result, state)
    initial = rot6d_to_matrix(state[3:9])
    rz = Rotation.from_rotvec([0, 0, 0.1]).as_matrix()
    ry = Rotation.from_rotvec([0, 0.2, 0]).as_matrix()
    np.testing.assert_allclose(rot6d_to_matrix(absolute[1, 3:9]), ry @ rz @ initial, atol=1e-6)
    assert not np.allclose(rot6d_to_matrix(absolute[1, 3:9]), initial @ rz @ ry)
    np.testing.assert_allclose(
        rot6d_to_matrix(absolute[0, 12:18]),
        Rotation.from_rotvec([-0.1, 0, 0]).as_matrix() @ initial,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("normalized", True),
        ("normalized", 0),
        ("normalized", None),
        ("action_representation", "absolute20"),
        ("actions", np.zeros((32, 20)).tolist()),
        ("actions", np.zeros((31, 14)).tolist()),
        ("actions", np.full((32, 14), np.nan).tolist()),
        ("actions", np.full((32, 14), np.inf).tolist()),
    ],
)
def test_response_contract_rejected(field, value):
    with pytest.raises(ValueError):
        validate_response({**response(), field: value})


def test_reject_out_of_range_gripper():
    result = response()
    result["actions"][0][12] = 1.01
    with pytest.raises(ValueError, match="openness"):
        validate_response(result)


def stream_args(tmp_path):
    return smolvla_stream.parse_stream_args(
        smolvla_stream.build_parser(),
        [
            "--dataset",
            str(tmp_path),
            "--output",
            str(tmp_path / "trace.jsonl"),
            "--max-chunks",
            "1",
        ],
    )


def config():
    return {
        "workspace_min": [0.25, -0.3, -0.27],
        "workspace_max": [0.56, 0.5, 0.35],
        "max_input_age_ms": 200,
        "max_target_step_m": 0.04,
        "max_target_step_rad": 0.35,
        "topics": {
            key: "/" + key
            for key in (
                "head",
                "wrist_left",
                "wrist_right",
                "left_pose",
                "right_pose",
                "left_gripper",
                "right_gripper",
            )
        },
        "joint_servo_chunk_topic": "/franka_duo/joint_servo/action_chunk",
        "joint_servo_status_topic": "/franka_duo/joint_servo/status",
        "trace_topic": "/franka_duo/eval/rgb20d_trace",
    }


def test_motion_guards_and_continuous_openness(contract, tmp_path):
    from dataclasses import replace

    contract.action_spec = replace(
        contract.action_spec, workspace_min=(0.25, -0.3, -0.27), workspace_max=(0.56, 0.5, 0.35)
    )
    args = stream_args(tmp_path)
    good, _ = delta14_stream.prepare_delta14_chunk(response(), contract, state20(), args, config())
    np.testing.assert_allclose(good[0, 18:], [0.25, 0.75])
    for column, value, error in ((0, 0.05, "translation"), (3, 2 * np.pi, "rotation")):
        bad = response()
        bad["actions"][0][column] = value
        with pytest.raises(ValueError, match=error):
            delta14_stream.prepare_delta14_chunk(bad, contract, state20(), args, config())
    outside = response()
    for row in outside["actions"]:
        row[0] = 0.01
    with pytest.raises(ValueError, match="workspace"):
        delta14_stream.prepare_delta14_chunk(outside, contract, state20(), args, config())


def test_websocket_persistent_binary_protocol():
    pytest.importorskip("aiohttp")
    pytest.importorskip("msgpack")
    import msgpack
    from aiohttp import web

    async def scenario():
        requests = []
        connections = []

        async def serve(request):
            ws = web.WebSocketResponse(protocols=("smolvla.msgpack.v1",))
            await ws.prepare(request)
            connections.append(ws)
            async for message in ws:
                value = msgpack.unpackb(message.data, raw=False)
                requests.append(value)
                result = {**response(), "request_id": value["request_id"]}
                await ws.send_bytes(msgpack.packb(result, use_bin_type=True))
            return ws

        app = web.Application()
        app.router.add_get("/infer", serve)
        runner = web.AppRunner(app)
        await runner.setup()
        server = web.TCPSite(runner, "127.0.0.1", 0)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]
        images = dict.fromkeys(("head", "wrist_left", "wrist_right"), b"unchanged-image-bytes")
        try:
            async with Delta14Client(f"ws://127.0.0.1:{port}/infer") as client:
                for _ in range(3):
                    assert len((await client.infer(state20().tolist(), images))["actions"]) == 32
                with pytest.raises(ValueError, match="supports only"):
                    await client.infer(state20().tolist(), images, "unsupported")
            assert len(connections) == 1
            assert len({r["request_id"] for r in requests}) == 3
            assert requests[0]["images"] == images
            np.testing.assert_allclose(requests[0]["state"], state20())
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["request_id", "timeout", "server_error", "representation"])
def test_client_rejects_failed_or_incompatible_exchange(failure):
    pytest.importorskip("aiohttp")
    msgpack = pytest.importorskip("msgpack")
    from aiohttp import WSMsgType

    class Socket:
        closed = False

        async def send_bytes(self, data):
            self.request = msgpack.unpackb(data, raw=False)

        async def receive(self):
            if failure == "timeout":
                await asyncio.sleep(1)
            result = {**response(), "request_id": self.request["request_id"]}
            if failure == "request_id":
                result["request_id"] = "wrong"
            elif failure == "server_error":
                result["error"] = "busy"
            elif failure == "representation":
                result["action_representation"] = "absolute20"
            return SimpleNamespace(type=WSMsgType.BINARY, data=msgpack.packb(result))

        async def close(self):
            self.closed = True

    async def scenario():
        client = Delta14Client(timeout=0.01)
        client.ws = Socket()
        images = dict.fromkeys(("head", "wrist_left", "wrist_right"), b"image")
        with pytest.raises((RuntimeError, ValueError, TimeoutError)):
            await client.infer(state20().tolist(), images)
        if failure in ("request_id", "timeout"):
            assert client.ws.closed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "publish,rejection", [(False, None), (True, None), (True, "motion"), (True, "left_hold")]
)
def test_stream_converts_before_publishing_without_mcap(
    monkeypatch, contract, tmp_path, publish, rejection
):
    import time

    from franka_duo_tele_data import replay_rgb20d

    args = stream_args(tmp_path)
    args.publish = args.enable_robot = publish
    args.max_chunks = 2 if publish else 1
    messages = {}
    observations = []
    completion_polls = []
    current = SimpleNamespace(
        started=False,
        holding=True,
        step=0,
        last_step=0,
        fault=False,
        fault_reason="",
        playback_speed=args.speed,
        action_rate_hz=30,
        commit_lead_steps=0,
        blend_steps=4,
        blend_mode="quintic_hold_v1",
        chunks=0,
        last_chunk_start_step=0,
        tracking_error_rad=0,
    )
    callback = None

    class Message:
        def __init__(self, **kwargs):
            self.layout = SimpleNamespace()
            self.__dict__.update(kwargs)

    class Publisher:
        def __init__(self, topic):
            self.topic = topic

        def get_subscription_count(self):
            return 1

        def publish(self, msg):
            messages.setdefault(self.topic, []).append(msg)
            if self.topic == config()["joint_servo_chunk_topic"]:
                current.chunks += 1
                current.last_step = msg.layout.data_offset + 31
                current.last_chunk_start_step = msg.layout.data_offset
                current.step = msg.layout.data_offset
                current.started = True
                current.holding = False
                callback(Message(data="status"))

    class Node:
        def create_subscription(self, msg_type, topic, cb, qos):
            nonlocal callback
            if topic == config()["joint_servo_status_topic"]:
                callback = cb
                callback(Message(data="status"))

        def create_publisher(self, msg_type, topic, qos):
            assert topic == config()["joint_servo_chunk_topic"]
            return Publisher(topic)

        def count_publishers(self, topic):
            return 1

        def destroy_node(self):
            pass

    for name, attrs in {
        "rclpy": {
            "init": lambda: None,
            "create_node": lambda _: Node(),
            "spin": lambda _: None,
            "ok": lambda: True,
            "shutdown": lambda: None,
        },
        "rclpy.qos": {"qos_profile_sensor_data": None},
        "geometry_msgs.msg": {"PoseStamped": Message},
        "sensor_msgs.msg": {"Image": Message, "JointState": Message},
        "std_msgs.msg": {"Float32MultiArray": Message, "MultiArrayDimension": Message, "String": Message},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    class Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def infer(self, state, images, task):
            np.testing.assert_array_equal(state, state20())
            assert current.holding
            if current.started:
                # The wall-clock servo timeline keeps advancing during a slow
                # request. A sequential chunk must still send all 32 rows.
                current.step += 50
                callback(Message(data="status"))
            if rejection == "motion":
                bad = response()
                bad["actions"][0][0] = 1
                return bad
            if rejection == "left_hold":
                current.started = True
                current.step = 30
                current.holding = False
                callback(Message(data="status"))
            return response()

    observation = SimpleNamespace(state=state20(), images={}, stamp_ns=time.time_ns(), source_stamps_ns={})
    stale_frame_given = False

    def next_observation(**kw):
        nonlocal stale_frame_given
        assert current.holding, "must not observe/request while the preceding chunk is moving"
        if current.started:
            assert current.step >= current.last_step
        observations.append(current.chunks)
        observation.stamp_ns = time.time_ns()
        if current.chunks == 1 and not stale_frame_given:
            # A delayed frame from before hold must be discarded before infer.
            observation.stamp_ns -= 100_000_000
            stale_frame_given = True
        return observation

    async def advance_servo(_):
        assert current.started and not current.holding
        completion_polls.append(current.chunks)
        # Reaching the last step without holding must still block inference.
        if current.step < current.last_step:
            current.step = current.last_step
        else:
            current.holding = True
        callback(Message(data="status"))

    monkeypatch.setattr(
        smolvla_stream, "RGB20DReader", lambda *a, **kw: SimpleNamespace(next=next_observation)
    )
    monkeypatch.setattr(smolvla_stream.asyncio, "sleep", advance_servo)
    # The final chunk is waited on outside asyncio.run().
    def finish_servo(_):
        current.step = current.last_step
        current.holding = True
        callback(Message(data="status"))

    monkeypatch.setattr(smolvla_stream.time, "sleep", finish_servo)
    monkeypatch.setattr(smolvla_stream, "RGB20DContract", lambda _: contract)
    inference_config = config()
    del inference_config["trace_topic"]
    monkeypatch.setattr(smolvla_stream, "load_mapping", lambda _: inference_config)
    monkeypatch.setattr(delta14_stream, "load_mapping", lambda _: inference_config)
    monkeypatch.setattr(delta14_stream, "Delta14Client", Client)
    monkeypatch.setattr(smolvla_stream, "encode_images", lambda *a: {})
    monkeypatch.setattr(smolvla_stream, "parse_status", lambda _: ServoStatus(**vars(current)))
    monkeypatch.setattr(replay_rgb20d, "check_joint_servo_controllers", lambda *a: None)

    def refuse_recording(*a, **kw):
        pytest.fail("inference must not start an MCAP recorder or recording relay")

    monkeypatch.setattr(replay_rgb20d, "optional_recorder", refuse_recording)

    def run():
        return delta14_stream.run(args)

    if rejection is not None:
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            run()
        assert config()["joint_servo_chunk_topic"] not in messages
        records = [json.loads(line) for line in args.output.read_text().splitlines()]
        assert any(r["event"] == "response" for r in records)
        assert records[-1]["event"] == "error"
        return
    assert run() == 0
    records = [json.loads(line) for line in args.output.read_text().splitlines()]
    raw = next(r for r in records if r["event"] == "response")
    assert np.asarray(raw["raw_actions"]).shape == (32, 14)
    event = next(r for r in records if r["event"] == ("publish" if publish else "dry_run"))
    assert np.asarray(event["actions"]).shape == (32, 20)
    np.testing.assert_allclose(np.asarray(event["actions"])[0, 18:], [0.25, 0.75])
    topic = config()["joint_servo_chunk_topic"]
    assert len(messages.get(topic, [])) == (args.max_chunks if publish else 0)
    if publish:
        actual = np.asarray(messages[topic][0].data).reshape(32, 20)
        expected = np.stack([contract.action_spec.to_link0_action(row) for row in event["actions"]])
        np.testing.assert_allclose(actual, expected)
        assert completion_polls == [1, 1]
        assert observations == [0, 1, 1, 2]
        published = messages[topic]
        assert published[1].layout.data_offset >= published[0].layout.data_offset + 31 + 50 + 9
        assert all(np.asarray(msg.data).reshape(-1, 20).shape == (32, 20) for msg in published)
        assert all(r["skipped_expired_rows"] == 0 for r in records if r["event"] == "accepted")
    assert config()["trace_topic"] not in messages


@pytest.mark.parametrize("flags", [["--publish"], ["--enable-robot"]])
def test_cli_requires_both_robot_gates(flags):
    with pytest.raises(SystemExit):
        delta14_stream.main(["--dataset", "unused", *flags])


def test_delta14_cli_has_no_recording_option():
    with pytest.raises(SystemExit) as exc:
        delta14_stream.main(["--dataset", "unused", "--record-mcap"])
    assert exc.value.code == 2


def test_original_absolute20_path_still_thresholds_grippers(contract, tmp_path):
    args = stream_args(tmp_path)
    original = np.tile(state20(), (32, 1))
    original[:, 18:] = [0.25, 0.75]
    actions, _ = smolvla_stream.prepare_absolute_chunk(
        {"actions": original.tolist()},
        contract,
        state20(),
        args,
        config(),
    )
    np.testing.assert_array_equal(actions[:, 18:], np.tile([0, 1], (32, 1)))
