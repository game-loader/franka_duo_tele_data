import asyncio
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
from PIL import Image

from franka_duo_tele_data import labs_joint_inference as jp
from franka_duo_tele_data.labs_action_recording import ActionRecording, load_recording
from franka_duo_tele_data.labs_action_viewer import write_viewer
from franka_duo_tele_data.labs_client import LabsClient, command_for, model_input_state, parse_args
from franka_duo_tele_data.labs_inference import CAMERAS, LabsContract, episode_start, start_identity
from franka_duo_tele_data.labs_relay import admit_command


@pytest.fixture
def contract():
    return LabsContract(Path(__file__).resolve().parents[1] / "configs/labs_fr3_31")


def observation(contract):
    state = contract.state(
        {"left": [0.1, -0.4, 0, -1.8, 0, 1.5, 0.2], "right": [-0.1, -0.5, 0.1, -1.7, 0.1, 1.6, -0.2]}, [1, 0]
    )
    return SimpleNamespace(state=state, stamp_ns=time.time_ns())


def info(contract):
    return {
        "protocol": "smolvla.msgpack.v1",
        "state_dim": 16,
        "action_dim": 16,
        "chunk_size": 32,
        "prediction_horizon": 50,
        "n_action_steps": 32,
        "state_names": jp.JOINT_NAMES,
        "action_names": jp.JOINT_NAMES,
        "action_contract": jp.SCHEMA,
        "state_input_normalized": False,
        "action_normalized": False,
        "image_shape_hwc": [480, 640, 3],
        "cameras": {k: f"observation.images.{k}" for k in CAMERAS},
        "default_task": contract.task,
    }


def reply(contract):
    actions = np.tile(jp.model_state(observation(contract).state), (32, 1))
    actions[:, 0] += np.arange(1, 33) * 0.0001
    actions[:, 8] -= np.arange(1, 33) * 0.0002
    actions[:, 7] = 0.25
    actions[:, 15] = 0.75
    return {
        "request_id": "r",
        "actions": actions.tolist(),
        "action": actions[0].tolist(),
        "action_normalized": False,
        "chunk_size": 32,
        "prediction_horizon": 50,
    }


def fastwam_health(contract):
    return {
        "ready": True, "busy": False, "normalized": False,
        "model": "FastWAM-FR3-Joint16", "variant": "joint16",
        "state_dim": 16, "action_dim": 16, "horizon": 32, "action_rate_hz": 30,
        "state_layout": jp.JOINT_NAMES, "action_layout": jp.JOINT_NAMES,
        "action_representation": jp.FASTWAM_REPRESENTATION,
        "joint_units": "radians", "task": contract.task,
    }


def fastwam_reply(contract):
    result = reply(contract)
    result.pop("action_normalized")
    result.pop("prediction_horizon")
    return {**result, "normalized": False, "action_representation": jp.FASTWAM_REPRESENTATION}


@pytest.mark.parametrize("change", [
    {"state_dim": 34}, {"action_dim": 14}, {"horizon": 16}, {"joint_units": "degrees"},
    {"state_layout": list(reversed(jp.JOINT_NAMES))}, {"action_layout": []},
    {"action_representation": "delta14"}, {"normalized": 0}, {"ready": False}, {"busy": True},
])
def test_fastwam_joint16_rejects_wrong_health(contract, change):
    with pytest.raises(ValueError, match="FastWAM joint16 health"):
        jp.validate_fastwam_health({**fastwam_health(contract), **change}, contract.task)


@pytest.mark.parametrize("change", [
    {"normalized": True}, {"normalized": 0}, {"action_normalized": True},
    {"action_representation": "delta14"}, {"actions": [[0] * 14] * 32},
    {"actions": [[0] * 16] * 16}, {"actions": [[float("nan")] * 16] * 32},
    {"horizon": 16}, {"action_dim": 14}, {"action_contract": "wrong"},
])
def test_fastwam_joint16_rejects_wrong_response(contract, change):
    with pytest.raises(ValueError):
        jp.adapt_fastwam_response({**fastwam_reply(contract), **change}, fastwam_health(contract))


@pytest.mark.parametrize("profile", ["joint16", "fastwam_joint16"])
def test_joint_order_absolute_no_accumulation_and_continuous_plan(contract, profile):
    obs = observation(contract)
    projected = model_input_state(obs.state, profile)
    np.testing.assert_array_equal(projected, np.r_[obs.state[20:27], 1, obs.state[27:34], 0])
    raw = reply(contract)
    result = (jp.adapt_fastwam_response(fastwam_reply(contract), fastwam_health(contract))
              if profile == "fastwam_joint16" else jp.validate_response(raw, info(contract)))
    command = command_for(result, obs, contract, server_profile=profile)
    assert command["schema"] == jp.COMMAND_SCHEMA
    assert command["chunk_integration"] == jp.INTEGRATION
    assert raw["actions"][0][7] == 0.25
    np.testing.assert_array_equal(
        np.array(command["targets"])[:, jp.JOINT_INDICES], np.array(raw["actions"])[:, jp.JOINT_INDICES]
    )
    admit_command(command, contract, obs.state[20:])
    plan = jp.build_plan(command, contract, obs.state[20:])
    np.testing.assert_array_equal(plan.joints[1:, :7], np.array(raw["actions"])[:, :7])
    np.testing.assert_array_equal(plan.joints[1:, 7:], np.array(raw["actions"])[:, 8:15])
    np.testing.assert_array_equal(plan.grippers, np.tile([0, 1], (32, 1)))
    np.testing.assert_array_equal(plan.sample(plan.duration)[0], plan.joints[-1])
    assert plan.diagnostics["velocity_limit_rad_s"] == 0.8
    assert plan.diagnostics["target_velocity_weight"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"action_normalized": True},
        {"action_normalized": 0},
        {"normalized": True},
        {"actions": [[0] * 14] * 32},
        {"actions": [[0] * 16] * 4},
        {"actions": [[float("nan")] * 16] * 32},
        {"chunk_size": 4},
        {"action_contract": "delta14"},
        {"action": [0] * 16},
    ],
)
def test_reject_bad_response(contract, change):
    with pytest.raises(ValueError):
        jp.validate_response({**reply(contract), **change}, info(contract))


@pytest.mark.parametrize(
    "change",
    [
        {"state_dim": 20},
        {"action_dim": 14},
        {"chunk_size": 4},
        {"action_contract": "delta14"},
        {"action_normalized": True},
        {"state_names": list(reversed(jp.JOINT_NAMES))},
    ],
)
def test_reject_bad_metadata(contract, change):
    with pytest.raises(ValueError, match="Joint16 info"):
        jp.validate_info({**info(contract), **change}, "smolvla.msgpack.v1", contract.task)


def test_joint_bounds_jump_and_gripper_rejected(contract):
    obs = observation(contract)
    command = command_for(
        jp.validate_response(reply(contract), info(contract)), obs, contract, server_profile="joint16"
    )
    bad = np.array(command["targets"])
    bad[-1, 0] += 0.3
    with pytest.raises(ValueError, match="jump"):
        jp.build_plan({**command, "targets": bad.tolist()}, contract, obs.state[20:])
    bad = np.array(command["targets"])
    bad[-1, 7] = 0.3
    with pytest.raises(ValueError, match="binary"):
        jp.build_plan({**command, "targets": bad.tolist()}, contract, obs.state[20:])
    # A small final step can still cross a physical joint bound.
    initial = obs.state[20:].copy()
    initial[0] = contract.fk["left"].bounds[0][1] - 0.01
    bad = np.array(command["targets"])
    bad[:, 0] = initial[0] + 0.02
    with pytest.raises(ValueError, match="tracking input"):
        jp.build_plan({**command, "targets": bad.tolist()}, contract, initial)
    with pytest.raises(ValueError, match="absolute joint16"):
        admit_command({**command, "chunk_integration": "cumulative"}, contract, obs.state[20:])


def test_joint16_dataset_restore_and_archive(contract, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    state = observation(contract).state
    features = {k: {"shape": [16], "names": jp.JOINT_NAMES} for k in ["observation.state", "action"]}
    features.update({f"observation.images.{k}": {"shape": [480, 640, 3]} for k in CAMERAS})
    (root / "meta/info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "fps": 30, "features": features})
    )
    (root / "meta/conversion_manifest.json").write_text(
        json.dumps(
            {
                "schema": jp.SCHEMA,
                "joint_units": "rad",
                "joint_representation": "absolute joint position",
                "normalized": False,
                "task": contract.task,
            }
        )
    )
    pq.write_table(
        pa.table({"observation.state": [jp.model_state(state)], "episode_index": [0], "frame_index": [0]}),
        root / "data/e.parquet",
    )
    loaded = LabsContract(contract.config, root)
    start = episode_start(root, 0, loaded)
    np.testing.assert_allclose(start, state, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(start[18:], state[18:])
    assert start_identity(start, 0, loaded) == start_identity(state, 0, contract)
    folder = tmp_path / "record"
    folder.mkdir()
    recorder = ActionRecording(
        folder,
        {
            "execution_rate_hz": 9,
            "urdf": {s: (contract.config / f"{s}.urdf").read_text() for s in ["left", "right"]},
        },
    )
    raw = reply(contract)
    result = jp.validate_response(raw, info(contract))
    command = command_for(result, observation(contract), contract, server_profile="joint16")
    recorder.event(
        {
            "event": "observation",
            "index": 0,
            "state": state.tolist(),
            "model_input_state": jp.model_state(state),
        }
    )
    recorder.event({"event": "wire_response", "response": raw})
    recorder.event({"event": "inference", "response": result, "command": command})
    joints, _ = jp.split_targets(command["targets"])
    recorder.event(
        {
            "event": "tracking_sample",
            "measured_joints": joints[-1].tolist(),
            "status": {
                "command_id": command["command_id"],
                "phase": "holding",
                "holding_target": joints[-1].tolist(),
            },
        }
    )
    recorder.event({"event": "completed", "command_id": command["command_id"]})
    bundle = load_recording(recorder.close("normal"))
    c = bundle["chunks"][0]
    assert c["raw_response"] == raw
    assert c["summary"]["model_endpoint_error"]["left"]["position_m"] == 0
    np.testing.assert_array_equal(c["summary"]["model_endpoint_joint_error_rad"], np.zeros(14))
    write_viewer(bundle, folder / "replay.html")
    assert (folder / "replay.html").exists()


@pytest.mark.parametrize("profile", ["joint16", "fastwam_joint16"])
def test_joint16_websocket_raw_state_and_32_absolute_targets(contract, profile):
    from aiohttp import web

    async def scenario():
        received = []
        fastwam = profile == "fastwam_joint16"
        protocol = "fastwam.msgpack.v1" if fastwam else "smolvla.msgpack.v1"
        response = fastwam_reply(contract) if fastwam else reply(contract)

        async def metadata(_request):
            return web.json_response(fastwam_health(contract) if fastwam else info(contract))

        async def infer(request):
            ws = web.WebSocketResponse(protocols=(protocol,))
            await ws.prepare(request)
            async for m in ws:
                r = msgpack.unpackb(m.data, raw=False)
                received.append(r)
                await ws.send_bytes(
                    msgpack.packb({**response, "request_id": r["request_id"]}, use_bin_type=True)
                )
            return ws

        app = web.Application()
        app.router.add_get("/health" if fastwam else "/info", metadata)
        app.router.add_get("/infer", infer)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            data = io.BytesIO()
            Image.new("RGB", (640, 480)).save(data, format="PNG")
            obs = observation(contract)
            async with LabsClient(
                f"ws://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/infer",
                contract,
                server_profile=profile,
            ) as client:
                raw = []
                client.response_sink = raw.append
                result = await client.infer(obs.state, dict.fromkeys(CAMERAS, data.getvalue()))
                cmd = command_for(result, obs, contract, server_profile=profile)
                assert len(cmd["targets"]) == 32
                assert cmd["schema"] == jp.COMMAND_SCHEMA
                assert cmd["chunk_integration"] == jp.INTEGRATION
                assert result["action"] == result["actions"][0]
                assert raw[0]["actions"][0][7] == 0.25
                if fastwam:
                    assert raw[0]["action_representation"] == jp.FASTWAM_REPRESENTATION
                np.testing.assert_array_equal(
                    np.array(cmd["targets"])[:, jp.JOINT_INDICES],
                    np.array(raw[0]["actions"])[:, jp.JOINT_INDICES],
                )
            assert received[0]["state"] == jp.model_state(obs.state)
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_joint16_requires_both_robot_gates():
    for flags in [["--publish"], ["--enable-robot"]]:
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--server-profile",
                    "joint16",
                    "--infer",
                    "--url",
                    "ws://unused/infer",
                    "--dataset",
                    "unused",
                    *flags,
                ]
            )
