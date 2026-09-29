"""Labs WebSocket client with explicit physical action contracts. Dry-run by default."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from . import (
    labs_action_contract as physical_contract,
    labs_fastwam_eef as eef_policy,
    labs_joint_inference as joint_policy,
)
from .labs_action_delta import DELTA_NAMES
from .labs_action_recording import ActionRecording
from .labs_inference import (
    ABSOLUTE_INTEGRATION,
    ABSOLUTE_REPRESENTATION,
    CAMERAS,
    CHUNK_INTEGRATION,
    CHUNK_REFERENCE,
    COMMAND_SCHEMA,
    COMMAND_TOPIC,
    DELTA_REPRESENTATION,
    FR3_REPRESENTATION,
    GRIPPER_POSTPROCESS,
    RETURN_SCHEMA,
    STATUS_SCHEMA,
    STATUS_TOPIC,
    TOPICS,
    LabsContract,
    LabsObservationCache,
    absolute_targets,
    episode_start,
    reconstruct_chunk,
    start_identity,
    validate_response,
)
from .labs_mcap_to_lerobot import STATE_NAMES
from .labs_task_starts import load_task_starts
from .smolvla_client import InferenceConnectionError, SmolVLAClient

JOINT_PROFILES = ("joint16", "fastwam_joint16", "fastwam_joint16_next", "absolute_joint16")
EEF_PROFILES = ("smolvla", "fastwam_eef20", "absolute20")


def model_input_state(state, server_profile):
    """Project a full control observation to the selected model's raw input."""
    value = np.asarray(state, dtype=np.float32)
    if value.shape != (34,) or not np.isfinite(value).all():
        raise ValueError("Model projection requires a finite internal state34")
    if server_profile in JOINT_PROFILES:
        return joint_policy.model_state(value)
    return value[:20].tolist() if server_profile in EEF_PROFILES else value.tolist()


def integration_for(profile):
    if profile in JOINT_PROFILES:
        return joint_policy.INTEGRATION
    return ABSOLUTE_INTEGRATION if profile in EEF_PROFILES else CHUNK_INTEGRATION


class LabsClient(SmolVLAClient):
    """Reuse binary transport with the selected Labs model's wire contract."""

    action_representation = DELTA_REPRESENTATION
    subprotocol = "fastwam.msgpack.v1"

    def __init__(self, url, contract, timeout=10, *, server_profile="labs", joint16_protocol="smolvla.msgpack.v1", wire_protocol="fastwam.msgpack.v1", task_id=None):
        if server_profile not in ("labs", "c23", "smolvla", "joint16", "fastwam_joint16", "fastwam_eef20", "fastwam_joint16_next", "absolute20", "absolute_joint16"):
            raise ValueError("Unknown Labs server profile")
        self.server_profile = server_profile
        if server_profile == "smolvla":
            self.subprotocol = "smolvla.msgpack.v1"
        if server_profile == "joint16":
            self.subprotocol = joint16_protocol
        if server_profile in physical_contract.PROFILES:
            self.subprotocol = wire_protocol
        if task_id is not None and (
            type(task_id) is not int or task_id not in (1, 2, 3, 4)
            or server_profile != "absolute20" or self.subprotocol != "fastwam.msgpack.v1"
        ):
            raise ValueError("task_id requires an integer 1..4 and FastWAM absolute20")
        self.task_id = task_id
        super().__init__(url, timeout=timeout)
        self.contract = contract
        self.health = None
        self.response_sink = None

    async def __aenter__(self):
        await super().__aenter__()
        try:
            if self.ws.protocol != self.subprotocol:
                raise ValueError(f"Server must negotiate {self.subprotocol}")
            if self.server_profile in physical_contract.PROFILES:
                self.health = {"client_action_contract": self.server_profile,
                               "metadata_check": "not_requested", "protocol": self.subprotocol}
                return self
            url = urlsplit(self.url)
            health_url = urlunsplit(
                (
                    "https" if url.scheme == "wss" else "http", url.netloc,
                    "/info" if self.server_profile in ("smolvla", "joint16") else "/health", "", "",
                )
            )
            deadline = time.monotonic() + self.timeout
            while True:
                async with self.session.get(health_url, timeout=self.timeout) as response:
                    response.raise_for_status()
                    self.health = await response.json()
                # Registry loading/another request may briefly mark the server busy.
                # Wait before inference, while preserving the actual health values.
                wait_for_registry = (
                    self.server_profile == "fastwam_joint16_next"
                    and self.health.get("model") == "FastWAM-FR3-PolicyRegistry"
                    and self.health.get("task") == self.contract.task
                    and (self.health.get("ready") is not True or self.health.get("busy") is True)
                    and time.monotonic() < deadline
                )
                if not wait_for_registry:
                    break
                await asyncio.sleep(1)
            if self.server_profile == "smolvla":
                self.validate_smolvla_info()
                return self
            if self.server_profile == "joint16":
                joint_policy.validate_info(self.health, self.subprotocol, self.contract.task)
                return self
            if self.server_profile == "fastwam_eef20":
                eef_policy.validate_health(self.health, self.contract.task)
                return self
            if self.server_profile == "fastwam_joint16_next":
                joint_policy.validate_next_health(self.health, self.contract.task)
                return self
            if self.server_profile == "fastwam_joint16":
                joint_policy.validate_fastwam_health(self.health, self.contract.task)
                return self
            expected = {
                "state_dim": 34,
                "action_dim": 14,
                "horizon": 32,
                "action_rate_hz": 30,
                "state_layout": STATE_NAMES,
                "action_layout": DELTA_NAMES,
            }
            if (
                self.health.get("ready") is not True
                or self.health.get("busy") is not False
                or self.health.get("normalized") is not False
                or self.health.get("action_representation") not in (DELTA_REPRESENTATION, FR3_REPRESENTATION)
                or any(self.health.get(k) != v for k, v in expected.items())
                or self.health.get("task") != self.contract.task
            ):
                raise ValueError("FastWAM health does not match the Labs state34/action14 contract")
            if self.server_profile == "c23" and (
                self.health.get("model") != "FastWAM-FR3-C23" or self.health.get("variant") != "c23"
            ):
                raise ValueError("C23 profile requires model=FastWAM-FR3-C23 and variant=c23")
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    def validate_smolvla_info(self):
        expected = {
            "policy_type": "smolvla",
            "protocol": self.subprotocol,
            "state_dim": 20,
            "configured_state_dim": 20,
            "state_input_contract": "dual_link8_pose_rot6d_columns_and_grippers20_v1",
            "action_dim": 20,
            "chunk_size": 32,
            "prediction_horizon": 50,
            "state_names": STATE_NAMES[:20],
            "action_names": STATE_NAMES[:20],
            "action_contract": ABSOLUTE_REPRESENTATION,
            "image_shape_hwc": [480, 640, 3],
            "cameras": {name: f"observation.images.{name}" for name in CAMERAS},
            "default_task": self.contract.task,
        }
        if (
            any(self.health.get(k) != v for k, v in expected.items())
            or self.health.get("state_input_normalized") is not False
            or self.health.get("action_normalized") is not False
            or self.health.get("absolute_action") is not True
            or self.health.get("is_recorded_command_action") is not False
        ):
            raise ValueError("SmolVLA info does not match the raw state20/absolute future measured pose20 contract")

    def adapt_smolvla_response(self, result):
        # Preserve the wire reply before adding aliases for the shared planner.
        # Inverse normalization is server-owned; only rotations/grippers are canonicalized.
        if result.get("action_normalized") is not False:
            raise ValueError("SmolVLA must explicitly return action_normalized=false")
        if "normalized" in result and result["normalized"] is not False:
            raise ValueError("Conflicting SmolVLA normalization declarations")
        for key in ("action_contract", "action_representation"):
            if key in result and result[key] != self.health["action_contract"]:
                raise ValueError("SmolVLA response contract disagrees with info")
        actions = np.asarray(result.get("actions"), dtype=float)
        rows = self.health["chunk_size"]
        if actions.shape != (rows, 20):
            raise ValueError(f"Expected SmolVLA actions[{rows},20] matching info.chunk_size")
        for key in ("chunk_size", "prediction_horizon"):
            if key in result and result[key] != self.health[key]:
                raise ValueError(f"SmolVLA response {key} disagrees with info")
        if "action" in result and not np.array_equal(np.asarray(result["action"]), actions[0]):
            raise ValueError("SmolVLA action must equal actions[0]")
        if not np.isfinite(actions).all():
            raise ValueError("Nonfinite SmolVLA actions")
        targets = absolute_targets(
            {**result, "normalized": False, "action_representation": ABSOLUTE_REPRESENTATION}, rows=rows
        )
        adapted = {
            **result, "normalized": False, "action_representation": self.health["action_contract"],
            "actions": targets.tolist(),
            "client_gripper_postprocess": "binary_open_ge_0.5_v1",
            "client_rotation_postprocess": "gram_schmidt_columns_v1",
            "is_recorded_command_action": False,
        }
        if "action" in result:
            adapted["action"] = targets[0].tolist()
        return adapted

    def build_request(self, state, images, task, request_id):
        payload = super().build_request(state, images, task, request_id)
        if self.task_id is not None:
            del payload["task"]
            payload["task_id"] = self.task_id
        # Format hints are a FastWAM wire extension. The SmolVLA endpoint
        # accepts only state/images/task/request_id and rejects extra keys.
        if self.server_profile == "absolute20" and self.subprotocol == "fastwam.msgpack.v1":
            payload.update(state_format="rot6d_cols20", action_format="absolute20")
        if self.server_profile == "fastwam_eef20":
            payload.update(eef_policy.REQUEST_FIELDS)
        if self.server_profile == "fastwam_joint16_next":
            payload["policy"] = joint_policy.NEXT_POLICY
        return payload

    async def infer(self, state, images, task=None):
        from PIL import Image

        if self.task_id is not None and task is not None:
            raise ValueError("Provide task_id or task text, not both")
        state = self.contract.validate_state(state)
        if set(images) != set(CAMERAS):
            raise ValueError("Provide head, wrist_left and wrist_right images")
        for data in images.values():
            with Image.open(io.BytesIO(data)) as image:
                if image.mode != "RGB" or image.size != (640, 480):
                    raise ValueError("Labs wire images must be 640x480 RGB PNG/JPEG")
                image.verify()
        result = await super().infer(model_input_state(state, self.server_profile), images, task or self.contract.task)
        if self.response_sink is not None:
            self.response_sink(result)
        if self.server_profile in physical_contract.PROFILES:
            return physical_contract.adapt_response(result, self.server_profile)
        if self.server_profile == "fastwam_eef20":
            return eef_policy.adapt_response(result, self.health)
        if self.server_profile == "fastwam_joint16_next":
            return joint_policy.adapt_next_response(result, self.health)
        if self.server_profile == "fastwam_joint16":
            return joint_policy.adapt_fastwam_response(result, self.health)
        if self.server_profile == "joint16":
            return joint_policy.validate_response(result, self.health)
        if self.server_profile == "smolvla":
            return self.adapt_smolvla_response(result)
        actions = validate_response(result)
        representation = self.health["action_representation"]
        rows = self.health["horizon"]
        if result["action_representation"] != representation or len(actions) != rows:
            raise ValueError("Response representation/horizon disagrees with server metadata")
        return result


def encode_images(images):
    from PIL import Image

    result = {}
    for name, pixels in images.items():
        out = io.BytesIO()
        Image.fromarray(pixels).save(out, format="PNG", compress_level=1)
        result[name] = out.getvalue()
    return result


def command_for(result, observation, contract, *, speed=0.3, server_profile="labs", execute_steps=None):
    if not np.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError("Require 0 < speed <= 1")
    joint_mode = server_profile in JOINT_PROFILES
    if joint_mode:
        joint_policy.split_targets(result["actions"])
    if server_profile in EEF_PROFILES:
        targets = absolute_targets(result, rows=len(result["actions"]) if server_profile == "absolute20" else 32).tolist()
    else:
        targets = result["actions"] if joint_mode else reconstruct_chunk(result, observation.state).tolist()
    if execute_steps is not None:
        if isinstance(execute_steps, bool) or not isinstance(execute_steps, int) or not 1 <= execute_steps <= len(targets):
            raise ValueError(f"execute_steps must be between 1 and the returned chunk length ({len(targets)})")
        # Preserve full raw/validated replies; only the published target prefix is shortened.
        targets = targets[:execute_steps]
    return {
        "schema": joint_policy.COMMAND_SCHEMA if joint_mode else COMMAND_SCHEMA,
        "mode": "policy_chunk",
        "command_id": uuid.uuid4().hex,
        "request_id": result["request_id"],
        "chunk_reference": CHUNK_REFERENCE,
        "chunk_integration": integration_for(server_profile),
        "gripper_postprocess": GRIPPER_POSTPROCESS,
        "model_hashes": contract.model_hashes,
        "created_ns": time.time_ns(),
        "observation_ns": observation.stamp_ns,
        "reference_state": observation.state.tolist(),
        "targets": targets,
        "prediction_rows": len(result["actions"]),
        "execution_rows": len(targets),
        "action_rate_hz": 30,
        "speed": speed,
        "execution_rate_hz": 30 * speed,
    }


def start_command(start_state, contract, episode, *, speed=0.3, task_id=None):
    # The relay reads current joints itself. No camera/FK/IK observation is
    # needed to execute this separate, measured-joint positioning operation.
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0 < speed <= 1:
        raise ValueError("Return requires 0 < speed <= 1")
    if task_id is not None and (type(task_id) is not int or task_id not in (1, 2, 3, 4) or episode != 0):
        raise ValueError("Task return requires task_id 1..4 and episode 0")
    return {
        "schema": RETURN_SCHEMA,
        "mode": "return_to_start",
        "command_id": uuid.uuid4().hex,
        "created_ns": time.time_ns(),
        "model_hashes": contract.model_hashes,
        "start_episode": episode,
        "start_identity": start_identity(start_state, episode, contract),
        **({"task_id": task_id} if task_id is not None else {}),
        "speed": speed,
        "execution_rate_hz": 30 * speed,
    }


async def return_before_inference(command, *, publish, wait_hold, record):
    """Once per session; a failed return prevents all inference requests."""
    record({"event": "return_to_start", "command": command, "published": False})
    if publish is None:
        return
    ready = await wait_hold()
    if command.get("task_id") is not None and (
        ready.get("task_start_identities", {}).get(str(command["task_id"])) != command["start_identity"]
    ):
        raise RuntimeError("Relay has not loaded the selected task start; reload the station relay first")
    publish(command)
    record({"event": "published", "command_id": command["command_id"]})
    status = await wait_hold(command["command_id"], timeout=120)
    if status.get("completed_start_identity") != command["start_identity"]:
        raise RuntimeError("Relay did not verify the selected episode start")
    record({"event": "completed", "command_id": command["command_id"], "status": status})


def require_relay(status, contract, *, command_id=None):
    if status.get("schema") != STATUS_SCHEMA or status.get("model_hashes") != contract.model_hashes:
        raise ValueError("Incompatible Labs site relay/model")
    if status.get("fault"):
        raise RuntimeError(f"Labs relay fault: {status['fault']}")
    if status.get("enabled") is not True:
        raise RuntimeError("Labs relay robot output is disabled")
    if command_id is not None and status.get("command_id") != command_id:
        return False
    return status.get("ready") is True and status.get("phase") == "holding"


def restore_preview(args, contract, start_state, *, dataset=None):
    """Offline target inspection; do not require ROS, cameras, server or IK."""
    args.output.mkdir(parents=True, exist_ok=False)
    command = start_command(start_state, contract, args.start_episode, speed=args.speed, task_id=args.task_id)
    record = {
        "event": "return_to_start",
        "command": command,
        "published": False,
        "dataset": str(dataset or args.dataset),
        "target_joints": start_state[20:].tolist(),
    }
    (args.output / "trace.jsonl").write_text(json.dumps(record, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "restore": "dry-run",
                "episode": args.start_episode,
                "task_id": args.task_id,
                "dataset": str(dataset or args.dataset),
                "target_joints": start_state[20:].tolist(),
                "output": str(args.output),
            }
        )
    )
    return 0


def launch_relay(args):
    """Keep the site relay alive between client runs to hold its last target."""
    directory = Path("outputs/labs_relay")
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        "flock",
        "--nonblock",
        "--conflict-exit-code",
        "75",
        str(directory / "process.lock"),
        sys.executable,
        "-m",
        "franka_duo_tele_data.labs_relay",
        "--config",
        str(args.config),
        "--dataset",
        str(args.dataset),
        "--start-episode",
        str(args.start_episode),
        "--publish",
        "--enable-robot",
        "--hold-current-on-start",
    ]
    if args.ik:
        command.extend(["--ik", str(args.ik)])
    with (directory / "relay.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    (directory / "launcher.pid").write_text(str(process.pid) + "\n")
    print(f"Labs relay started; log: {directory / 'relay.log'}; remains running to hold the endpoint.")
    return process


def run(args):
    if args.publish != args.enable_robot or args.capture_only and args.publish:
        raise ValueError("Robot publication requires both gates and cannot be capture-only")
    contract = LabsContract(args.config, args.dataset)
    task_metadata = ({"task_id": args.task_id} if args.task_id is not None
                     else {"task": args.task or contract.task})
    restore_dataset = args.dataset
    start_state = None
    if args.restore:
        if args.task_id is not None:
            task_start = load_task_starts(args.config, task_id=args.task_id)[args.task_id]
            restore_dataset, start_state = task_start.dataset, task_start.state
        else:
            start_state = episode_start(args.dataset, args.start_episode, contract)
    if args.restore and not args.infer and not args.publish:
        return restore_preview(args, contract, start_state, dataset=restore_dataset)
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import String

    cache = LabsObservationCache(contract)
    # Let asyncio handle Ctrl+C while ROS keeps collecting the accepted chunk's tail.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("labs_policy_client")
    status_lock = threading.Lock()
    latest = {"value": None, "received": 0.0, "error": None}
    thread = None
    recording = None
    try:
        for key, topic in TOPICS.items() if args.infer or args.capture_only else []:
            node.create_subscription(
                Image if key in CAMERAS else JointState,
                topic,
                lambda message, k=key: cache.store(k, message),
                qos_profile_sensor_data,
            )

        def on_status(message):
            value = None
            with status_lock:
                try:
                    value = json.loads(message.data)
                    latest.update(value=value, received=time.monotonic())
                except Exception as exc:
                    latest["error"] = exc
            if value is not None and recording is not None and args.infer:
                try:
                    recording.status(value, cache)
                except Exception as exc:
                    with status_lock:
                        latest["error"] = exc

        publisher = None
        if args.publish:
            node.create_subscription(String, STATUS_TOPIC, on_status, 10)
            publisher = node.create_publisher(String, COMMAND_TOPIC, 1)
        thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        thread.start()

        def relay_status():
            if node.count_publishers(COMMAND_TOPIC) != 1 or publisher.get_subscription_count() != 1:
                raise RuntimeError("Expected one client publisher and one Labs relay subscriber")
            with status_lock:
                if latest["error"]:
                    raise RuntimeError("Invalid relay status") from latest["error"]
                if latest["value"] is None or time.monotonic() - latest["received"] > 0.5:
                    raise TimeoutError("Labs relay status missing or stale")
                return latest["value"].copy()

        async def wait_hold(command_id=None, timeout=10):
            reason = "no completion acknowledgment"
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                value = relay_status()
                if require_relay(value, contract, command_id=command_id):
                    if args.server_profile in JOINT_PROFILES and joint_policy.COMMAND_SCHEMA not in value.get("supported_command_schemas", []):
                        raise RuntimeError("Relay needs joint16 upgrade/restart before publication")
                    return value
                reason = value.get("reason") or value.get("phase")
                await asyncio.sleep(0.02)
            raise TimeoutError(f"Labs relay not ready/completed: {reason}")

        args.output.mkdir(parents=True, exist_ok=False)
        recording = ActionRecording(args.output, {
            "url": args.url, "server_profile": args.server_profile, "speed": args.speed,
            "execute_steps": args.execute_steps,
            "execution_rate_hz": 30 * args.speed, "model_hashes": contract.model_hashes,
            "urdf": {side: (args.config / f"{side}.urdf").read_text() for side in ("left", "right")},
            "dataset": str(args.dataset), **task_metadata,
            "restore_dataset": str(restore_dataset) if args.restore else None,
            "chunk_reference": CHUNK_REFERENCE,
            "chunk_integration": integration_for(args.server_profile),
            "model_state_dim": {"absolute20": 20, "absolute_joint16": 16, "smolvla": 20, "fastwam_eef20": 20, "joint16": 16, "fastwam_joint16": 16, "fastwam_joint16_next": 16}.get(args.server_profile, 34),
        })
        print(f"Action recording: {args.output / 'actions.msgpack'}", flush=True)
        with (args.output / "trace.jsonl").open("x") as log:

            def record(value):
                log.write(json.dumps(value, allow_nan=False) + "\n")
                log.flush()
                recording.event(value)

            async def session():
                if publisher is not None:
                    # Allow DDS discovery only before the first command.
                    await asyncio.sleep(1)
                    if args.manage_relay and publisher.get_subscription_count() == 0:
                        process = launch_relay(args)
                        deadline = time.monotonic() + 10
                        while time.monotonic() < deadline:
                            if process.poll() not in (None, 75):
                                raise RuntimeError("Relay exited; inspect outputs/labs_relay/relay.log")
                            with status_lock:
                                received = latest["value"] is not None
                            if received and publisher.get_subscription_count() == 1:
                                break
                            await asyncio.sleep(0.1)
                    await wait_hold()
                after_ns = time.time_ns()
                client = None
                active_command = None
                try:
                    if args.infer:
                        client = await LabsClient(
                            args.url, contract, args.timeout, server_profile=args.server_profile,
                            joint16_protocol=args.joint16_protocol, wire_protocol=args.wire_protocol,
                            task_id=args.task_id,
                        ).__aenter__()
                        record(
                            {
                                "event": "server_health",
                                "health": client.health,
                                "client_chunk_reference": CHUNK_REFERENCE,
                                "client_chunk_integration": integration_for(args.server_profile),
                                "server_profile": args.server_profile,
                                "execution_rate_hz": 30 * args.speed,
                                "reference_authority": "user-confirmed Labs client contract",
                            }
                        )
                    if args.restore:
                        reset = start_command(start_state, contract, args.start_episode, speed=args.speed,
                                              task_id=args.task_id)
                        record(
                            {
                                "event": "episode_start",
                                "dataset": str(restore_dataset),
                                **task_metadata,
                                "episode": args.start_episode,
                                "frame_index": 0,
                                "state34": start_state.tolist(),
                                "identity": reset["start_identity"],
                            }
                        )
                        await return_before_inference(
                            reset,
                            publish=(
                                lambda command: publisher.publish(
                                    String(data=json.dumps(command, allow_nan=False))
                                )
                            )
                            if publisher
                            else None,
                            wait_hold=wait_hold,
                            record=record,
                        )
                        print(
                            json.dumps(
                                {
                                    "restore": "completed" if args.publish else "dry-run",
                                    "episode": args.start_episode,
                                    "published": args.publish,
                                }
                            )
                        )
                        after_ns = time.time_ns()
                    if not args.infer and not args.capture_only:
                        return
                    for index in range(args.max_chunks):
                        for attempt in range(2):
                            deadline = time.monotonic() + 5
                            while True:
                                observation = cache.next(
                                    after_ns=after_ns, timeout_s=max(0, deadline - time.monotonic())
                                )
                                images = encode_images(observation.images)
                                if time.time_ns() - observation.stamp_ns <= 200_000_000:
                                    break
                                # Cold decoder/encoder imports can age the first frame.
                                # Discard it and capture again, without relaxing freshness.
                                if time.monotonic() >= deadline:
                                    raise TimeoutError("Observation expired during image encoding")
                            frame_dir = args.output / f"observation_{index:06d}_attempt_{attempt}"
                            frame_dir.mkdir()
                            (frame_dir / "state.json").write_text(json.dumps(observation.state.tolist()))
                            for name, data in images.items():
                                (frame_dir / f"{name}.png").write_bytes(data)
                            record(
                                {
                                    "event": "observation",
                                    "index": index,
                                    "state": observation.state.tolist(),
                                    "model_input_state": model_input_state(observation.state, args.server_profile),
                                    "model_input_normalized": False,
                                    "stamp_ns": observation.stamp_ns,
                                    "source_stamps_ns": observation.source_stamps_ns,
                                    "model_hashes": contract.model_hashes,
                                    **task_metadata,
                                    "image_directory": frame_dir.name,
                                }
                            )
                            if args.capture_only:
                                print(json.dumps({"capture": "ok", "state_dim": 34, "images": [640, 480, 3]}))
                                break
                            started = time.monotonic()
                            # Retain the raw reply even when validation rejects it.
                            import msgpack

                            def save_reply(value, folder=frame_dir):
                                (folder / "response.msgpack").write_bytes(msgpack.packb(value, use_bin_type=True))
                                recording.event({"event": "wire_response", "response": value})

                            client.response_sink = save_reply
                            try:
                                result = await client.infer(observation.state, images, task=args.task)
                                break
                            except InferenceConnectionError as exc:
                                record({"event": "transport_retry", "index": index, "attempt": attempt,
                                        "error": str(exc), "published": False})
                                if attempt == 1:
                                    raise
                                print("Inference connection lost; reconnecting once and capturing a fresh observation...", flush=True)
                                await client.reconnect()
                                after_ns = time.time_ns()
                        if args.capture_only:
                            break
                        record(
                            {
                                "event": "raw_response",
                                "response": result,
                                "round_trip_ms": (time.monotonic() - started) * 1000,
                            }
                        )
                        command = command_for(
                            result, observation, contract, speed=args.speed,
                            server_profile=args.server_profile, execute_steps=args.execute_steps,
                        )
                        record(
                            {
                                "event": "inference",
                                "response": result,
                                "command": command,
                                "round_trip_ms": (time.monotonic() - started) * 1000,
                                "published": False,
                            }
                        )
                        if publisher is not None:
                            if not require_relay(relay_status(), contract):
                                raise RuntimeError("Relay stopped holding during inference")
                            # The relay independently checks age, joint drift, limits and IK.
                            publisher.publish(String(data=json.dumps(command, allow_nan=False)))
                            active_command = command["command_id"]
                            record({"event": "published", "command_id": command["command_id"]})
                            final_status = await wait_hold(command["command_id"], timeout=120)
                            record(
                                {
                                    "event": "completed",
                                    "command_id": command["command_id"],
                                    "status": final_status,
                                }
                            )
                            active_command = None
                        print(
                            json.dumps(
                                {
                                    "chunk": index,
                                    "rows": len(command["targets"]),
                                    "received_rows": command["prediction_rows"],
                                    "execution_rate_hz": command["execution_rate_hz"],
                                    "published": args.publish,
                                    "chunk_reference": CHUNK_REFERENCE,
                                    "chunk_integration": command["chunk_integration"],
                                }
                            )
                        )
                        after_ns = time.time_ns()  # Discard delayed frames from before completion.
                except asyncio.CancelledError:
                    recording.export("interrupt_snapshot")
                    if active_command is not None:
                        print("Stopping new requests; recording the accepted chunk until completion...", flush=True)
                        try:
                            final_status = await wait_hold(active_command, timeout=120)
                            record({"event": "completed", "command_id": active_command, "status": final_status})
                        except Exception as exc:
                            record({"event": "interrupt_tail_error", "command_id": active_command,
                                    "error": str(exc)})
                    raise
                finally:
                    if client is not None:
                        await client.__aexit__(None, None, None)

            async def interruptible_session():
                task = asyncio.current_task()
                previous = signal.getsignal(signal.SIGINT)
                interrupted = False

                def interrupt(_signum, _frame):
                    nonlocal interrupted
                    if interrupted:
                        raise KeyboardInterrupt
                    interrupted = True
                    task.cancel()

                signal.signal(signal.SIGINT, interrupt)
                try:
                    await session()
                finally:
                    signal.signal(signal.SIGINT, previous)

            exit_reason = "normal"
            try:
                asyncio.run(interruptible_session())
            except BaseException as exc:
                exit_reason = type(exc).__name__
                record({"event": "stopped", "error": str(exc), "type": type(exc).__name__})
                if isinstance(exc, asyncio.CancelledError):
                    exit_reason = "KeyboardInterrupt"
                    raise KeyboardInterrupt from None
                raise
            finally:
                path = recording.close(exit_reason)
                print(f"Saved action chunks and execution feedback: {path}", flush=True)
    finally:
        rclpy.shutdown()
        if thread is not None:
            thread.join(timeout=2)
        node.destroy_node()
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true", help="Return to the dataset episode start")
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Infer at the current pose; combine with --restore to return first",
    )
    parser.add_argument(
        "--manage-relay",
        action="store_true",
        help="Start a persistent site relay if none is discovered (robot gates required)",
    )
    parser.add_argument("--ik", type=Path, help="Host-built labs_fr3_ik executable for a new relay")
    parser.add_argument("--url", help="Model WebSocket /infer endpoint")
    parser.add_argument("--server-profile", choices=("labs", "c23", "smolvla", "joint16", "fastwam_joint16", "fastwam_eef20", "fastwam_joint16_next", "absolute20", "absolute_joint16"), default="labs")
    parser.add_argument("--joint16-protocol", default="smolvla.msgpack.v1", help="Joint16 WebSocket subprotocol")
    parser.add_argument("--wire-protocol", default="fastwam.msgpack.v1", help="WebSocket subprotocol for physical action modes")
    parser.add_argument("--task", help="Language instruction (default: validated dataset/server task)")
    parser.add_argument("--task-id", type=int, choices=(1, 2, 3, 4),
                        help="FastWAM absolute20 task selector; sends integer task_id instead of task text")
    parser.add_argument("--config", type=Path, default=Path("configs/labs_fr3_31"))
    parser.add_argument("--dataset", type=Path, help="Labs dataset containing the measured episode start")
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(f"outputs/labs_client_{time.time_ns()}"))
    parser.add_argument("--capture-only", action="store_true", help="Validate live input without a server")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument("--execute-steps", type=int, default=None,
                        help="Execute only the first N rows per reply (default: all); save the full reply")
    parser.add_argument(
        "--speed", type=float, default=0.3,
        help="Policy/return time scale (default 0.3: 9 Hz policy reference, slower joint return)",
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    args = parser.parse_args(argv)
    if args.task_id is not None:
        if args.task is not None:
            parser.error("Use --task-id or --task, not both")
        if args.server_profile != "absolute20" or args.wire_protocol != "fastwam.msgpack.v1":
            parser.error("--task-id requires --server-profile absolute20 --wire-protocol fastwam.msgpack.v1")
    if args.publish != args.enable_robot:
        parser.error("Robot publication requires both --publish and --enable-robot")
    if args.capture_only and (args.restore or args.infer):
        parser.error("--capture-only cannot combine with --restore/--infer")
    # Preserve the older run_labs_client.sh behavior when no operation was selected.
    if not args.capture_only and not (args.restore or args.infer):
        args.restore = args.infer = True
    if args.task_id is not None and args.restore and args.start_episode != 0:
        parser.error("Task-specific --restore uses episode 0/frame 0; --start-episode must be 0")
    if args.capture_only and args.publish:
        parser.error("--capture-only cannot publish")
    if args.infer and (not args.url or not args.url.startswith(("ws://", "wss://"))):
        parser.error("--url ws://.../infer or wss://.../infer is required")
    if not args.capture_only and args.dataset is None:
        parser.error("--dataset is required for the Labs contract and episode-start target")
    if args.start_episode < 0:
        parser.error("--start-episode must be nonnegative")
    horizon = 256 if args.server_profile in physical_contract.PROFILES else 32
    if args.execute_steps is not None and not 1 <= args.execute_steps <= horizon:
        parser.error(f"--execute-steps must be between 1 and {horizon} for this profile")
    if not 0 < args.speed <= 1 or not np.isfinite(args.timeout) or args.timeout <= 0 or args.max_chunks < 1:
        parser.error("Require 0 < speed <= 1, positive timeout and max-chunks")
    return args


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
