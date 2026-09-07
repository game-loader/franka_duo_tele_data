"""Exercise the RGB20D black-box policy boundary with recorded LeRobot actions.

Offline reads recorded observations. Live reads ROS observations; raw MCAP
recording is opt-in. Only --live --publish --enable-robot sends site commands.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import json
import threading
import time
from pathlib import Path

import numpy as np

from .cartesian_chunk import ActionChunk, CartesianChunkBuffer, ChunkRequest, check_tracking, pose_distance
from .config_io import load_mapping
from .rgb20d_io import CAMERAS, RGB20DCache, RGB20DContract, RGB20DObservation, RGB20DReader, RobotStateReader


class DatasetEpisodePolicy:
    """Replace predict_chunk with a server adapter later; request/response stay unchanged."""

    def __init__(self, contract: RGB20DContract, episode: int, *, horizon: int = 32, start: int = 0):
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq

        if horizon < 2 or horizon > 128 or start < 0:
            raise ValueError("horizon must be in [2,128] and start must be nonnegative")
        self.contract, self.horizon, self.start = contract, horizon, start
        episodes = pq.read_table(contract.root / "meta/episodes").to_pylist()
        matches = [row for row in episodes if row["episode_index"] == episode]
        if len(matches) != 1:
            raise ValueError(f"episode {episode} not found uniquely")
        self.meta = matches[0]
        table = (
            ds.dataset(contract.root / "data", format="parquet")
            .to_table(filter=ds.field("episode_index") == episode)
            .sort_by("frame_index")
        )
        if table.num_rows != self.meta["length"]:
            raise ValueError("episode metadata length differs from data")
        indices = table["frame_index"].to_numpy()
        if not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError("episode frame indices are not consecutive")
        self.states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[start:]
        self.actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[start:]
        self.stamps = table["observation.source_timestamp_ns"].to_numpy()[start:]
        if len(self.states) == 0:
            raise ValueError("start index is outside the episode")
        if not np.array_equal(self.actions[:-1], self.states[1:]):
            raise ValueError("action[t] differs from state[t+1] inside this episode")
        for state, action in zip(self.states, self.actions, strict=True):
            contract.validate_action(state)
            contract.validate_action(action)

    def predict_chunk(self, request: ChunkRequest) -> ActionChunk:
        inputs = request.observation.policy_input()
        if inputs["observation.state"].shape != (20,) or not np.isfinite(inputs["observation.state"]).all():
            raise ValueError("black-box state input must be finite float32[20]")
        for i, camera in enumerate(CAMERAS):
            image = inputs[f"observation.images.camera{i + 1}"]
            if image.dtype != np.uint8 or image.shape != self.contract.image_shapes[camera]:
                raise ValueError(f"invalid black-box RGB input for {camera}")
        return self.recorded_chunk(request.start_step, request.observation.stamp_ns)

    def recorded_chunk(self, start_step: int, feedback_stamp_ns: int) -> ActionChunk:
        actions = self.actions[start_step : start_step + self.horizon].copy()
        return ActionChunk(start_step, feedback_stamp_ns, actions)

    def observations(self):
        import av

        with contextlib.ExitStack() as stack:
            decoders = {}
            for key in CAMERAS:
                prefix = f"videos/observation.images.{key}"
                path = self.contract.info["video_path"].format(
                    video_key=f"observation.images.{key}",
                    chunk_index=self.meta[f"{prefix}/chunk_index"],
                    file_index=self.meta[f"{prefix}/file_index"],
                )
                container = stack.enter_context(av.open(str(self.contract.root / path)))
                stream = container.streams.video[0]
                start_time = self.meta[f"{prefix}/from_timestamp"] + self.start / self.contract.fps
                container.seek(max(0, int(start_time / stream.time_base)), stream=stream, backward=True)
                decoders[key] = (iter(container.decode(stream)), start_time)
            for index, state in enumerate(self.states):
                images = {}
                for key, (decoder, start_time) in decoders.items():
                    target = start_time + index / self.contract.fps
                    for frame in decoder:
                        if float(frame.time) >= target - 0.5 / self.contract.fps:
                            break
                    else:
                        raise ValueError(f"{key} video ended before episode frame {index}")
                    if abs(float(frame.time) - target) > 0.5 / self.contract.fps:
                        raise ValueError(f"{key} video timestamp does not match data frame {index}")
                    images[key] = frame.to_ndarray(format="rgb24")
                yield RGB20DObservation(
                    int(self.stamps[index]),
                    state,
                    images,
                    {"head": int(self.stamps[index])},
                    time.monotonic_ns(),
                )


def verify_offline(contract: RGB20DContract, episodes: list[int], horizon: int, report: Path) -> dict:
    summaries = []
    for episode in episodes:
        policy = DatasetEpisodePolicy(contract, episode, horizon=horizon)
        buffer = CartesianChunkBuffer(contract)
        maximum_position, maximum_angle, roundtrip = 0.0, 0.0, 0.0
        requests = 0
        previous = None
        for index, observation in enumerate(policy.observations()):
            if buffer.remaining <= horizon // 2 and buffer.last_start + horizon < len(policy.actions):
                buffer.submit(policy.predict_chunk(ChunkRequest(index, observation)))
                requests += 1
            action = buffer.pop()
            np.testing.assert_array_equal(action, policy.actions[index])
            link0 = contract.action_spec.to_link0_action(action)
            # Independent round trip checks position AND orientation; applying
            # an extra tool offset or converting twice would fail this check.
            from .action_spec import matrix_to_rot6d, rot6d_to_matrix

            for offset, side in ((0, "left"), (9, "right")):
                transform = contract.transforms[side]
                restored = np.r_[
                    transform[:3, :3] @ link0[offset : offset + 3] + transform[:3, 3],
                    matrix_to_rot6d(transform[:3, :3] @ rot6d_to_matrix(link0[offset + 3 : offset + 9])),
                ]
                error = float(np.max(np.abs(restored - action[offset : offset + 9])))
                roundtrip = max(roundtrip, error)
                if error > 2e-5:
                    raise ValueError("midpoint/link0 conversion round trip failed")
            if previous is not None:
                position, angle = pose_distance(previous, action)
                maximum_position = max(maximum_position, float(position.max()))
                maximum_angle = max(maximum_angle, float(angle.max()))
            previous = action
        summary = {
            "episode_index": episode,
            "frames": len(policy.actions),
            "chunks": requests,
            "max_target_step_m": maximum_position,
            "max_target_step_rad": maximum_angle,
            "max_coordinate_roundtrip_error": roundtrip,
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    result = {
        "schema": "franka_duo_rgb20d_blackbox_check_v1",
        "robot_commands_published": 0,
        "dataset": str(contract.root.resolve()),
        "fps": contract.fps,
        "camera_shapes_hwc": contract.image_shapes,
        "state_dim": 20,
        "action_dim": 20,
        "frames": sum(s["frames"] for s in summaries),
        "episodes": summaries,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def check_live_controllers(node) -> None:
    """Read-only checks: never load, activate, or recover a controller here."""
    from controller_manager_msgs.srv import ListControllers
    from rcl_interfaces.srv import GetParameters

    def call(service_type, name, request):
        client = node.create_client(service_type, name)
        try:
            if not client.wait_for_service(timeout_sec=3):
                raise TimeoutError(f"control service unavailable: {name}")
            future = client.call_async(request)
            deadline = time.monotonic() + 3
            while not future.done():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"control service timed out: {name}")
                time.sleep(0.01)
            return future.result()
        finally:
            node.destroy_client(client)

    relay = call(
        GetParameters,
        "/franka_duo_rgb20d_relay/get_parameters",
        GetParameters.Request(names=["action_frame", "enable_robot", "input_topic"]),
    )
    if (
        relay.values[0].string_value != "link0"
        or not relay.values[1].bool_value
        or relay.values[2].string_value != "/franka_duo/rgb20d/action"
    ):
        raise ValueError("site relay must enable robot output and explicitly accept link0 RGB20D actions")
    for side in ("left", "right"):
        controller_name = f"{side}_policy_cartesian_pose_controller"
        controllers = call(
            ListControllers, f"/{side}/controller_manager/list_controllers", ListControllers.Request()
        ).controller
        active = [c for c in controllers if c.state == "active"]
        selected = [
            c
            for c in active
            if c.name == controller_name
            and c.type == "franka_duo_policy_control/PolicyCartesianPoseController"
        ]
        if len(selected) != 1 or any(c.name != controller_name and c.claimed_interfaces for c in active):
            raise ValueError(
                f"{side} requires an active policy Cartesian controller and no competing command controller"
            )
        settings = call(
            GetParameters,
            f"/{side}/{controller_name}/get_parameters",
            GetParameters.Request(
                names=[
                    "allow_motion",
                    "target_timeout_s",
                    "expected_frame_id",
                    "target_topic",
                    "smoothing_backend",
                ]
            ),
        )
        if (
            not settings.values[0].bool_value
            or not 0.05 < settings.values[1].double_value <= 0.5
            or settings.values[2].string_value != f"{side}_fr3v2_link0"
            or settings.values[3].string_value != f"/franka_duo/policy/{side}/target_pose"
            or settings.values[4].string_value != "ruckig_velocity_v1"
        ):
            raise ValueError(f"{side} controller requires link0, allow_motion, Ruckig and a bounded watchdog")
        if len(node.get_publishers_info_by_topic(f"/franka_duo/policy/{side}/target_pose")) != 1:
            raise ValueError(f"{side} Cartesian target must have exactly one relay publisher")


def check_joint_servo_controllers(node, config, speed: float) -> None:
    """Read-only checks for the joint-servo path; never loads or switches controllers."""
    from controller_manager_msgs.srv import ListControllers
    from rcl_interfaces.srv import GetParameters

    def call(service_type, name, request):
        client = node.create_client(service_type, name)
        try:
            if not client.wait_for_service(timeout_sec=3):
                raise TimeoutError(f"control service unavailable: {name}")
            future = client.call_async(request)
            deadline = time.monotonic() + 3
            while not future.done():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"control service timed out: {name}")
                time.sleep(0.01)
            return future.result()
        finally:
            node.destroy_client(client)

    servo = call(
        GetParameters,
        "/franka_duo_joint_servo/get_parameters",
        GetParameters.Request(names=["action_frame", "playback_speed", "chunk_topic", "status_topic"]),
    )
    if (
        servo.values[0].string_value != "link0"
        or abs(servo.values[1].double_value - speed) > 1e-9
        or servo.values[2].string_value != config["joint_servo_chunk_topic"]
        or servo.values[3].string_value != config["joint_servo_status_topic"]
    ):
        raise ValueError("joint servo must use link0 actions, the replay speed and the configured topics")
    for side in ("left", "right"):
        relay = call(
            GetParameters,
            f"/{side}_gello_target_relay/get_parameters",
            GetParameters.Request(names=["enable_robot", "output_topic", "input_topic"]),
        )
        if (
            not relay.values[0].bool_value
            or relay.values[1].string_value != f"/{side}/gello/joint_states"
            or relay.values[2].string_value != f"/franka_duo/joint_servo/{side}/target"
        ):
            raise ValueError(f"{side} gello_target_relay must enable robot output on the site topics")
        controllers = call(
            ListControllers, f"/{side}/controller_manager/list_controllers", ListControllers.Request()
        ).controller
        active = [c for c in controllers if c.state == "active"]
        impedance = [c for c in active if c.name == "joint_impedance_controller"]
        if len(impedance) != 1 or any(
            c.name != "joint_impedance_controller" and c.claimed_interfaces for c in active
        ):
            raise ValueError(
                f"{side} requires an active joint_impedance_controller and no competing controller"
            )
        if len(node.get_publishers_info_by_topic(f"/{side}/gello/joint_states")) != 1:
            raise ValueError(f"{side} gello target topic must have exactly one publisher (the relay)")
    if len(node.get_publishers_info_by_topic(config["joint_servo_status_topic"])) != 1:
        raise ValueError("joint servo status topic must have exactly one publisher")


def run_joint_servo(args, contract: RGB20DContract) -> None:
    """Recorded-action replay through the site joint servo (impedance controller path)."""
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

    from .joint_servo_client import ChunkPacer, chunk_payload, parse_status

    if args.publish != args.enable_robot:
        raise ValueError("robot publication requires both --publish and --enable-robot")
    config = load_mapping(args.config)
    for key in ("joint_servo_chunk_topic", "joint_servo_status_topic"):
        if not str(config.get(key, "")).startswith("/franka_duo/joint_servo/"):
            raise ValueError(f"{key} must be a site joint_servo topic")
    contract.action_spec = dataclasses.replace(
        contract.action_spec,
        workspace_min=tuple(config["workspace_min"]),
        workspace_max=tuple(config["workspace_max"]),
    )
    policy = DatasetEpisodePolicy(contract, args.episode, horizon=args.horizon, start=args.start_index)
    count = min(args.max_steps, len(policy.actions))
    node = None
    ros_thread = None
    latest = {"status": None, "status_ns": 0}
    lock = threading.Lock()
    try:
        rclpy.init()
        node = rclpy.create_node("franka_duo_rgb20d_joint_servo_replay")
        cache = RGB20DCache(history_size=120)
        reader = RobotStateReader(cache, contract, max_age_ms=config["max_input_age_ms"])
        for side in ("left", "right"):
            node.create_subscription(
                PoseStamped,
                config["topics"][f"{side}_pose"],
                getattr(cache, f"store_{side}_pose"),
                qos_profile_sensor_data,
            )
            node.create_subscription(
                JointState,
                config["topics"][f"{side}_gripper"],
                getattr(cache, f"store_{side}_gripper_states"),
                qos_profile_sensor_data,
            )

        def on_status(msg):
            with lock:
                latest["status"] = msg.data
                latest["status_ns"] = time.monotonic_ns()

        node.create_subscription(String, config["joint_servo_status_topic"], on_status, 10)
        publisher = (
            node.create_publisher(Float32MultiArray, config["joint_servo_chunk_topic"], 10)
            if args.publish
            else None
        )
        ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        ros_thread.start()
        if args.publish:
            check_joint_servo_controllers(node, config, args.speed)
            deadline = time.monotonic() + 15
            while publisher is not None and publisher.get_subscription_count() != 1:
                if time.monotonic() > deadline:
                    raise TimeoutError("unique joint servo chunk subscriber is not present")
                time.sleep(0.02)

        def status():
            with lock:
                text, stamp_ns = latest["status"], latest["status_ns"]
            if text is None:
                return None
            if time.monotonic_ns() - stamp_ns > 500_000_000:
                raise TimeoutError("joint servo status expired")
            return parse_status(text)

        deadline = time.monotonic() + 5
        initial = status()
        while initial is None:
            if time.monotonic() > deadline:
                raise TimeoutError("joint servo status not received")
            time.sleep(0.02)
            initial = status()
        if initial.started or initial.fault:
            raise RuntimeError("joint servo must be idle (not started, no fault) before replay")

        observation = reader.next(timeout_s=3)
        start_position, start_rotation = pose_distance(observation.state, policy.actions[0])
        print(
            json.dumps(
                {
                    "start_position_error_m": start_position.tolist(),
                    "start_rotation_error_rad": start_rotation.tolist(),
                    "joint_servo": True,
                }
            ),
            flush=True,
        )
        check_tracking(
            observation.state,
            policy.actions[0],
            max_m=config["max_start_distance_m"],
            max_rad=config["max_start_rotation_rad"],
        )

        pacer = ChunkPacer(args.horizon, count)
        dt = 1 / contract.fps
        sent = 0
        max_position, max_rotation = np.zeros(2), np.zeros(2)
        last_report = time.monotonic()
        # Generous bound: at 0.1x speed the whole replay takes count / 3 seconds.
        timeout = time.monotonic() + 30 + count / (contract.fps * args.speed) * 1.5
        while True:
            time.sleep(dt)
            if time.monotonic() > timeout:
                raise TimeoutError("joint servo replay did not finish in time")
            current = status()
            if current is not None and current.fault:
                raise RuntimeError(f"joint servo fault: {current.fault_reason}")
            observation = reader.next(timeout_s=0.5)
            if current is not None and current.started:
                index = min(count - 1, max(0, int(current.step)))
                position_error, rotation_error = pose_distance(observation.state, policy.actions[index])
                check_tracking(
                    observation.state,
                    policy.actions[index],
                    max_m=config["max_tracking_distance_m"],
                    max_rad=config["max_tracking_rotation_rad"],
                )
                max_position = np.maximum(max_position, position_error)
                max_rotation = np.maximum(max_rotation, rotation_error)
            start = pacer.next_start(current)
            if start is not None:
                actions = policy.actions[start : min(count, start + args.horizon)]
                # The servo runs with action_frame=link0; convert once here like the Cartesian path.
                link0_rows = np.stack([contract.action_spec.to_link0_action(row) for row in actions])
                data, dims, offset = chunk_payload(link0_rows, start)
                if publisher is not None:
                    message = Float32MultiArray(data=data)
                    message.layout.dim = [
                        MultiArrayDimension(label="rows", size=dims[0], stride=dims[0] * dims[1]),
                        MultiArrayDimension(label="action", size=dims[1], stride=dims[1]),
                    ]
                    message.layout.data_offset = offset
                    publisher.publish(message)
                pacer.record(start)
                sent += 1
                if publisher is None and start > 0:
                    # Dry run: without a servo start there is nothing to pace; stop after one pass.
                    break
            if publisher is None and pacer.last_start >= 0 and current is not None and not current.started:
                break
            if pacer.finished(current):
                break
            if time.monotonic() - last_report > 2:
                last_report = time.monotonic()
                print(
                    json.dumps(
                        {
                            "servo_step": None if current is None else current.step,
                            "chunks_sent": sent,
                            "tracking_error_rad": None if current is None else current.tracking_error_rad,
                        }
                    ),
                    flush=True,
                )
        print(
            json.dumps(
                {
                    "frames": count,
                    "chunks_published": sent if args.publish else 0,
                    "max_tracking_position_error_m": max_position.tolist(),
                    "max_tracking_rotation_error_rad": max_rotation.tolist(),
                }
            ),
            flush=True,
        )
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        if ros_thread is not None:
            ros_thread.join(timeout=2)
        if node is not None:
            node.destroy_node()


def optional_recorder(args, config):
    if not args.record_mcap:
        return None
    from .mcap_recorder import RawMcapRecorder, load_config as load_mcap_config, preflight_ros2

    raw = load_mcap_config(args.config.parent / config["mcap_config"])
    available = set(raw.topics)
    if raw.arm_sampling is not None:
        available.update(route.source_topic for route in raw.arm_sampling.routes)
    if not set(config["topics"].values()).issubset(available):
        raise ValueError("all live inputs must be part of the raw MCAP contract")
    raw = dataclasses.replace(
        raw,
        dataset_name="franka_duo_rgb20d_replay",
        max_episodes=1,
        topics=tuple(dict.fromkeys((*raw.topics, config["trace_topic"]))),
    )
    return RawMcapRecorder(raw, preflight_ros2())


def run_live(args, contract: RGB20DContract) -> None:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float32MultiArray, String

    if args.publish != args.enable_robot:
        raise ValueError("robot publication requires both --publish and --enable-robot")
    config = load_mapping(args.config)
    for key in (
        "max_input_age_ms",
        "max_start_distance_m",
        "max_start_rotation_rad",
        "max_tracking_distance_m",
        "max_tracking_rotation_rad",
        "max_target_step_m",
        "max_target_step_rad",
    ):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    contract.action_spec = dataclasses.replace(
        contract.action_spec,
        workspace_min=tuple(config["workspace_min"]),
        workspace_max=tuple(config["workspace_max"]),
    )
    command_topic = str(config["command_topic"])
    if command_topic != "/franka_duo/rgb20d/action":
        raise ValueError("RGB20D commands must target the dedicated site relay /franka_duo/rgb20d/action")
    trace_topic = str(config["trace_topic"])
    if trace_topic == command_topic:
        raise ValueError("trace topic must differ from command topic")
    policy = DatasetEpisodePolicy(contract, args.episode, horizon=args.horizon, start=args.start_index)
    count = min(args.max_steps, len(policy.actions))
    recorder = optional_recorder(args, config)
    node = None
    ros_thread = None
    input_thread = None
    stop = threading.Event()
    latest = {"observation": None, "error": None}
    lock = threading.Lock()
    try:
        if recorder is not None:
            recorder.start_episode()
        rclpy.init()
        node = rclpy.create_node("franka_duo_rgb20d_replay")
        cache = RGB20DCache(history_size=120)
        reader_class = RobotStateReader if args.recorded_actions else RGB20DReader
        reader = reader_class(cache, contract, max_age_ms=config["max_input_age_ms"])
        for camera in () if args.recorded_actions else CAMERAS:
            node.create_subscription(
                Image,
                config["topics"][camera],
                lambda msg, key=camera: cache.store_image(key, msg),
                qos_profile_sensor_data,
            )
        for side in ("left", "right"):
            node.create_subscription(
                PoseStamped,
                config["topics"][f"{side}_pose"],
                getattr(cache, f"store_{side}_pose"),
                qos_profile_sensor_data,
            )
            node.create_subscription(
                JointState,
                config["topics"][f"{side}_gripper"],
                getattr(cache, f"store_{side}_gripper_states"),
                qos_profile_sensor_data,
            )
        trace = node.create_publisher(String, trace_topic, 10) if recorder is not None else None
        publisher = node.create_publisher(Float32MultiArray, command_topic, 1) if args.publish else None
        ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        ros_thread.start()
        if args.publish:
            check_live_controllers(node)
        deadline = time.monotonic() + 15
        while (trace is not None and trace.get_subscription_count() < 1) or (
            publisher is not None and publisher.get_subscription_count() != 1
        ):
            if time.monotonic() > deadline:
                raise TimeoutError("raw trace recorder or unique site relay is not subscribed")
            time.sleep(0.02)

        def capture():
            try:
                while not stop.is_set():
                    observation = reader.next()
                    with lock:
                        latest["observation"] = observation
                    if args.recorded_actions:
                        stop.wait(1 / contract.fps)
            except Exception as exc:
                with lock:
                    latest["error"] = exc

        input_thread = threading.Thread(target=capture, daemon=True)
        input_thread.start()

        def current():
            if recorder is not None:
                relay_error = recorder.arm_relay_health_error()
                if relay_error is not None:
                    raise RuntimeError(relay_error)
                if recorder.active is None or recorder.active.process.poll() is not None:
                    raise RuntimeError("raw MCAP recorder stopped during replay")
            with lock:
                observation, error = latest["observation"], latest["error"]
            if error is not None:
                raise error
            if (
                observation is not None
                and time.monotonic_ns() - observation.arrival_ns > config["max_input_age_ms"] * 1e6
            ):
                raise TimeoutError("live RGB/state feedback expired")
            return observation

        deadline = time.monotonic() + 3
        observation = current()
        while observation is None:
            if time.monotonic() > deadline:
                raise TimeoutError("live input not ready")
            time.sleep(0.01)
            observation = current()
        if args.snapshot_input:
            args.snapshot_input.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.snapshot_input, **observation.policy_input())
        start_position, start_rotation = pose_distance(observation.state, policy.actions[0])
        print(
            json.dumps(
                {
                    "start_position_error_m": start_position.tolist(),
                    "start_rotation_error_rad": start_rotation.tolist(),
                    "record_mcap": args.record_mcap,
                }
            ),
            flush=True,
        )
        # Recorded trajectories have an absolute start pose. No silent offsets,
        # teleporting to frame zero, or substituting a measured action target.
        check_tracking(
            observation.state,
            policy.actions[0],
            max_m=config["max_start_distance_m"],
            max_rad=config["max_start_rotation_rad"],
        )
        buffer = CartesianChunkBuffer(
            contract, max_step_m=config["max_target_step_m"], max_step_rad=config["max_target_step_rad"]
        )

        def predict(start_step, feedback):
            if args.recorded_actions:
                return policy.recorded_chunk(start_step, feedback.stamp_ns)
            return policy.predict_chunk(ChunkRequest(start_step, feedback))

        buffer.submit(predict(0, observation))
        dt = 1 / contract.fps
        started = time.monotonic()
        pending = None
        tick, index, action, link0 = 0, -1, None, None
        max_position, max_rotation = np.zeros(2), np.zeros(2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
            end_tick = int(np.ceil(count / args.speed))
            while tick < end_tick:
                tick += 1
                due = started + tick * dt
                time.sleep(max(0, due - time.monotonic()))
                if time.monotonic() - due > min(dt, 0.05):
                    raise TimeoutError("control deadline missed; refusing burst catch-up commands")
                observation = current()
                if pending is not None and pending.done():
                    buffer.submit(pending.result())
                    pending = None
                # Slow the recorded trajectory while keeping a 30 Hz watchdog
                # heartbeat. No servo reset, stop/start goal, or burst of rows.
                scheduled_index = int((tick - 1) * args.speed + 1e-9)
                if scheduled_index > index:
                    action = buffer.pop()
                    index += 1
                    link0 = contract.action_spec.to_link0_action(action)
                check_tracking(
                    observation.state,
                    action,
                    max_m=config["max_tracking_distance_m"],
                    max_rad=config["max_tracking_rotation_rad"],
                )
                position_error, rotation_error = pose_distance(observation.state, action)
                max_position = np.maximum(max_position, position_error)
                max_rotation = np.maximum(max_rotation, rotation_error)
                if publisher is not None:
                    publisher.publish(Float32MultiArray(data=link0.tolist()))
                if trace is not None:
                    trace.publish(
                        String(
                            data=json.dumps(
                                {
                                    "schema": "franka_duo_rgb20d_replay_v1",
                                    "step": index,
                                    "tick": tick,
                                    "dataset_episode": args.episode,
                                    "dataset_frame": args.start_index + index,
                                    "head_stamp_ns": None if args.recorded_actions else observation.stamp_ns,
                                    "feedback_stamp_ns": observation.stamp_ns,
                                    "input_mode": "robot_state" if args.recorded_actions else "rgb_state",
                                    "source_stamps_ns": observation.source_stamps_ns,
                                    "state": observation.state.tolist(),
                                    "midpoint_action": action.tolist(),
                                    "link0_action": link0.tolist(),
                                    "published": args.publish,
                                    "wall_period_s": dt,
                                    "replay_speed": args.speed,
                                    "control_elapsed_s": time.monotonic() - started,
                                }
                            )
                        )
                    )
                if (
                    pending is None
                    and buffer.cursor > buffer.last_start
                    and buffer.remaining <= args.horizon // 2
                    and buffer.last_start + args.horizon < min(count, len(policy.actions))
                ):
                    pending = worker.submit(predict, buffer.cursor, observation)
        # Allow the controller watchdog to decelerate after the final command.
        time.sleep(1.5)
        if recorder is not None:
            recorder.save_episode()
        print(
            json.dumps(
                {
                    "frames": count,
                    "robot_commands_published": tick if args.publish else 0,
                    "raw_mcap": str(recorder.dataset_path) if recorder is not None else None,
                    "max_tracking_position_error_m": max_position.tolist(),
                    "max_tracking_rotation_error_rad": max_rotation.tolist(),
                }
            ),
            flush=True,
        )
    except BaseException:
        if recorder is not None and recorder.active is not None:
            recorder.preserve_interrupted_episode(reason="RGB20D replay interrupted or rejected")
        raise
    finally:
        stop.set()
        if input_thread is not None:
            input_thread.join(timeout=2)
        if rclpy.ok():
            rclpy.shutdown()
        if ros_thread is not None:
            ros_thread.join(timeout=2)
        if node is not None:
            node.destroy_node()
        if recorder is not None:
            recorder.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/tmr_rgb20d.yaml"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--recorded-actions", action="store_true", help="live robot feedback without cameras")
    parser.add_argument(
        "--joint-servo", action="store_true", help="send absolute-step chunks to the site joint servo"
    )
    parser.add_argument("--record-mcap", action="store_true", help="optionally record raw evaluation MCAP")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--snapshot-input", type=Path)
    parser.add_argument(
        "--export-ptp-target", type=Path, help="export selected first action for the site PTP tool"
    )
    parser.add_argument("--report", type=Path, default=Path("outputs/rgb20d_blackbox_report.json"))
    args = parser.parse_args(argv)
    if not 0 < args.speed <= 1 or args.max_steps < 1 or args.horizon < 2 or args.horizon > 128:
        parser.error("speed must be in (0,1], max-steps positive, horizon in [2,128]")
    if (args.publish or args.enable_robot) and not args.live:
        parser.error("recorded observations cannot enable robot output; use --live")
    if args.live and args.all_episodes:
        parser.error("live replay is restricted to one explicitly selected episode")
    if args.recorded_actions and (not args.live or args.snapshot_input):
        parser.error("--recorded-actions requires --live and cannot snapshot model RGB inputs")
    if args.record_mcap and not args.live:
        parser.error("--record-mcap requires --live")
    contract = RGB20DContract(args.dataset)
    if args.export_ptp_target:
        if args.live or args.all_episodes or args.record_mcap:
            parser.error("PTP target export is offline and selects one episode/frame")
        policy = DatasetEpisodePolicy(contract, args.episode, start=args.start_index)
        args.export_ptp_target.parent.mkdir(parents=True, exist_ok=True)
        args.export_ptp_target.write_text(
            json.dumps(
                {
                    "dataset": str(contract.root.resolve()),
                    "episode_index": args.episode,
                    "frame_index": args.start_index,
                    "actions": [policy.actions[0, :18].tolist()],
                    "grippers": policy.actions[0, 18:].tolist(),
                    "coordinate_transforms": contract.manifest["coordinate_transforms"],
                },
                indent=2,
            )
            + "\n"
        )
        print(str(args.export_ptp_target))
        return 0
    if args.joint_servo and (not args.live or not args.recorded_actions or args.record_mcap):
        parser.error("--joint-servo requires --live --recorded-actions and no MCAP recording")
    if args.joint_servo:
        run_joint_servo(args, contract)
    elif args.live:
        run_live(args, contract)
    else:
        if args.start_index:
            parser.error("offline verification checks entire episodes; omit --start-index")
        episodes = list(range(contract.info["total_episodes"])) if args.all_episodes else [args.episode]
        verify_offline(contract, episodes, args.horizon, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
