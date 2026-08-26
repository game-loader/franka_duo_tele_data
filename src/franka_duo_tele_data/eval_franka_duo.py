#!/usr/bin/env python3
"""Run an exported Franka Duo policy on live ROS camera input.

The executable is intentionally a relay-oriented evaluator.  It prints the
20D Cartesian action by default and never publishes to a robot controller
unless both ``--publish`` and ``--enable-robot`` are supplied.  The relay topic
and message type are explicit configuration so this tool cannot silently send
an action to an unrelated Franka driver.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .action_spec import FrankaDuoActionSpec
from .config_io import load_mapping
from .franka_duo_eval_io import (
    EvalCameraConfig,
    EvalObservationCache,
    PointCloudConfig,
    SynchronizedObservation,
    SynchronizedObservationReader,
)
from .mcap_recorder import (
    McapRecorderConfig,
    RawMcapRecorder,
    load_config as load_mcap_config,
    parse_reward,
    preflight_ros2,
    prompt_reward,
)
from .rl100_eval_policy import PolicyBundle, load_policy_bundle

LOGGER = logging.getLogger("franka_duo_eval")


@dataclasses.dataclass
class EvalConfig:
    topics: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "head_image": "/isaac/head_camera/image_raw",
            "head_depth": "/isaac/head_camera/depth",
            "head_camera_info": "/isaac/head_camera/camera_info",
            "wrist_left_image": "/isaac/left_wrist_camera/image_raw",
            "wrist_right_image": "/isaac/right_wrist_camera/image_raw",
            "joint_states": "/isaac/joint_states_full",
        }
    )
    cameras: dict[str, EvalCameraConfig] = dataclasses.field(default_factory=dict)
    fps: int = 30
    sync_history_size: int = 30
    rgb_match_tolerance_ms: float = 45.0
    depth_match_tolerance_ms: float = 16.0
    state_match_tolerance_ms: float = 50.0
    max_message_age_ms: float = 150.0
    sync_wait_timeout_ms: float = 75.0
    state_gripper_closed: float | None = None
    state_gripper_open: float | None = None
    publish_topic: str = "/franka_duo/policy_action"
    trace_topic: str = "/franka_duo/eval/action_trace"
    output_jsonl: Path | None = None
    inference_timeout_ms: float = 100.0
    max_steps: int | None = None
    mcap_config: Path | None = None
    mcap_output_root: Path | None = None
    mcap_dataset_name: str | None = None


def _default_cameras(topics: Mapping[str, str]) -> dict[str, EvalCameraConfig]:
    return {
        "head": EvalCameraConfig(
            key="head",
            image_topic=topics["head_image"],
            depth_topic=topics["head_depth"],
            camera_info_topic=topics["head_camera_info"],
            width=1280,
            height=720,
            fps=30,
            depth_scale=0.001,
        ),
        "wrist_left": EvalCameraConfig(
            key="wrist_left",
            image_topic=topics["wrist_left_image"],
            width=480,
            height=270,
            fps=30,
        ),
        "wrist_right": EvalCameraConfig(
            key="wrist_right",
            image_topic=topics["wrist_right_image"],
            width=480,
            height=270,
            fps=30,
        ),
    }


def load_eval_config(path: Path | None) -> EvalConfig:
    config = EvalConfig()
    config.cameras = _default_cameras(config.topics)
    if path is None:
        return config
    raw = load_mapping(path)
    if not isinstance(raw, Mapping):
        raise ValueError("eval config must contain a mapping")
    if isinstance(raw.get("topics"), Mapping):
        config.topics.update({str(key): str(value) for key, value in raw["topics"].items()})
    for key in (
        "fps",
        "sync_history_size",
        "rgb_match_tolerance_ms",
        "depth_match_tolerance_ms",
        "state_match_tolerance_ms",
        "max_message_age_ms",
        "sync_wait_timeout_ms",
        "state_gripper_closed",
        "state_gripper_open",
        "publish_topic",
        "trace_topic",
        "inference_timeout_ms",
        "max_steps",
    ):
        if key in raw:
            setattr(config, key, raw[key])
    if raw.get("output_jsonl") is not None:
        config.output_jsonl = Path(str(raw["output_jsonl"]))
    mcap_values = raw.get("mcap", {})
    if not isinstance(mcap_values, Mapping):
        raise ValueError("eval config mcap must contain a mapping")
    if mcap_values.get("config") is not None:
        mcap_path = Path(str(mcap_values["config"]))
        config.mcap_config = mcap_path if mcap_path.is_absolute() else path.parent / mcap_path
    if mcap_values.get("output_root") is not None:
        config.mcap_output_root = Path(str(mcap_values["output_root"]))
    if mcap_values.get("dataset_name") is not None:
        config.mcap_dataset_name = str(mcap_values["dataset_name"])
    camera_values = raw.get("cameras", {})
    if not isinstance(camera_values, Mapping):
        raise ValueError("eval config cameras must contain a mapping")
    defaults = _default_cameras(config.topics)
    cameras: dict[str, EvalCameraConfig] = {}
    for key, default in defaults.items():
        value = camera_values.get(key, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"camera config {key} must contain a mapping")
        cameras[key] = EvalCameraConfig(
            key=key,
            image_topic=str(value.get("image", default.image_topic)),
            depth_topic=value.get("depth", default.depth_topic),
            camera_info_topic=value.get("camera_info", default.camera_info_topic),
            width=int(value.get("width", default.width)),
            height=int(value.get("height", default.height)),
            fps=int(value.get("fps", default.fps)),
            depth_scale=float(value.get("depth_scale", default.depth_scale)),
            depth_registered=bool(value.get("depth_registered", default.depth_registered)),
        )
    config.cameras = cameras
    if config.fps <= 0 or config.sync_history_size <= 0:
        raise ValueError("fps and sync_history_size must be positive")
    for key in (
        "rgb_match_tolerance_ms",
        "depth_match_tolerance_ms",
        "state_match_tolerance_ms",
        "max_message_age_ms",
        "sync_wait_timeout_ms",
        "inference_timeout_ms",
    ):
        if float(getattr(config, key)) <= 0:
            raise ValueError(f"{key} must be positive")
    if config.max_steps is not None and int(config.max_steps) <= 0:
        raise ValueError("max_steps must be positive when set")
    if not config.cameras["head"].depth_topic or not config.cameras["head"].camera_info_topic:
        raise ValueError("head depth and camera_info topics are required for ZED point-cloud eval")
    for key, camera in config.cameras.items():
        if camera.fps != config.fps:
            raise ValueError(f"camera {key} fps {camera.fps} does not match eval fps {config.fps}")
    return config


def _source_frame(observation: SynchronizedObservation, source: str | Sequence[str]) -> np.ndarray:
    sources = {
        "head": observation.head_rgb,
        "wrist_left": observation.wrist_left_rgb,
        "wrist_right": observation.wrist_right_rgb,
        "point_cloud": observation.point_cloud,
        "state": observation.state,
    }
    if isinstance(source, str):
        if source not in sources or sources[source] is None:
            raise ValueError(f"Policy input source {source!r} is not available in this observation")
        return np.asarray(sources[source])
    source_names = [str(item) for item in source]
    if not source_names:
        raise ValueError("Policy input source list cannot be empty")
    frames = [_source_frame(observation, item) for item in source_names]
    if any(frame.ndim != 3 for frame in frames) or any(
        frame.shape[0] != frames[0].shape[0] for frame in frames
    ):
        raise ValueError("Stacked image sources must be HWC arrays with matching heights")
    return np.ascontiguousarray(np.concatenate(frames, axis=1))


def build_policy_observation(
    observation: SynchronizedObservation, manifest: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Map physical sensor names to the exact keys used during policy training."""

    raw_inputs = manifest.get("inputs", manifest.get("observation", {}))
    if not isinstance(raw_inputs, Mapping):
        raise ValueError("manifest.inputs must be a mapping")
    point_key = raw_inputs.get("point_cloud_key", "observation.point_cloud")
    if point_key is None:
        raise ValueError("manifest.inputs.point_cloud_key is required")
    result: dict[str, np.ndarray] = {str(point_key): observation.point_cloud}
    state_key = raw_inputs.get("state_key")
    if state_key:
        if observation.state is None:
            raise ValueError(f"Policy requires {state_key}, but no synchronized state is available")
        result[str(state_key)] = observation.state
    image_keys = raw_inputs.get("image_keys", {})
    if not isinstance(image_keys, Mapping):
        raise ValueError("manifest.inputs.image_keys must be a mapping")
    for model_key, source in image_keys.items():
        if not isinstance(source, str) and not isinstance(source, Sequence):
            raise ValueError(f"manifest image source for {model_key} must be a string or string list")
        result[str(model_key)] = _source_frame(observation, source)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path, help="Exported model bundle directory")
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="Bare checkpoint directory; pair with --manifest when manifest.json is not inside it",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Explicit manifest for --checkpoint (manifest paths are resolved separately from weights)",
    )
    parser.add_argument("--config", type=Path, default=None, help="ROS topics/camera YAML or JSON")
    parser.add_argument("--device", default="auto", help="torch device used by the policy")
    parser.add_argument("--publish", action="store_true", help="Publish to the configured relay topic")
    parser.add_argument(
        "--enable-robot",
        action="store_true",
        help="Required second gate for publishing; use only after checking relay wiring",
    )
    parser.add_argument("--once", action="store_true", help="Infer one synchronized frame and exit")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Also append actions to JSONL")
    parser.add_argument(
        "--mcap-config",
        type=Path,
        default=None,
        help="Raw-topic MCAP YAML; required here or as eval config mcap.config",
    )
    parser.add_argument("--mcap-output-root", type=Path, default=None)
    parser.add_argument("--mcap-dataset-name", default=None)
    reward = parser.add_mutually_exclusive_group()
    reward.add_argument(
        "--reward", type=parse_reward, default=None, help="Finite reward for this eval episode"
    )
    reward.add_argument(
        "--prompt-reward",
        action="store_true",
        help="Prompt for a finite reward after normal eval completion",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def _ros_node(config: EvalConfig, cache: EvalObservationCache, *, require_state: bool):
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, JointState
    except ImportError as exc:  # pragma: no cover - requires robot host
        raise RuntimeError(
            "ROS 2 Python packages are unavailable; source the robot ROS environment first"
        ) from exc

    if not rclpy.ok():
        rclpy.init()

    class FrankaDuoEvalNode(Node):
        def __init__(self):
            super().__init__("franka_duo_policy_eval")
            self.create_subscription(
                Image,
                config.cameras["head"].image_topic,
                lambda msg: cache.store_image("head", msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Image,
                config.cameras["head"].depth_topic,
                cache.store_depth,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                CameraInfo,
                config.cameras["head"].camera_info_topic,
                cache.store_camera_info,
                qos_profile_sensor_data,
            )
            for key in ("wrist_left", "wrist_right"):
                self.create_subscription(
                    Image,
                    config.cameras[key].image_topic,
                    lambda msg, key=key: cache.store_image(key, msg),
                    qos_profile_sensor_data,
                )
            if require_state:
                self.create_subscription(
                    JointState,
                    config.topics["joint_states"],
                    cache.store_joint_states,
                    qos_profile_sensor_data,
                )

    return rclpy, FrankaDuoEvalNode()


def _action_record(observation: SynchronizedObservation, action: np.ndarray) -> dict[str, Any]:
    return {
        "schema": "franka_duo_eval_action_v1",
        "stamp_ns": int(observation.stamp_ns),
        "source_stamps_ns": observation.source_stamps_ns,
        "action_layout": "left_xyz_rot6d_rows,right_xyz_rot6d_rows,left_gripper,right_gripper",
        "action": [float(value) for value in action],
    }


def _deduplicate_topics(topics: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(topics))


def _required_eval_input_topics(config: EvalConfig, *, require_state: bool) -> tuple[str, ...]:
    topics: list[str] = []
    for camera in config.cameras.values():
        topics.append(camera.image_topic)
        if camera.depth_topic:
            topics.append(camera.depth_topic)
        if camera.camera_info_topic:
            topics.append(camera.camera_info_topic)
    if require_state:
        topics.append(config.topics["joint_states"])
    return _deduplicate_topics(topics)


def build_eval_mcap_config(
    config: EvalConfig,
    args: argparse.Namespace,
    *,
    require_state: bool,
) -> McapRecorderConfig:
    """Build one mandatory raw capture contract for an eval invocation."""

    mcap_config_path = getattr(args, "mcap_config", None) or config.mcap_config
    if mcap_config_path is None:
        raise ValueError("eval requires a raw MCAP config via --mcap-config or eval YAML mcap.config")
    raw_config = load_mcap_config(Path(mcap_config_path))
    missing = sorted(
        set(_required_eval_input_topics(config, require_state=require_state)) - set(raw_config.topics)
    )
    if missing:
        raise ValueError("Raw MCAP config does not record eval input topics: " + ", ".join(missing))

    direct_reward = getattr(args, "reward", None)
    prompt_for_reward = bool(getattr(args, "prompt_reward", False))
    if direct_reward is not None and prompt_for_reward:
        raise ValueError("--reward and --prompt-reward are mutually exclusive")
    if direct_reward is not None:
        parse_reward(str(direct_reward))

    output_root = getattr(args, "mcap_output_root", None) or config.mcap_output_root or raw_config.output_root
    dataset_name = (
        getattr(args, "mcap_dataset_name", None)
        or config.mcap_dataset_name
        or f"{raw_config.dataset_name}_eval"
    )
    return dataclasses.replace(
        raw_config,
        output_root=Path(output_root),
        dataset_name=str(dataset_name),
        topics=_deduplicate_topics((*raw_config.topics, config.trace_topic)),
        rewarded=direct_reward is not None or prompt_for_reward,
        max_episodes=1,
    )


def _preserve_incomplete_eval(recorder: RawMcapRecorder, *, reason: str) -> None:
    if recorder.active is None:
        return
    try:
        path = recorder.preserve_interrupted_episode(reason=reason)
        LOGGER.warning("Retained incomplete eval capture at %s", path)
    except Exception:
        # RawMcapRecorder retains and marks the bag incomplete before reporting
        # an end-event or rosbag shutdown failure.
        LOGGER.exception("Failed while finalizing incomplete eval capture")


def _wait_for_trace_subscription(publisher: Any, *, timeout_s: float) -> None:
    """Wait until rosbag2 has discovered the eval-owned trace publisher."""

    deadline = time.monotonic() + timeout_s
    while publisher.get_subscription_count() < 1:
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the MCAP recorder to subscribe to trace_topic")
        time.sleep(0.02)


def run(args: argparse.Namespace) -> int:
    config = load_eval_config(args.config)
    checkpoint = args.bundle if args.bundle is not None else args.checkpoint
    if args.manifest is not None and args.bundle is not None:
        raise ValueError("--manifest is only needed with --checkpoint")
    bundle: PolicyBundle = load_policy_bundle(
        checkpoint,
        device=args.device,
        manifest_path=args.manifest,
    )
    action_spec: FrankaDuoActionSpec = bundle.action_spec
    pointcloud: PointCloudConfig = bundle.pointcloud_config
    require_state = bool(getattr(bundle, "requires_state", False))
    if require_state and (config.state_gripper_closed is None or config.state_gripper_open is None):
        raise ValueError(
            "This bundle consumes state; set state_gripper_closed/state_gripper_open in eval config"
        )
    if args.publish and not args.enable_robot:
        raise ValueError("--publish requires the explicit second safety gate --enable-robot")
    if args.publish and not config.publish_topic:
        raise ValueError("publish_topic must be configured explicitly")
    if not config.trace_topic:
        raise ValueError("trace_topic must be configured for MCAP eval provenance")
    if config.trace_topic == config.publish_topic:
        raise ValueError("trace_topic must differ from the robot relay publish_topic")
    if args.publish and action_spec.workspace_min is None:
        raise ValueError("publishing requires manifest workspace_min/workspace_max safety limits")
    max_steps = 1 if args.once else args.max_steps if args.max_steps is not None else config.max_steps
    if max_steps is not None and int(max_steps) <= 0:
        raise ValueError("--max-steps must be positive")
    mcap_config = build_eval_mcap_config(config, args, require_state=require_state)
    ros2_executable = preflight_ros2()
    mcap_recorder = RawMcapRecorder(mcap_config, ros2_executable)
    mcap_recorder.start_episode()
    LOGGER.info("Eval raw MCAP: %s", mcap_recorder.dataset_path)

    rclpy = None
    node = None
    spin_thread = None
    output_handle = None
    try:
        cache = EvalObservationCache(history_size=config.sync_history_size)
        rclpy, node = _ros_node(config, cache, require_state=require_state)
        from std_msgs.msg import String

        trace_publisher = node.create_publisher(String, config.trace_topic, 10)
        publisher = None
        if args.publish:
            from std_msgs.msg import Float32MultiArray

            publisher = node.create_publisher(Float32MultiArray, config.publish_topic, 10)
        spin_thread = threading.Thread(
            target=rclpy.spin, args=(node,), name="franka-eval-ros-spin", daemon=True
        )
        spin_thread.start()
        _wait_for_trace_subscription(
            trace_publisher,
            timeout_s=mcap_config.startup_timeout_s,
        )
        reader = SynchronizedObservationReader(
            cache,
            config.cameras,
            pointcloud,
            rgb_tolerance_ms=config.rgb_match_tolerance_ms,
            depth_tolerance_ms=config.depth_match_tolerance_ms,
            state_tolerance_ms=config.state_match_tolerance_ms,
            max_message_age_ms=config.max_message_age_ms,
            sync_wait_timeout_ms=config.sync_wait_timeout_ms,
            state_gripper_closed=config.state_gripper_closed,
            state_gripper_open=config.state_gripper_open,
        )
        output_file = args.output_jsonl or config.output_jsonl
        output_handle = output_file.open("a", encoding="utf-8") if output_file else None
        bundle.reset()
        steps = 0
        while max_steps is None or steps < max_steps:
            observation = reader.next(timeout_s=1.0, require_state=require_state)
            model_observation = build_policy_observation(observation, bundle.manifest)
            started = time.monotonic()
            action = np.asarray(bundle.predict(model_observation), dtype=np.float32).reshape(-1)
            inference_ms = (time.monotonic() - started) * 1000.0
            if inference_ms > float(config.inference_timeout_ms):
                raise TimeoutError(
                    f"policy inference took {inference_ms:.1f} ms, exceeding {config.inference_timeout_ms:.1f} ms"
                )
            action = action_spec.validate(action)
            record = _action_record(observation, action)
            record["inference_ms"] = inference_ms
            encoded_record = json.dumps(record, separators=(",", ":"), ensure_ascii=True)
            print(encoded_record, flush=True)
            trace_message = String()
            trace_message.data = encoded_record
            trace_publisher.publish(trace_message)
            if output_handle is not None:
                output_handle.write(encoded_record + "\n")
                output_handle.flush()
            if publisher is not None:
                from std_msgs.msg import Float32MultiArray

                message = Float32MultiArray()
                message.data = action.tolist()
                publisher.publish(message)
            steps += 1
        end_requested_unix_ns = time.time_ns()
        reward = getattr(args, "reward", None)
        if bool(getattr(args, "prompt_reward", False)):
            reward = prompt_reward()
        path = mcap_recorder.save_episode(
            reward,
            requested_unix_ns=end_requested_unix_ns,
        )
        LOGGER.info("Saved eval MCAP episode to %s", path)
    except KeyboardInterrupt:
        _preserve_incomplete_eval(mcap_recorder, reason="keyboard_interrupt")
        return 130
    except BaseException:
        _preserve_incomplete_eval(mcap_recorder, reason="eval_error")
        raise
    finally:
        if output_handle is not None:
            output_handle.close()
        with contextlib.suppress(Exception):
            bundle.reset()
        if node is not None:
            with contextlib.suppress(Exception):
                node.destroy_node()
        if rclpy is not None:
            with contextlib.suppress(Exception):
                rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        _preserve_incomplete_eval(mcap_recorder, reason="unexpected_shutdown")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s"
    )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
