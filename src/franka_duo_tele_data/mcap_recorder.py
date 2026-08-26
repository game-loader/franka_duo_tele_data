#!/usr/bin/env python3
"""Record operator-delimited ROS 2 episodes as MCAP bags.

Camera, gripper, and TF messages go directly to ``ros2 bag record``.  When arm
sampling is configured, a supervised typed rclpy relay first caps the six
high-rate arm streams by forwarding only the latest unseen sample at 100 Hz.
No stream is synchronized, aggregated, normalized, or passed through FK.
One additional ``std_msgs/msg/String`` topic marks episode start.  On stop,
rosbag2 is closed before any reward prompt; the end boundary and optional
reward are then written to sidecar manifests.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import termios
import time
import tty
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import yaml

LOGGER = logging.getLogger("franka_duo_mcap_recorder")

STORAGE_ID = "mcap"
STORAGE_PRESET_PROFILE = "zstd_fast"
EVENT_MESSAGE_TYPE = "std_msgs/msg/String"
DEFAULT_EVENT_TOPIC = "/franka_duo_tele_data/episode_event"
DATASET_MANIFEST_NAME = "mcap_dataset_manifest.json"
EPISODE_MANIFEST_NAME = "episode_manifest.json"
SCHEMA_VERSION = 3
ARM_RELAY_MODULE = "franka_duo_tele_data.arm_rate_relay"
ARM_RELAY_READY_NAME = ".arm_rate_relay.ready.json"


class McapRecorderError(RuntimeError):
    """Raised when raw MCAP capture cannot meet its recording contract."""


@dataclass(frozen=True, slots=True)
class ArmSamplingRoute:
    """One high-rate arm source and its recorder-only relay destination."""

    source_topic: str
    recorded_topic: str


@dataclass(frozen=True, slots=True)
class ArmSamplingConfig:
    """Lossy latest-unseen rate cap applied only to the six arm streams."""

    rate_hz: float
    routes: tuple[ArmSamplingRoute, ...]


@dataclass(frozen=True, slots=True)
class McapRecorderConfig:
    """Configuration for one versioned raw-MCAP recording session."""

    output_root: Path
    dataset_name: str
    topics: tuple[str, ...]
    event_topic: str = DEFAULT_EVENT_TOPIC
    rewarded: bool = True
    max_episodes: int = 100
    startup_timeout_s: float = 15.0
    metadata_publish_timeout_s: float = 15.0
    shutdown_timeout_s: float = 30.0
    arm_sampling: ArmSamplingConfig | None = None

    @property
    def recorded_topics(self) -> tuple[str, ...]:
        """Return raw topics plus the recorder-owned start-event topic."""

        return _deduplicate((*self.topics, self.event_topic))


@dataclass(slots=True)
class ActiveEpisode:
    index: int
    path: Path
    process: Any
    command: tuple[str, ...]
    start_requested_unix_ns: int
    start_event_unix_ns: int


def _deduplicate(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _require_topic_name(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value == "/"
        or value.endswith("/")
        or "//" in value
        or "=" in value
        or any(char.isspace() for char in value)
    ):
        raise ValueError(f"{field} must be a valid absolute ROS topic name")
    return value


def validate_config(config: McapRecorderConfig) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config.dataset_name):
        raise ValueError("dataset_name must contain only letters, numbers, '.', '_' and '-'")
    if not config.topics:
        raise ValueError("At least one raw ROS topic must be configured")
    for index, topic in enumerate(config.topics):
        _require_topic_name(topic, field=f"topics[{index}]")
    if len(set(config.topics)) != len(config.topics):
        raise ValueError("mcap.topics must be unique")
    _require_topic_name(config.event_topic, field="event_topic")
    if config.event_topic in config.topics:
        raise ValueError("event_topic must not overlap mcap.topics")
    if config.max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    for name in ("startup_timeout_s", "metadata_publish_timeout_s", "shutdown_timeout_s"):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    sampling = config.arm_sampling
    if sampling is None:
        return
    try:
        rate_hz = float(sampling.rate_hz)
    except (TypeError, ValueError) as exc:
        raise ValueError("arm_sampling.rate_hz must be finite and positive") from exc
    if isinstance(sampling.rate_hz, bool) or not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("arm_sampling.rate_hz must be finite and positive")
    if len(sampling.routes) != 6:
        raise ValueError("arm_sampling.routes must contain exactly 6 source-to-recorded routes")
    sources: list[str] = []
    recorded: list[str] = []
    for index, route in enumerate(sampling.routes):
        source = _require_topic_name(
            route.source_topic,
            field=f"arm_sampling.routes[{index}].source_topic",
        )
        destination = _require_topic_name(
            route.recorded_topic,
            field=f"arm_sampling.routes[{index}].recorded_topic",
        )
        if source == destination:
            raise ValueError(f"arm_sampling route {index} source_topic must differ from recorded_topic")
        sources.append(source)
        recorded.append(destination)
    if len(set(sources)) != len(sources):
        raise ValueError("arm_sampling source_topic values must be unique")
    if len(set(recorded)) != len(recorded):
        raise ValueError("arm_sampling recorded_topic values must be unique")
    missing_outputs = sorted(set(recorded).difference(config.topics))
    if missing_outputs:
        raise ValueError(
            "Every arm_sampling recorded_topic must occur in mcap.topics; missing: "
            + ", ".join(missing_outputs)
        )
    directly_recorded_sources = sorted(set(sources).intersection(config.topics))
    if directly_recorded_sources:
        raise ValueError(
            "arm_sampling source topics must not occur directly in mcap.topics: "
            + ", ".join(directly_recorded_sources)
        )
    if config.event_topic in sources:
        raise ValueError("event_topic must not overlap an arm_sampling source_topic")


def _topic_strings(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field} must be a list of absolute ROS topic names")
    return [_require_topic_name(topic, field=f"{field}[{index}]") for index, topic in enumerate(value)]


def configured_raw_topics(data: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract an explicit topic list or the topics from the existing TMR YAML.

    ``mcap.topics`` (or top-level ``mcap_topics``) is authoritative when
    present.  Otherwise, values under top-level ``topics`` and the
    ``image``/``depth``/``camera_info`` entries under ``cameras`` are used.
    This extraction only reads names; it does not import the old synchronizer.
    """

    mcap = data.get("mcap", {})
    if mcap is None:
        mcap = {}
    if not isinstance(mcap, Mapping):
        raise ValueError("mcap must be a mapping")

    if "topics" in mcap:
        topics = _topic_strings(mcap["topics"], field="mcap.topics")
    elif "mcap_topics" in data:
        topics = _topic_strings(data["mcap_topics"], field="mcap_topics")
    else:
        topics = []
        robot_topics = data.get("topics", {})
        if not isinstance(robot_topics, Mapping):
            raise ValueError("topics must be a mapping when mcap.topics is not set")
        for key, topic in robot_topics.items():
            if topic is not None:
                topics.append(_require_topic_name(topic, field=f"topics.{key}"))

        cameras = data.get("cameras", {})
        if not isinstance(cameras, Mapping):
            raise ValueError("cameras must be a mapping when mcap.topics is not set")
        for camera_name, camera in cameras.items():
            if not isinstance(camera, Mapping):
                raise ValueError(f"cameras.{camera_name} must be a mapping")
            for key in ("image", "depth", "camera_info"):
                topic = camera.get(key)
                if topic is not None:
                    topics.append(_require_topic_name(topic, field=f"cameras.{camera_name}.{key}"))

    extra_topics = mcap.get("extra_topics", [])
    if extra_topics:
        topics.extend(_topic_strings(extra_topics, field="mcap.extra_topics"))
    return _deduplicate(topics)


def _arm_sampling_from_mapping(data: Mapping[str, Any]) -> ArmSamplingConfig | None:
    raw = data.get("arm_sampling")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("arm_sampling must be a mapping")
    if isinstance(raw.get("rate_hz"), bool):
        raise ValueError("arm_sampling.rate_hz must be a number")
    try:
        rate_hz = float(raw["rate_hz"])
    except KeyError as exc:
        raise ValueError("arm_sampling.rate_hz is required") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("arm_sampling.rate_hz must be a number") from exc
    raw_routes = raw.get("routes")
    if not isinstance(raw_routes, list | tuple):
        raise ValueError("arm_sampling.routes must be a list")
    routes: list[ArmSamplingRoute] = []
    for index, raw_route in enumerate(raw_routes):
        if not isinstance(raw_route, Mapping):
            raise ValueError(f"arm_sampling.routes[{index}] must be a mapping")
        if "source_topic" not in raw_route or "recorded_topic" not in raw_route:
            raise ValueError(f"arm_sampling.routes[{index}] requires source_topic and recorded_topic")
        routes.append(
            ArmSamplingRoute(
                source_topic=_require_topic_name(
                    raw_route["source_topic"],
                    field=f"arm_sampling.routes[{index}].source_topic",
                ),
                recorded_topic=_require_topic_name(
                    raw_route["recorded_topic"],
                    field=f"arm_sampling.routes[{index}].recorded_topic",
                ),
            )
        )
    return ArmSamplingConfig(rate_hz=rate_hz, routes=tuple(routes))


def load_config(path: Path) -> McapRecorderConfig:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, Mapping):
        raise ValueError(f"Config root must be a mapping: {path}")
    mcap = data.get("mcap", {}) or {}
    if not isinstance(mcap, Mapping):
        raise ValueError("mcap must be a mapping")

    config = McapRecorderConfig(
        output_root=Path(data.get("output_root", "datasets/franka_duo_mcap")),
        dataset_name=str(data.get("dataset_name", "franka_duo_raw")),
        topics=configured_raw_topics(data),
        event_topic=str(mcap.get("event_topic", DEFAULT_EVENT_TOPIC)),
        rewarded=bool(mcap.get("rewarded", data.get("rewarded", True))),
        max_episodes=int(data.get("max_episodes", 100)),
        startup_timeout_s=float(mcap.get("startup_timeout_s", 15.0)),
        metadata_publish_timeout_s=float(mcap.get("metadata_publish_timeout_s", 15.0)),
        shutdown_timeout_s=float(mcap.get("shutdown_timeout_s", 30.0)),
        arm_sampling=_arm_sampling_from_mapping(data),
    )
    validate_config(config)
    return config


def parse_reward(value: str) -> float:
    try:
        reward = float(value.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid reward {value!r}; enter a finite number") from exc
    if not math.isfinite(reward):
        raise ValueError(f"Invalid reward {value!r}; enter a finite number")
    return reward


def prompt_reward(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> float:
    while True:
        try:
            return parse_reward(input_fn("Episode reward (finite scalar): "))
        except ValueError as exc:
            output_fn(f"{exc}. Please try again.")


def _command_output(completed: Any) -> str:
    return f"{getattr(completed, 'stdout', '') or ''}\n{getattr(completed, 'stderr', '') or ''}"


def _run_preflight_command(
    argv: Sequence[str],
    *,
    run_fn: Callable[..., Any],
    timeout_s: float,
) -> Any:
    try:
        completed = run_fn(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise McapRecorderError(f"ROS 2 preflight command failed: {' '.join(argv)}: {exc}") from exc
    if completed.returncode != 0:
        detail = _command_output(completed).strip()
        raise McapRecorderError(
            f"ROS 2 preflight command failed ({completed.returncode}): {' '.join(argv)}"
            + (f"\n{detail}" if detail else "")
        )
    return completed


def preflight_ros2(
    executable: str = "ros2",
    *,
    which_fn: Callable[[str], str | None] = shutil.which,
    run_fn: Callable[..., Any] = subprocess.run,
    timeout_s: float = 10.0,
) -> str:
    """Resolve ROS 2 and verify the MCAP plugin, preset flag, and event type."""

    resolved = which_fn(executable)
    if resolved is None:
        raise McapRecorderError(f"Cannot find {executable!r}; source the ROS 2 installation before recording")

    storage = _run_preflight_command([resolved, "bag", "list", "storage"], run_fn=run_fn, timeout_s=timeout_s)
    if re.search(r"(?:^|[^a-z0-9])mcap(?:$|[^a-z0-9])", _command_output(storage).lower()) is None:
        raise McapRecorderError("The rosbag2 MCAP storage plugin is not installed or not discoverable")

    help_result = _run_preflight_command(
        [resolved, "bag", "record", "--help"], run_fn=run_fn, timeout_s=timeout_s
    )
    if "--storage-preset-profile" not in _command_output(help_result):
        raise McapRecorderError("ros2 bag record does not support --storage-preset-profile")

    _run_preflight_command(
        [resolved, "interface", "show", EVENT_MESSAGE_TYPE], run_fn=run_fn, timeout_s=timeout_s
    )
    return resolved


def build_record_command(
    ros2_executable: str,
    output_path: Path,
    topics: Sequence[str],
) -> list[str]:
    if not topics:
        raise ValueError("At least one topic is required")
    return [
        ros2_executable,
        "bag",
        "record",
        "--storage",
        STORAGE_ID,
        "--storage-preset-profile",
        STORAGE_PRESET_PROFILE,
        "--output",
        str(output_path),
        *topics,
    ]


def build_event_payload(
    event: str,
    *,
    dataset_name: str,
    dataset_version: int,
    episode_index: int,
    event_unix_ns: int,
    requested_unix_ns: int,
    reward: float | None = None,
    outcome: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "franka_duo_tele_data.episode_event.v1",
        "event": event,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "episode_index": episode_index,
        "event_unix_ns": event_unix_ns,
        "requested_unix_ns": requested_unix_ns,
    }
    if outcome is not None:
        payload["outcome"] = outcome
    if reward is not None:
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        payload["reward"] = reward
    return payload


def publish_event(
    ros2_executable: str,
    event_topic: str,
    payload: Mapping[str, Any],
    *,
    run_fn: Callable[..., Any] = subprocess.run,
    timeout_s: float = 15.0,
) -> None:
    """Publish exactly one JSON event encoded in ``std_msgs/String``."""

    event_json = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    message = json.dumps({"data": event_json}, separators=(",", ":"), ensure_ascii=True)
    argv = [ros2_executable, "topic", "pub", "--once", event_topic, EVENT_MESSAGE_TYPE, message]
    try:
        completed = run_fn(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise McapRecorderError(f"Could not publish episode event: {exc}") from exc
    if completed.returncode != 0:
        detail = _command_output(completed).strip()
        raise McapRecorderError(
            f"Could not publish episode event ({completed.returncode})" + (f": {detail}" if detail else "")
        )


def allocate_dataset_version(output_root: Path, dataset_name: str) -> tuple[Path, int]:
    """Atomically allocate the next ``<name>_vN`` directory."""

    output_root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(dataset_name)}_v([0-9]+)$")
    versions = [
        int(match.group(1))
        for entry in output_root.iterdir()
        if entry.is_dir() and (match := pattern.fullmatch(entry.name)) is not None
    ]
    version = max(versions, default=0) + 1
    while True:
        candidate = output_root / f"{dataset_name}_v{version}"
        try:
            candidate.mkdir()
        except FileExistsError:
            version += 1
            continue
        return candidate, version


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def stop_process_gracefully(
    process: Any,
    *,
    timeout_s: float,
    process_name: str = "ros2 bag record",
) -> int:
    """Stop a managed process with SIGINT, escalating only if needed."""

    returncode = process.poll()
    if returncode is not None:
        return int(returncode)
    process.send_signal(signal.SIGINT)
    try:
        return int(process.wait(timeout=timeout_s))
    except subprocess.TimeoutExpired:
        LOGGER.warning("%s ignored SIGINT; sending SIGTERM", process_name)
        process.terminate()
    try:
        return int(process.wait(timeout=min(5.0, timeout_s)))
    except subprocess.TimeoutExpired:
        LOGGER.error("%s ignored SIGTERM; sending SIGKILL", process_name)
        process.kill()
        return int(process.wait(timeout=5.0))


def build_arm_relay_command(
    python_executable: str,
    sampling: ArmSamplingConfig,
    *,
    ready_file: Path,
    startup_timeout_s: float,
) -> list[str]:
    """Build the exact relay subprocess argv shared by manual and eval."""

    try:
        rate_hz = float(sampling.rate_hz)
        startup_timeout = float(startup_timeout_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("Relay rate and startup timeout must be numbers") from exc
    if not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("arm_sampling.rate_hz must be finite and positive")
    if not math.isfinite(startup_timeout) or startup_timeout <= 0:
        raise ValueError("startup_timeout_s must be finite and positive")
    command = [
        python_executable,
        "-m",
        ARM_RELAY_MODULE,
        "--rate-hz",
        str(rate_hz),
        "--startup-timeout",
        str(startup_timeout),
        "--ready-file",
        str(ready_file),
    ]
    for route in sampling.routes:
        command.extend(["--topic", f"{route.source_topic}={route.recorded_topic}"])
    return command


def _validate_relay_ready_payload(
    value: Any,
    sampling: ArmSamplingConfig,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise McapRecorderError("Arm relay ready file must contain a JSON object")
    if value.get("schema_version") != 1:
        raise McapRecorderError("Arm relay ready file has an unsupported schema_version")
    if value.get("status") != "ready":
        raise McapRecorderError("Arm relay ready file does not declare status=ready")
    ready_unix_ns = value.get("ready_unix_ns")
    if isinstance(ready_unix_ns, bool) or not isinstance(ready_unix_ns, int) or ready_unix_ns <= 0:
        raise McapRecorderError("Arm relay ready file has no valid ready_unix_ns")
    if isinstance(value.get("rate_hz"), bool):
        raise McapRecorderError("Arm relay ready file has no valid rate_hz")
    try:
        ready_rate_hz = float(value["rate_hz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise McapRecorderError("Arm relay ready file has no valid rate_hz") from exc
    if not math.isclose(ready_rate_hz, float(sampling.rate_hz), rel_tol=1e-9, abs_tol=0.0):
        raise McapRecorderError(
            f"Arm relay ready rate {ready_rate_hz} does not match configured {sampling.rate_hz}"
        )
    ready_routes = value.get("routes")
    if not isinstance(ready_routes, list) or len(ready_routes) != len(sampling.routes):
        raise McapRecorderError("Arm relay ready file does not contain all configured routes")

    expected = {route.source_topic: route.recorded_topic for route in sampling.routes}
    resolved: dict[str, tuple[str, str]] = {}
    for index, route in enumerate(ready_routes):
        if not isinstance(route, Mapping):
            raise McapRecorderError(f"Arm relay ready route {index} must be a JSON object")
        source = route.get("source_topic")
        destination = route.get("destination_topic")
        message_type = route.get("message_type")
        if not isinstance(source, str) or not isinstance(destination, str):
            raise McapRecorderError(f"Arm relay ready route {index} has invalid topic names")
        if not isinstance(message_type, str) or not message_type.strip():
            raise McapRecorderError(f"Arm relay ready route {index} has no resolved message type")
        if source in resolved:
            raise McapRecorderError(f"Arm relay ready file repeats source topic {source}")
        resolved[source] = (destination, message_type)
    if set(resolved) != set(expected):
        raise McapRecorderError("Arm relay ready sources do not match the configured routes")
    mismatched = [source for source, destination in expected.items() if resolved[source][0] != destination]
    if mismatched:
        raise McapRecorderError(
            "Arm relay ready destinations do not match configuration: " + ", ".join(mismatched)
        )
    return dict(value)


class ArmRateRelayProcess:
    """Supervise one arm relay across any number of MCAP episodes."""

    def __init__(
        self,
        sampling: ArmSamplingConfig,
        *,
        ready_file: Path,
        startup_timeout_s: float,
        shutdown_timeout_s: float,
        python_executable: str = sys.executable,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        monotonic_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.sampling = sampling
        self.ready_file = ready_file
        self.startup_timeout_s = float(startup_timeout_s)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self.python_executable = python_executable
        self._popen_factory = popen_factory
        self._monotonic = monotonic_fn
        self._sleep = sleep_fn
        self.process: Any | None = None
        self.ready_payload: dict[str, Any] | None = None
        self.command = tuple(
            build_arm_relay_command(
                python_executable,
                sampling,
                ready_file=ready_file,
                startup_timeout_s=self.startup_timeout_s,
            )
        )

    def health_error(self) -> str | None:
        if self.process is None:
            return "arm relay has not been started"
        returncode = self.process.poll()
        if returncode is not None:
            return f"arm relay exited with code {returncode}"
        if self.ready_payload is None:
            return "arm relay has not completed its ready handshake"
        return None

    def start(self) -> dict[str, Any]:
        if self.process is not None:
            error = self.health_error()
            if error is not None:
                raise McapRecorderError(error)
            assert self.ready_payload is not None
            return self.ready_payload

        self.ready_file.parent.mkdir(parents=True, exist_ok=True)
        self.ready_file.unlink(missing_ok=True)
        process = self._popen_factory(
            list(self.command),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.process = process
        # Give the relay its full discovery timeout plus a small supervisor
        # margin in which to atomically publish the ready marker.
        deadline = self._monotonic() + self.startup_timeout_s + 1.0
        try:
            while not self.ready_file.is_file():
                returncode = process.poll()
                if returncode is not None:
                    raise McapRecorderError(f"Arm relay exited before becoming ready (code {returncode})")
                if self._monotonic() >= deadline:
                    raise McapRecorderError(f"Timed out waiting for arm relay ready file {self.ready_file}")
                self._sleep(0.02)
            try:
                value = json.loads(self.ready_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise McapRecorderError(f"Cannot read arm relay ready file: {exc}") from exc
            payload = _validate_relay_ready_payload(value, self.sampling)
            returncode = process.poll()
            if returncode is not None:
                raise McapRecorderError(
                    f"Arm relay exited immediately after becoming ready (code {returncode})"
                )
            self.ready_payload = payload
            return payload
        except BaseException:
            with contextlib.suppress(Exception):
                self.stop()
            raise

    def stop(self) -> int | None:
        process = self.process
        self.process = None
        self.ready_payload = None
        try:
            if process is None:
                return None
            return stop_process_gracefully(
                process,
                timeout_s=self.shutdown_timeout_s,
                process_name="arm rate relay",
            )
        finally:
            self.ready_file.unlink(missing_ok=True)

    def __enter__(self) -> ArmRateRelayProcess:
        self.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop()


class RawMcapRecorder:
    """Own a versioned dataset and one active rosbag2 process at a time."""

    def __init__(
        self,
        config: McapRecorderConfig,
        ros2_executable: str,
        *,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        run_fn: Callable[..., Any] = subprocess.run,
        time_ns_fn: Callable[[], int] = time.time_ns,
        monotonic_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        relay_manager: ArmRateRelayProcess | None = None,
    ) -> None:
        validate_config(config)
        self.config = config
        self.ros2_executable = ros2_executable
        self._popen_factory = popen_factory
        self._run_fn = run_fn
        self._time_ns = time_ns_fn
        self._monotonic = monotonic_fn
        self._sleep = sleep_fn
        self.dataset_path, self.dataset_version = allocate_dataset_version(
            config.output_root, config.dataset_name
        )
        self.active: ActiveEpisode | None = None
        self.saved_episodes = 0
        self._next_episode_index = 0
        if relay_manager is not None and config.arm_sampling is None:
            raise ValueError("relay_manager requires config.arm_sampling")
        if relay_manager is not None and relay_manager.sampling != config.arm_sampling:
            raise ValueError("relay_manager sampling contract does not match config.arm_sampling")
        self.arm_relay = relay_manager
        if self.arm_relay is None and config.arm_sampling is not None:
            self.arm_relay = ArmRateRelayProcess(
                config.arm_sampling,
                ready_file=self.dataset_path / ARM_RELAY_READY_NAME,
                startup_timeout_s=config.startup_timeout_s,
                shutdown_timeout_s=config.shutdown_timeout_s,
                popen_factory=popen_factory,
                monotonic_fn=monotonic_fn,
                sleep_fn=sleep_fn,
            )
        arm_sampling_manifest: dict[str, Any] | None = None
        if config.arm_sampling is not None:
            arm_sampling_manifest = {
                "rate_hz": float(config.arm_sampling.rate_hz),
                "selection_policy": "latest_unseen",
                "stale_message_republication": False,
                "lossy": True,
                "source_subscription_qos": "best_effort_keep_last_depth_1",
                "recorded_output_qos": "reliable_keep_last_depth_10",
                "source_payload": "logical_fields_preserved_after_typed_reserialization",
                "serialized_byte_identity": "not_guaranteed",
                "source_header_stamp": "preserved_when_present",
                "bag_receipt_timestamp": "relay_publish_time",
                "source_receipt_timestamp": "not_preserved",
                "routes": [
                    {
                        "source_topic": route.source_topic,
                        "recorded_topic": route.recorded_topic,
                    }
                    for route in config.arm_sampling.routes
                ],
                "runtime": None,
            }
        self._manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "format": "ros2_mcap",
            "storage_id": STORAGE_ID,
            "storage_preset_profile": STORAGE_PRESET_PROFILE,
            "dataset_name": config.dataset_name,
            "dataset_version": self.dataset_version,
            "created_at_unix_ns": self._time_ns(),
            "topics": list(config.topics),
            "metadata_event_topic": config.event_topic,
            "rewarded": config.rewarded,
            "arm_sampling": arm_sampling_manifest,
            "message_handling": {
                "subscription_and_serialization": (
                    "typed_rclpy_arm_relay_then_rosbag2" if config.arm_sampling is not None else "rosbag2"
                ),
                "online_decoding": False,
                "online_typed_deserialization": config.arm_sampling is not None,
                "online_synchronization": False,
                "online_resampling": config.arm_sampling is not None,
                "online_aggregation": False,
                "recording_stop_before_reward": True,
                "reward_storage": "sidecar_manifest_only",
                "metadata_event_scope": "start_only",
            },
            "episodes": [],
        }
        self._write_dataset_manifest()

    def _write_dataset_manifest(self) -> None:
        _write_json_atomic(self.dataset_path / DATASET_MANIFEST_NAME, self._manifest)

    def start_arm_relay(self) -> dict[str, Any] | None:
        """Start once and keep the relay alive across episode boundaries."""

        if self.arm_relay is None:
            return None
        payload = self.arm_relay.start()
        try:
            sampling_manifest = self._manifest["arm_sampling"]
            assert isinstance(sampling_manifest, dict)
            if sampling_manifest["runtime"] is None:
                process = self.arm_relay.process
                sampling_manifest["runtime"] = {
                    "ready": payload,
                    "command": list(self.arm_relay.command),
                    "process_pid": getattr(process, "pid", None),
                    "shutdown_exit_code": None,
                    "stopped_unix_ns": None,
                }
                self._write_dataset_manifest()
        except BaseException:
            with contextlib.suppress(Exception):
                self.close()
            raise
        return payload

    def arm_relay_health_error(self) -> str | None:
        if self.arm_relay is None:
            return None
        return self.arm_relay.health_error()

    def close(self) -> None:
        """Stop the session-wide arm relay; safe to call repeatedly."""

        if self.arm_relay is None or self.arm_relay.process is None:
            return
        returncode = self.arm_relay.stop()
        sampling_manifest = self._manifest["arm_sampling"]
        if isinstance(sampling_manifest, dict) and isinstance(sampling_manifest.get("runtime"), dict):
            sampling_manifest["runtime"]["shutdown_exit_code"] = returncode
            sampling_manifest["runtime"]["stopped_unix_ns"] = self._time_ns()
            self._write_dataset_manifest()

    def _wait_until_started(self, process: Any, episode_path: Path) -> None:
        deadline = self._monotonic() + self.config.startup_timeout_s
        while not episode_path.is_dir():
            returncode = process.poll()
            if returncode is not None:
                raise McapRecorderError(
                    f"ros2 bag record exited before creating the bag directory (code {returncode})"
                )
            if self._monotonic() >= deadline:
                raise McapRecorderError(f"Timed out waiting for ros2 bag record to create {episode_path}")
            self._sleep(0.02)

    def _publish(self, payload: Mapping[str, Any]) -> None:
        publish_event(
            self.ros2_executable,
            self.config.event_topic,
            payload,
            run_fn=self._run_fn,
            timeout_s=self.config.metadata_publish_timeout_s,
        )

    def start_episode(self, *, requested_unix_ns: int | None = None) -> ActiveEpisode:
        if self.active is not None:
            raise McapRecorderError("An episode is already recording")
        index = self._next_episode_index
        episode_path = self.dataset_path / f"episode_{index:06d}"
        if episode_path.exists():
            raise McapRecorderError(f"Episode path already exists: {episode_path}")
        self.start_arm_relay()
        command = build_record_command(
            self.ros2_executable,
            episode_path,
            self.config.recorded_topics,
        )
        start_requested = requested_unix_ns if requested_unix_ns is not None else self._time_ns()
        process = None
        try:
            process = self._popen_factory(
                command,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._wait_until_started(process, episode_path)
            event_time = self._time_ns()
            active = ActiveEpisode(
                index=index,
                path=episode_path,
                process=process,
                command=tuple(command),
                start_requested_unix_ns=start_requested,
                start_event_unix_ns=event_time,
            )
            self.active = active
            self._publish(
                build_event_payload(
                    "start",
                    dataset_name=self.config.dataset_name,
                    dataset_version=self.dataset_version,
                    episode_index=index,
                    event_unix_ns=event_time,
                    requested_unix_ns=start_requested,
                )
            )
        except BaseException:
            if process is not None:
                with contextlib.suppress(Exception):
                    stop_process_gracefully(process, timeout_s=self.config.shutdown_timeout_s)
            shutil.rmtree(episode_path, ignore_errors=True)
            self.active = None
            with contextlib.suppress(Exception):
                self.close()
            raise
        LOGGER.info("Recording episode %d to %s", index, episode_path)
        return active

    def _finish_active(
        self,
        *,
        outcome: str,
        requested_unix_ns: int | None,
        reward: float | None,
        reward_provider: Callable[[], float] | None,
        retain: bool,
        intended_complete: bool,
    ) -> Path | None:
        active = self.active
        if active is None:
            raise McapRecorderError("No episode is recording")
        if reward is not None and reward_provider is not None:
            raise ValueError("Provide either reward or reward_provider, not both")
        if reward is not None:
            reward = float(reward)
            if not math.isfinite(reward):
                raise ValueError("reward must be finite")
        if intended_complete and self.config.rewarded and reward is None and reward_provider is None:
            raise ValueError("A finite reward is required before saving this episode")

        end_requested = requested_unix_ns if requested_unix_ns is not None else self._time_ns()
        returncode = stop_process_gracefully(active.process, timeout_s=self.config.shutdown_timeout_s)
        recording_stopped_unix_ns = self._time_ns()
        self.active = None
        LOGGER.info("Stopped raw capture for episode %d before reward handling", active.index)
        process_ok = returncode in (0, -signal.SIGINT)
        relay_error = self.arm_relay_health_error()
        capture_ok = process_ok and relay_error is None
        reward_error: BaseException | None = None
        reward_recorded_unix_ns: int | None = None
        if intended_complete and capture_ok and active.path.is_dir() and reward_provider is not None:
            try:
                reward = float(reward_provider())
                if not math.isfinite(reward):
                    raise ValueError("reward must be finite")
                reward_recorded_unix_ns = self._time_ns()
            except BaseException as exc:
                reward_error = exc
                outcome = "reward_input_error"
                intended_complete = False
                retain = True
        elif reward is not None:
            reward_recorded_unix_ns = self._time_ns()

        complete = intended_complete and reward_error is None and capture_ok and active.path.is_dir()
        status = "complete" if complete else "incomplete"
        episode_manifest = {
            "schema_version": SCHEMA_VERSION,
            "format": "ros2_mcap",
            "storage_id": STORAGE_ID,
            "storage_preset_profile": STORAGE_PRESET_PROFILE,
            "episode_index": active.index,
            "status": status,
            "outcome": outcome,
            "reward": reward,
            "start_requested_unix_ns": active.start_requested_unix_ns,
            "start_event_unix_ns": active.start_event_unix_ns,
            "end_requested_unix_ns": end_requested,
            "recording_stopped_unix_ns": recording_stopped_unix_ns,
            "reward_recorded_unix_ns": reward_recorded_unix_ns,
            "rosbag_exit_code": returncode,
            "arm_relay_error": relay_error,
            "arm_sampling": self._manifest["arm_sampling"],
            "reward_error": str(reward_error) if reward_error is not None else None,
            "topics": list(self.config.topics),
            "metadata_event_topic": self.config.event_topic,
            "metadata_events_recorded": ["start"],
            "reward_storage": "episode_manifest",
            "record_command": list(active.command),
        }

        if not retain:
            shutil.rmtree(active.path, ignore_errors=True)
            self._next_episode_index += 1
            if not capture_ok:
                detail = relay_error or f"ros2 bag record exited with code {returncode}"
                raise McapRecorderError(detail)
            return None

        if active.path.is_dir():
            _write_json_atomic(active.path / EPISODE_MANIFEST_NAME, episode_manifest)
        self._manifest["episodes"].append(
            {
                "episode_index": active.index,
                "path": active.path.name,
                "status": status,
                "outcome": outcome,
                "reward": reward,
            }
        )
        self._write_dataset_manifest()
        self._next_episode_index += 1
        if complete:
            self.saved_episodes += 1
        if reward_error is not None:
            raise reward_error
        if relay_error is not None:
            raise McapRecorderError(f"Episode retained as incomplete because {relay_error}")
        if not process_ok:
            raise McapRecorderError(
                f"Episode retained as incomplete because ros2 bag record exited with code {returncode}"
            )
        if not active.path.is_dir():
            raise McapRecorderError("ros2 bag record did not create an episode directory")
        return active.path

    def save_episode(
        self,
        reward: float | None,
        *,
        requested_unix_ns: int | None = None,
        reward_provider: Callable[[], float] | None = None,
    ) -> Path:
        try:
            path = self._finish_active(
                outcome="saved",
                requested_unix_ns=requested_unix_ns,
                reward=reward,
                reward_provider=reward_provider,
                retain=True,
                intended_complete=True,
            )
        except BaseException:
            if self.active is None:
                with contextlib.suppress(Exception):
                    self.close()
            raise
        assert path is not None
        if self.saved_episodes >= self.config.max_episodes:
            self.close()
        return path

    def discard_episode(self, *, requested_unix_ns: int | None = None, reason: str = "discarded") -> None:
        self._finish_active(
            outcome=reason,
            requested_unix_ns=requested_unix_ns,
            reward=None,
            reward_provider=None,
            retain=False,
            intended_complete=False,
        )

    def preserve_interrupted_episode(self, *, reason: str = "interrupted") -> Path:
        try:
            path = self._finish_active(
                outcome=reason,
                requested_unix_ns=self._time_ns(),
                reward=None,
                reward_provider=None,
                retain=True,
                intended_complete=False,
            )
        finally:
            with contextlib.suppress(Exception):
                self.close()
        assert path is not None
        return path


class TerminalKeys:
    """Single-key terminal input that can temporarily return to cooked mode."""

    def __init__(self, stream: TextIO = sys.stdin) -> None:
        self.stream = stream
        self.fd = stream.fileno()
        self._original: list[Any] | None = None
        self._active = False

    def start(self) -> None:
        if not self.stream.isatty():
            raise McapRecorderError("Interactive MCAP recording requires a TTY")
        if self._active:
            return
        if self._original is None:
            self._original = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self._active = True

    def pause(self) -> None:
        if self._active and self._original is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._original)
            self._active = False

    def close(self) -> None:
        self.pause()

    @contextlib.contextmanager
    def cooked(self):
        self.pause()
        try:
            yield
        finally:
            self.start()

    def read(self, timeout_s: float = 0.1) -> str | None:
        readable, _, _ = select.select([self.fd], [], [], timeout_s)
        if not readable:
            return None
        return os.read(self.fd, 1).decode(errors="ignore").lower()

    def __enter__(self) -> TerminalKeys:
        self.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="TMR or raw-MCAP YAML config")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--topic", action="append", dest="topics", help="Replace config topics; repeatable")
    parser.add_argument("--event-topic", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--rewarded",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop rosbag2 first, then prompt for a finite reward stored in the sidecar manifest",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def _apply_cli_overrides(config: McapRecorderConfig, args: argparse.Namespace) -> McapRecorderConfig:
    overridden = McapRecorderConfig(
        output_root=args.output_root if args.output_root is not None else config.output_root,
        dataset_name=args.dataset_name if args.dataset_name is not None else config.dataset_name,
        topics=tuple(args.topics) if args.topics is not None else config.topics,
        event_topic=args.event_topic if args.event_topic is not None else config.event_topic,
        rewarded=args.rewarded if args.rewarded is not None else config.rewarded,
        max_episodes=args.max_episodes if args.max_episodes is not None else config.max_episodes,
        startup_timeout_s=config.startup_timeout_s,
        metadata_publish_timeout_s=config.metadata_publish_timeout_s,
        shutdown_timeout_s=config.shutdown_timeout_s,
        arm_sampling=config.arm_sampling,
    )
    validate_config(overridden)
    return overridden


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s"
    )
    config = _apply_cli_overrides(load_config(args.config), args)
    ros2_executable = preflight_ros2()
    recorder = RawMcapRecorder(config, ros2_executable)

    try:
        recorder.start_arm_relay()
        LOGGER.info("Dataset: %s", recorder.dataset_path)
        LOGGER.info("Idle: r=start, q=quit. Recording: e/s=end+save, d=discard, q=discard+quit.")
        with TerminalKeys() as keyboard:

            def prompt_in_cooked_mode() -> float:
                with keyboard.cooked():
                    return prompt_reward()

            quit_requested = False
            while recorder.saved_episodes < config.max_episodes and not quit_requested:
                key = keyboard.read()
                if key is None:
                    continue
                requested_unix_ns = time.time_ns()
                if recorder.active is None:
                    if key == "q":
                        quit_requested = True
                    elif key == "r":
                        recorder.start_episode(requested_unix_ns=requested_unix_ns)
                    continue

                if key in {"e", "s"}:
                    reward_provider = prompt_in_cooked_mode if config.rewarded else None
                    path = recorder.save_episode(
                        None,
                        requested_unix_ns=requested_unix_ns,
                        reward_provider=reward_provider,
                    )
                    LOGGER.info("Saved episode to %s", path)
                elif key == "d":
                    recorder.discard_episode(requested_unix_ns=requested_unix_ns)
                    LOGGER.info("Discarded episode")
                elif key == "q":
                    recorder.discard_episode(requested_unix_ns=requested_unix_ns, reason="quit")
                    quit_requested = True
    except KeyboardInterrupt:
        if recorder.active is not None:
            with contextlib.suppress(Exception):
                path = recorder.preserve_interrupted_episode(reason="keyboard_interrupt")
                LOGGER.warning("Retained interrupted episode at %s", path)
        return 130
    finally:
        if recorder.active is not None:
            with contextlib.suppress(Exception):
                recorder.preserve_interrupted_episode(reason="unexpected_shutdown")
        with contextlib.suppress(Exception):
            recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
