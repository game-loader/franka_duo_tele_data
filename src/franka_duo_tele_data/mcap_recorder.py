#!/usr/bin/env python3
"""Record operator-delimited ROS 2 episodes as unmodified MCAP bags.

The recorder deliberately delegates message discovery, serialization, receipt
timestamps, and storage to ``ros2 bag record``.  Python never subscribes to,
decodes, synchronizes, resamples, or aggregates robot and camera messages.
Only one additional ``std_msgs/msg/String`` topic is published for episode
boundary and optional reward annotations.
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
SCHEMA_VERSION = 1


class McapRecorderError(RuntimeError):
    """Raised when raw MCAP capture cannot meet its recording contract."""


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

    @property
    def recorded_topics(self) -> tuple[str, ...]:
        """Return raw topics plus the recorder-owned metadata event topic."""

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
    if not isinstance(value, str) or not value.startswith("/") or any(char.isspace() for char in value):
        raise ValueError(f"{field} must be an absolute ROS topic name without whitespace")
    return value


def validate_config(config: McapRecorderConfig) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config.dataset_name):
        raise ValueError("dataset_name must contain only letters, numbers, '.', '_' and '-'")
    if not config.topics:
        raise ValueError("At least one raw ROS topic must be configured")
    for index, topic in enumerate(config.topics):
        _require_topic_name(topic, field=f"topics[{index}]")
    _require_topic_name(config.event_topic, field="event_topic")
    if config.max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    for name in ("startup_timeout_s", "metadata_publish_timeout_s", "shutdown_timeout_s"):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


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


def stop_process_gracefully(process: Any, *, timeout_s: float) -> int:
    """Stop rosbag2 with SIGINT, escalating only if it does not exit."""

    returncode = process.poll()
    if returncode is not None:
        return int(returncode)
    process.send_signal(signal.SIGINT)
    try:
        return int(process.wait(timeout=timeout_s))
    except subprocess.TimeoutExpired:
        LOGGER.warning("ros2 bag record ignored SIGINT; sending SIGTERM")
        process.terminate()
    try:
        return int(process.wait(timeout=min(5.0, timeout_s)))
    except subprocess.TimeoutExpired:
        LOGGER.error("ros2 bag record ignored SIGTERM; sending SIGKILL")
        process.kill()
        return int(process.wait(timeout=5.0))


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
            "message_handling": {
                "subscription_and_serialization": "rosbag2",
                "online_decoding": False,
                "online_synchronization": False,
                "online_resampling": False,
                "online_aggregation": False,
            },
            "episodes": [],
        }
        self._write_dataset_manifest()

    def _write_dataset_manifest(self) -> None:
        _write_json_atomic(self.dataset_path / DATASET_MANIFEST_NAME, self._manifest)

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
        command = build_record_command(
            self.ros2_executable,
            episode_path,
            self.config.recorded_topics,
        )
        start_requested = requested_unix_ns if requested_unix_ns is not None else self._time_ns()
        process = self._popen_factory(
            command,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
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
            with contextlib.suppress(Exception):
                stop_process_gracefully(process, timeout_s=self.config.shutdown_timeout_s)
            shutil.rmtree(episode_path, ignore_errors=True)
            self.active = None
            raise
        LOGGER.info("Recording episode %d to %s", index, episode_path)
        return active

    def _finish_active(
        self,
        *,
        outcome: str,
        requested_unix_ns: int | None,
        reward: float | None,
        retain: bool,
        intended_complete: bool,
    ) -> Path | None:
        active = self.active
        if active is None:
            raise McapRecorderError("No episode is recording")
        if reward is not None:
            reward = float(reward)
            if not math.isfinite(reward):
                raise ValueError("reward must be finite")
        if intended_complete and self.config.rewarded and reward is None:
            raise ValueError("A finite reward is required before saving this episode")

        end_requested = requested_unix_ns if requested_unix_ns is not None else self._time_ns()
        end_event_time = self._time_ns()
        event_error: BaseException | None = None
        try:
            self._publish(
                build_event_payload(
                    "end",
                    dataset_name=self.config.dataset_name,
                    dataset_version=self.dataset_version,
                    episode_index=active.index,
                    event_unix_ns=end_event_time,
                    requested_unix_ns=end_requested,
                    reward=reward,
                    outcome=outcome,
                )
            )
        except BaseException as exc:
            event_error = exc
        finally:
            returncode = stop_process_gracefully(active.process, timeout_s=self.config.shutdown_timeout_s)
            self.active = None

        process_ok = returncode in (0, -signal.SIGINT)
        complete = intended_complete and event_error is None and process_ok and active.path.is_dir()
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
            "end_event_unix_ns": end_event_time,
            "rosbag_exit_code": returncode,
            "metadata_event_error": str(event_error) if event_error is not None else None,
            "topics": list(self.config.topics),
            "metadata_event_topic": self.config.event_topic,
            "record_command": list(active.command),
        }

        if not retain:
            shutil.rmtree(active.path, ignore_errors=True)
            self._next_episode_index += 1
            if event_error is not None:
                raise McapRecorderError(f"Discarded episode event failed: {event_error}") from event_error
            if not process_ok:
                raise McapRecorderError(f"ros2 bag record exited with code {returncode}")
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
        if event_error is not None:
            raise McapRecorderError(
                f"Episode retained as incomplete because its end event failed: {event_error}"
            ) from event_error
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
    ) -> Path:
        path = self._finish_active(
            outcome="saved",
            requested_unix_ns=requested_unix_ns,
            reward=reward,
            retain=True,
            intended_complete=True,
        )
        assert path is not None
        return path

    def discard_episode(self, *, requested_unix_ns: int | None = None, reason: str = "discarded") -> None:
        self._finish_active(
            outcome=reason,
            requested_unix_ns=requested_unix_ns,
            reward=None,
            retain=False,
            intended_complete=False,
        )

    def preserve_interrupted_episode(self, *, reason: str = "interrupted") -> Path:
        path = self._finish_active(
            outcome=reason,
            requested_unix_ns=self._time_ns(),
            reward=None,
            retain=True,
            intended_complete=False,
        )
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
        help="Prompt for one finite episode reward before publishing the end event",
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
    LOGGER.info("Dataset: %s", recorder.dataset_path)
    LOGGER.info("Idle: r=start, q=quit. Recording: e/s=end+save, d=discard, q=discard+quit.")

    try:
        with TerminalKeys() as keyboard:
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
                    reward = None
                    if config.rewarded:
                        with keyboard.cooked():
                            reward = prompt_reward()
                    path = recorder.save_episode(reward, requested_unix_ns=requested_unix_ns)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
