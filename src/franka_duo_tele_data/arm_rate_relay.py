#!/usr/bin/env python3
"""Rate-limit high-frequency Franka state topics with a typed rclpy relay.

The relay discovers each source topic's ROS message type from the live graph,
then republishes the newest not-yet-forwarded message at most once per timer
period.  rclpy deserializes each source message and serializes it again for the
output publisher; the relay does not interpret or modify its fields, so an
existing ``header.stamp`` is preserved.  The output topic has a new DDS/rosbag
receipt timestamp because it is a new publication.

ROS imports stay inside runtime helpers so configuration and buffer behavior
can be tested on machines without ROS 2 installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("franka_duo_arm_rate_relay")

DEFAULT_RATE_HZ = 100.0
DEFAULT_STARTUP_TIMEOUT_S = 15.0
DEFAULT_OUTPUT_PREFIX = "/franka_duo_tele_data/rate100"
DEFAULT_NODE_NAME = "franka_duo_arm_rate_relay"
_BROADCASTER = "franka_robot_state_broadcaster"
_ARM_STREAMS = (
    "current_pose",
    "desired_joint_states",
    "measured_joint_states",
    "desired_end_effector_twist",
)
DEFAULT_SOURCE_TOPICS = tuple(
    f"/{side}/{_BROADCASTER}/{stream}" for side in ("left", "right") for stream in _ARM_STREAMS
)


class ArmRateRelayError(RuntimeError):
    """Raised when the relay cannot establish its live ROS topic contract."""


@dataclass(frozen=True, slots=True)
class RelayTopic:
    """One source-to-output relay route."""

    source: str
    output: str


@dataclass(frozen=True, slots=True)
class TopicDiscovery:
    """Result of comparing required sources with one ROS graph snapshot."""

    resolved: Mapping[str, str]
    missing: tuple[str, ...]
    ambiguous: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class RelayStats:
    received: int
    published: int
    superseded: int


def _require_absolute_topic(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value == "/"
        or value.endswith("/")
        or "//" in value
        or "=" in value
        or re.search(r"\s", value)
    ):
        raise ValueError(f"{field} must be an absolute ROS topic name without whitespace")
    return value


def output_topic_for(source: str, prefix: str = DEFAULT_OUTPUT_PREFIX) -> str:
    """Build the default recorder-only output topic for a source topic."""

    source = _require_absolute_topic(source, field="source")
    prefix = _require_absolute_topic(prefix, field="output_prefix")
    return f"{prefix}{source}"


def build_topic_routes(
    sources: Sequence[str] = DEFAULT_SOURCE_TOPICS,
    *,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
    output_overrides: Mapping[str, str] | None = None,
) -> tuple[RelayTopic, ...]:
    """Validate sources and produce deterministic source/output routes.

    ``output_overrides`` is suitable for a mapping loaded from YAML.  Every
    override key must also occur in ``sources``; unspecified outputs are
    derived by prefixing the complete source topic path.
    """

    output_prefix = _require_absolute_topic(output_prefix, field="output_prefix")
    source_topics = tuple(_require_absolute_topic(topic, field="source") for topic in sources)
    if not source_topics:
        raise ValueError("At least one source topic is required")
    if len(set(source_topics)) != len(source_topics):
        raise ValueError("Source topics must be unique")

    overrides = dict(output_overrides or {})
    unknown = set(overrides).difference(source_topics)
    if unknown:
        raise ValueError(f"Output overrides contain unknown sources: {sorted(unknown)}")

    routes = tuple(
        RelayTopic(
            source=source,
            output=_require_absolute_topic(
                overrides.get(source, output_topic_for(source, output_prefix)),
                field=f"output for {source}",
            ),
        )
        for source in source_topics
    )
    outputs = tuple(route.output for route in routes)
    if len(set(outputs)) != len(outputs):
        raise ValueError("Output topics must be unique")
    overlap = set(source_topics).intersection(outputs)
    if overlap:
        raise ValueError(f"Relay outputs must not overlap source topics: {sorted(overlap)}")
    return routes


def inspect_topic_types(
    names_and_types: Mapping[str, Sequence[str]] | Iterable[tuple[str, Sequence[str]]],
    source_topics: Sequence[str],
) -> TopicDiscovery:
    """Inspect one graph snapshot without guessing any message type."""

    entries = names_and_types.items() if isinstance(names_and_types, Mapping) else names_and_types
    graph: dict[str, set[str]] = {}
    for topic, message_types in entries:
        graph.setdefault(str(topic), set()).update(str(value) for value in message_types)

    resolved: dict[str, str] = {}
    missing: list[str] = []
    ambiguous: dict[str, tuple[str, ...]] = {}
    for topic in source_topics:
        types = tuple(sorted(graph.get(topic, ())))
        if not types:
            missing.append(topic)
        elif len(types) > 1:
            ambiguous[topic] = types
        else:
            resolved[topic] = types[0]
    return TopicDiscovery(resolved=resolved, missing=tuple(missing), ambiguous=ambiguous)


def wait_for_topic_types(
    node: Any,
    source_topics: Sequence[str],
    *,
    timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
    poll_interval_s: float = 0.1,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    """Wait for one type and at least one active publisher per source topic."""

    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be finite and positive")

    deadline = monotonic_fn() + timeout_s
    last = inspect_topic_types((), source_topics)
    no_publishers: tuple[str, ...] = ()
    while True:
        last = inspect_topic_types(node.get_topic_names_and_types(), source_topics)
        no_publishers = tuple(topic for topic in last.resolved if int(node.count_publishers(topic)) <= 0)
        if not last.missing and not last.ambiguous and not no_publishers:
            return dict(last.resolved)
        remaining = deadline - monotonic_fn()
        if remaining <= 0:
            problems: list[str] = []
            if last.missing:
                problems.append("missing=" + ",".join(last.missing))
            if last.ambiguous:
                details = ",".join(f"{topic}={list(types)}" for topic, types in last.ambiguous.items())
                problems.append("multiple_types=" + details)
            if no_publishers:
                problems.append("no_publishers=" + ",".join(no_publishers))
            raise ArmRateRelayError(
                "Timed out waiting for every source topic to advertise one unique ROS message type "
                "and at least one publisher: " + "; ".join(problems)
            )
        sleep_fn(min(poll_interval_s, remaining))


class LatestSampleBuffer:
    """Thread-safe one-element buffer with explicit superseded-sample stats."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Any | None = None
        self._sequence = 0
        self._consumed_sequence = 0
        self._received = 0
        self._published = 0
        self._superseded = 0

    def put(self, message: Any) -> None:
        with self._lock:
            if self._sequence > self._consumed_sequence:
                self._superseded += 1
            self._latest = message
            self._sequence += 1
            self._received += 1

    def take_latest(self) -> Any | None:
        """Consume the newest unseen sample, or return ``None``.

        Consuming before ROS publication avoids duplicate output from another
        timer tick.  A publication exception is treated as a fatal runtime
        error by :class:`ArmRateRelay`, rather than replaying an uncertain
        sample.
        """

        with self._lock:
            if self._sequence == self._consumed_sequence:
                return None
            message = self._latest
            self._consumed_sequence = self._sequence
            return message

    def mark_published(self) -> None:
        with self._lock:
            self._published += 1

    def stats(self) -> RelayStats:
        with self._lock:
            return RelayStats(
                received=self._received,
                published=self._published,
                superseded=self._superseded,
            )


@dataclass(slots=True)
class _RelayChannel:
    route: RelayTopic
    message_type: str
    publisher: Any
    subscription: Any
    buffer: LatestSampleBuffer


def sensor_data_qos() -> Any:
    """Create a depth-one best-effort QoS for the high-rate sources."""

    try:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:  # pragma: no cover - requires ROS host
        raise ArmRateRelayError("rclpy is unavailable; source the ROS 2 environment") from exc
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def reliable_output_qos() -> Any:
    """Create a reliable QoS so selected 100 Hz samples reach rosbag2."""

    try:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:  # pragma: no cover - requires ROS host
        raise ArmRateRelayError("rclpy is unavailable; source the ROS 2 environment") from exc
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def load_ros_message_type(message_type: str) -> type[Any]:
    """Load a graph-discovered namespaced ROS message type."""

    try:
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:  # pragma: no cover - requires ROS host
        raise ArmRateRelayError("rosidl_runtime_py is unavailable; source the ROS 2 environment") from exc
    try:
        return get_message(message_type)
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError) as exc:
        raise ArmRateRelayError(f"Cannot load ROS message type {message_type!r}") from exc


class ArmRateRelay:
    """ROS-node adapter that relays each newest unseen source at a fixed cap."""

    def __init__(
        self,
        node: Any,
        routes: Sequence[RelayTopic],
        topic_types: Mapping[str, str],
        *,
        rate_hz: float = DEFAULT_RATE_HZ,
        qos: Any | None = None,
        message_loader: Callable[[str], type[Any]] = load_ros_message_type,
    ) -> None:
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("rate_hz must be finite and positive")
        missing = [route.source for route in routes if route.source not in topic_types]
        if missing:
            raise ValueError(f"Missing resolved message types for: {missing}")

        self._node = node
        self._channels: list[_RelayChannel] = []
        source_qos = sensor_data_qos() if qos is None else qos
        output_qos = reliable_output_qos() if qos is None else qos
        for route in routes:
            message_type = topic_types[route.source]
            message_class = message_loader(message_type)
            publisher = node.create_publisher(message_class, route.output, output_qos)
            buffer = LatestSampleBuffer()
            subscription = node.create_subscription(
                message_class,
                route.source,
                buffer.put,
                source_qos,
            )
            self._channels.append(
                _RelayChannel(
                    route=route,
                    message_type=message_type,
                    publisher=publisher,
                    subscription=subscription,
                    buffer=buffer,
                )
            )
        self._timer = node.create_timer(1.0 / rate_hz, self._publish_latest)

    def _publish_latest(self) -> None:
        for channel in self._channels:
            message = channel.buffer.take_latest()
            if message is None:
                continue
            channel.publisher.publish(message)
            channel.buffer.mark_published()

    def stats(self) -> dict[str, RelayStats]:
        return {channel.route.source: channel.buffer.stats() for channel in self._channels}


def build_ready_payload(
    routes: Sequence[RelayTopic],
    topic_types: Mapping[str, str],
    *,
    rate_hz: float,
    node_name: str,
    ready_unix_ns: int | None = None,
) -> dict[str, Any]:
    """Build the supervisor handshake written after the full relay is live."""

    if not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be finite and positive")
    missing = [route.source for route in routes if route.source not in topic_types]
    if missing:
        raise ValueError(f"Missing resolved message types for: {missing}")
    timestamp = time.time_ns() if ready_unix_ns is None else int(ready_unix_ns)
    if timestamp <= 0:
        raise ValueError("ready_unix_ns must be positive")
    return {
        "schema_version": 1,
        "status": "ready",
        "ready_unix_ns": timestamp,
        "node_name": str(node_name),
        "rate_hz": float(rate_hz),
        "routes": [
            {
                "source_topic": route.source,
                "destination_topic": route.output,
                "message_type": topic_types[route.source],
            }
            for route in routes
        ],
    }


def write_ready_file(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a ready JSON file in its destination directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def remove_ready_file(path: Path | None) -> None:
    if path is not None:
        Path(path).unlink(missing_ok=True)


def _parse_topic_arg(value: str) -> tuple[str, str | None]:
    source, separator, output = value.partition("=")
    source = _require_absolute_topic(source, field="--topic source")
    if not separator:
        return source, None
    if not output:
        raise argparse.ArgumentTypeError("--topic SOURCE=OUTPUT requires a non-empty OUTPUT")
    return source, _require_absolute_topic(output, field="--topic output")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relay the newest Franka arm messages at no more than 100 Hz by default"
    )
    parser.add_argument(
        "--topic",
        action="append",
        type=_parse_topic_arg,
        metavar="SOURCE[=OUTPUT]",
        help="Replace defaults with a source topic and optional explicit output; repeat per route",
    )
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument(
        "--startup-timeout",
        "--startup-timeout-s",
        "--discovery-timeout-s",
        dest="startup_timeout_s",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_S,
        help="Seconds to wait for every source topic to advertise one unique type and active publisher",
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        help="Atomically written as JSON only after all routes and publishers are ready",
    )
    parser.add_argument("--node-name", default=DEFAULT_NODE_NAME)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def routes_from_args(args: argparse.Namespace) -> tuple[RelayTopic, ...]:
    if args.topic:
        sources = tuple(source for source, _output in args.topic)
        overrides = {source: output for source, output in args.topic if output is not None}
    else:
        sources = DEFAULT_SOURCE_TOPICS
        overrides = {}
    return build_topic_routes(
        sources,
        output_prefix=args.output_prefix,
        output_overrides=overrides,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    routes = routes_from_args(args)
    remove_ready_file(args.ready_file)

    try:
        import rclpy
    except ImportError as exc:  # pragma: no cover - requires ROS host
        raise ArmRateRelayError("rclpy is unavailable; source the ROS 2 environment") from exc

    rclpy.init(args=[])
    node = rclpy.create_node(args.node_name)
    relay: ArmRateRelay | None = None
    try:
        topic_types = wait_for_topic_types(
            node,
            tuple(route.source for route in routes),
            timeout_s=args.startup_timeout_s,
        )
        relay = ArmRateRelay(node, routes, topic_types, rate_hz=args.rate_hz)
        for route in routes:
            LOGGER.info(
                "%s [%s] -> %s at <= %.3f Hz",
                route.source,
                topic_types[route.source],
                route.output,
                args.rate_hz,
            )
        if args.ready_file is not None:
            write_ready_file(
                args.ready_file,
                build_ready_payload(
                    routes,
                    topic_types,
                    rate_hz=args.rate_hz,
                    node_name=args.node_name,
                ),
            )
            LOGGER.info("Relay ready marker written to %s", args.ready_file)
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover - interactive ROS runtime
        pass
    finally:
        if relay is not None:
            for topic, stats in relay.stats().items():
                LOGGER.info(
                    "%s: received=%d published=%d superseded=%d",
                    topic,
                    stats.received,
                    stats.published,
                    stats.superseded,
                )
        remove_ready_file(args.ready_file)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
