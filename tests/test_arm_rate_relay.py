from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from franka_duo_tele_data.arm_rate_relay import (
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_SOURCE_TOPICS,
    ArmRateRelay,
    ArmRateRelayError,
    LatestSampleBuffer,
    RelayStats,
    build_arg_parser,
    build_ready_payload,
    build_topic_routes,
    inspect_topic_types,
    output_topic_for,
    remove_ready_file,
    routes_from_args,
    wait_for_topic_types,
    write_ready_file,
)


def test_default_sources_cover_arm_and_gripper_streams() -> None:
    assert DEFAULT_SOURCE_TOPICS == (
        "/left/franka_robot_state_broadcaster/current_pose",
        "/left/franka_robot_state_broadcaster/measured_joint_states",
        "/right/franka_robot_state_broadcaster/current_pose",
        "/right/franka_robot_state_broadcaster/measured_joint_states",
        "/left/gripper/joint_states",
        "/right/gripper/joint_states",
    )


def test_default_output_keeps_complete_source_suffix() -> None:
    source = "/left/franka_robot_state_broadcaster/current_pose"
    assert output_topic_for(source) == f"{DEFAULT_OUTPUT_PREFIX}{source}"


def test_build_routes_accepts_explicit_yaml_style_overrides() -> None:
    routes = build_topic_routes(
        ("/left/current", "/right/current"),
        output_overrides={"/left/current": "/record/left_pose"},
    )
    assert [(route.source, route.output) for route in routes] == [
        ("/left/current", "/record/left_pose"),
        ("/right/current", f"{DEFAULT_OUTPUT_PREFIX}/right/current"),
    ]


@pytest.mark.parametrize(
    ("sources", "overrides", "match"),
    [
        (("/same", "/same"), {}, "unique"),
        (("/source",), {"/unknown": "/output"}, "unknown"),
        (("/a", "/b"), {"/a": "/out", "/b": "/out"}, "unique"),
        (("/source",), {"/source": "/source"}, "overlap"),
    ],
)
def test_build_routes_rejects_ambiguous_or_cyclic_mappings(sources, overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        build_topic_routes(sources, output_overrides=overrides)


def test_topic_discovery_reports_resolved_missing_and_ambiguous() -> None:
    discovery = inspect_topic_types(
        [
            ("/ready", ["sensor_msgs/msg/JointState"]),
            ("/ambiguous", ["custom/msg/A", "custom/msg/B"]),
            ("/ready", ["sensor_msgs/msg/JointState"]),
        ],
        ("/ready", "/missing", "/ambiguous"),
    )
    assert discovery.resolved == {"/ready": "sensor_msgs/msg/JointState"}
    assert discovery.missing == ("/missing",)
    assert discovery.ambiguous == {"/ambiguous": ("custom/msg/A", "custom/msg/B")}


def test_wait_for_topic_types_polls_until_every_source_exists() -> None:
    snapshots = iter(
        [
            [("/left", ["sensor_msgs/msg/JointState"])],
            [
                ("/left", ["sensor_msgs/msg/JointState"]),
                ("/right", ["custom_msgs/msg/FrankaState"]),
            ],
        ]
    )
    node = SimpleNamespace(
        get_topic_names_and_types=lambda: next(snapshots),
        count_publishers=lambda _topic: 1,
    )
    ticks = iter((0.0, 0.0, 0.1))
    sleeps: list[float] = []

    result = wait_for_topic_types(
        node,
        ("/left", "/right"),
        timeout_s=1.0,
        monotonic_fn=lambda: next(ticks),
        sleep_fn=sleeps.append,
    )

    assert result == {
        "/left": "sensor_msgs/msg/JointState",
        "/right": "custom_msgs/msg/FrankaState",
    }
    assert sleeps == [0.1]


def test_wait_for_topic_types_rejects_multiple_advertised_types_at_timeout() -> None:
    node = SimpleNamespace(
        get_topic_names_and_types=lambda: [("/arm", ["pkg/msg/A", "pkg/msg/B"])],
        count_publishers=lambda _topic: 1,
    )
    ticks = iter((0.0, 1.0))
    with pytest.raises(ArmRateRelayError, match="multiple_types=/arm"):
        wait_for_topic_types(node, ("/arm",), timeout_s=1.0, monotonic_fn=lambda: next(ticks))


def test_wait_for_topic_types_reports_missing_topics_at_timeout() -> None:
    node = SimpleNamespace(get_topic_names_and_types=lambda: [], count_publishers=lambda _topic: 0)
    ticks = iter((0.0, 2.0))
    with pytest.raises(ArmRateRelayError, match=r"missing=/left,/right"):
        wait_for_topic_types(
            node,
            ("/left", "/right"),
            timeout_s=1.0,
            monotonic_fn=lambda: next(ticks),
        )


def test_wait_for_topic_types_waits_for_an_active_publisher() -> None:
    publisher_counts = iter((0, 1))
    node = SimpleNamespace(
        get_topic_names_and_types=lambda: [("/arm", ["pkg/msg/State"])],
        count_publishers=lambda _topic: next(publisher_counts),
    )
    ticks = iter((0.0, 0.0, 0.1))
    sleeps: list[float] = []

    result = wait_for_topic_types(
        node,
        ("/arm",),
        timeout_s=1.0,
        monotonic_fn=lambda: next(ticks),
        sleep_fn=sleeps.append,
    )

    assert result == {"/arm": "pkg/msg/State"}
    assert sleeps == [0.1]


def test_wait_for_topic_types_reports_a_type_without_publishers_at_timeout() -> None:
    node = SimpleNamespace(
        get_topic_names_and_types=lambda: [("/arm", ["pkg/msg/State"])],
        count_publishers=lambda _topic: 0,
    )
    ticks = iter((0.0, 1.0))
    with pytest.raises(ArmRateRelayError, match="no_publishers=/arm"):
        wait_for_topic_types(node, ("/arm",), timeout_s=1.0, monotonic_fn=lambda: next(ticks))


def test_latest_sample_buffer_only_returns_latest_unseen_message() -> None:
    buffer = LatestSampleBuffer()
    first = object()
    latest = object()
    buffer.put(first)
    buffer.put(latest)

    assert buffer.take_latest() is latest
    buffer.mark_published()
    assert buffer.take_latest() is None
    assert buffer.stats() == RelayStats(received=2, published=1, superseded=1)


class _FakePublisher:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.messages: list[object] = []

    def publish(self, message: object) -> None:
        self.messages.append(message)


class _FakeNode:
    def __init__(self) -> None:
        self.publishers: dict[str, _FakePublisher] = {}
        self.callbacks: dict[str, object] = {}
        self.publisher_qos: dict[str, object] = {}
        self.subscription_qos: dict[str, object] = {}
        self.timer_period: float | None = None
        self.timer_callback = None

    def create_publisher(self, _message_class, topic, qos):
        publisher = _FakePublisher(topic)
        self.publishers[topic] = publisher
        self.publisher_qos[topic] = qos
        return publisher

    def create_subscription(self, _message_class, topic, callback, qos):
        self.callbacks[topic] = callback
        self.subscription_qos[topic] = qos
        return SimpleNamespace(topic=topic)

    def create_timer(self, period, callback):
        self.timer_period = period
        self.timer_callback = callback
        return SimpleNamespace(period=period)


def test_arm_relay_publishes_latest_at_most_once_per_timer_tick() -> None:
    routes = build_topic_routes(("/left/state", "/right/state"))
    topic_types = {"/left/state": "pkg/msg/Left", "/right/state": "pkg/msg/Right"}
    loaded: list[str] = []

    def load_message(message_type: str):
        loaded.append(message_type)
        return type(message_type.replace("/", "_"), (), {})

    node = _FakeNode()
    relay = ArmRateRelay(
        node,
        routes,
        topic_types,
        rate_hz=100.0,
        qos=object(),
        message_loader=load_message,
    )
    old_message = SimpleNamespace(header=SimpleNamespace(stamp=123))
    latest_message = SimpleNamespace(header=SimpleNamespace(stamp=456))
    node.callbacks["/left/state"](old_message)
    node.callbacks["/left/state"](latest_message)

    assert node.timer_period == pytest.approx(0.01)
    node.timer_callback()
    output = node.publishers[f"{DEFAULT_OUTPUT_PREFIX}/left/state"]
    assert output.messages == [latest_message]
    assert output.messages[0].header.stamp == 456

    node.timer_callback()
    assert output.messages == [latest_message]
    assert loaded == ["pkg/msg/Left", "pkg/msg/Right"]
    assert relay.stats()["/left/state"] == RelayStats(received=2, published=1, superseded=1)


def test_arm_relay_uses_latest_source_qos_and_reliable_recorded_output(monkeypatch) -> None:
    import franka_duo_tele_data.arm_rate_relay as relay_module

    source_qos = object()
    output_qos = object()
    monkeypatch.setattr(relay_module, "sensor_data_qos", lambda: source_qos)
    monkeypatch.setattr(relay_module, "reliable_output_qos", lambda: output_qos)
    node = _FakeNode()
    route = build_topic_routes(("/left/state",))[0]

    ArmRateRelay(
        node,
        (route,),
        {route.source: "pkg/msg/State"},
        message_loader=lambda _message_type: object,
    )

    assert node.subscription_qos[route.source] is source_qos
    assert node.publisher_qos[route.output] is output_qos


def test_cli_topic_arguments_replace_defaults_and_allow_explicit_output() -> None:
    args = build_arg_parser().parse_args(
        ["--topic", "/left/current", "--topic", "/right/current=/record/right", "--rate-hz", "50"]
    )
    routes = routes_from_args(args)
    assert [(route.source, route.output) for route in routes] == [
        ("/left/current", f"{DEFAULT_OUTPUT_PREFIX}/left/current"),
        ("/right/current", "/record/right"),
    ]
    assert args.rate_hz == 50.0


def test_ready_payload_contains_every_resolved_route_and_runtime_contract() -> None:
    routes = build_topic_routes(("/left/current", "/right/current"))
    types = {"/left/current": "pkg/msg/Pose", "/right/current": "pkg/msg/Pose"}

    payload = build_ready_payload(
        routes,
        types,
        rate_hz=100.0,
        node_name="relay",
        ready_unix_ns=123,
    )

    assert payload == {
        "schema_version": 1,
        "status": "ready",
        "ready_unix_ns": 123,
        "node_name": "relay",
        "rate_hz": 100.0,
        "routes": [
            {
                "source_topic": "/left/current",
                "destination_topic": f"{DEFAULT_OUTPUT_PREFIX}/left/current",
                "message_type": "pkg/msg/Pose",
            },
            {
                "source_topic": "/right/current",
                "destination_topic": f"{DEFAULT_OUTPUT_PREFIX}/right/current",
                "message_type": "pkg/msg/Pose",
            },
        ],
    }


def test_ready_file_is_valid_json_and_replaces_stale_marker(tmp_path) -> None:
    ready_file = tmp_path / "runtime" / "relay-ready.json"
    ready_file.parent.mkdir()
    ready_file.write_text("stale", encoding="utf-8")

    write_ready_file(ready_file, {"status": "ready", "routes": []})

    assert json.loads(ready_file.read_text(encoding="utf-8")) == {"status": "ready", "routes": []}
    assert list(ready_file.parent.glob("*.tmp")) == []
    remove_ready_file(ready_file)
    assert not ready_file.exists()


def test_cli_supports_startup_timeout_alias_and_ready_file(tmp_path) -> None:
    ready_file = tmp_path / "ready.json"
    args = build_arg_parser().parse_args(["--startup-timeout", "3.5", "--ready-file", str(ready_file)])
    assert args.startup_timeout_s == 3.5
    assert args.ready_file == ready_file
