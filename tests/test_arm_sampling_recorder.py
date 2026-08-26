from __future__ import annotations

import dataclasses
import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from franka_duo_tele_data.mcap_recorder import (
    ARM_RELAY_READY_NAME,
    DATASET_MANIFEST_NAME,
    EPISODE_MANIFEST_NAME,
    ArmRateRelayProcess,
    ArmSamplingConfig,
    ArmSamplingRoute,
    McapRecorderConfig,
    McapRecorderError,
    RawMcapRecorder,
    load_config,
    validate_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Process:
    def __init__(self, *, final_returncode: int = 0, running: bool = True) -> None:
        self.returncode = None if running else final_returncode
        self.final_returncode = final_returncode
        self.signals: list[int] = []
        self.terminated = False
        self.killed = False
        self.pid = 1234

    def poll(self):
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)

    def wait(self, timeout):
        del timeout
        self.returncode = self.final_returncode
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def _sampling() -> ArmSamplingConfig:
    routes = tuple(
        ArmSamplingRoute(
            source_topic=f"/{side}/source/{index}",
            recorded_topic=f"/rate100/{side}/{index}",
        )
        for side in ("left", "right")
        for index in range(2)
    )
    return ArmSamplingConfig(rate_hz=100.0, routes=routes)


def _config(tmp_path: Path, *, max_episodes: int = 2) -> McapRecorderConfig:
    sampling = _sampling()
    return McapRecorderConfig(
        output_root=tmp_path,
        dataset_name="arm_capture",
        topics=tuple(route.recorded_topic for route in sampling.routes) + ("/head/rgb",),
        rewarded=False,
        max_episodes=max_episodes,
        arm_sampling=sampling,
    )


def _ready_payload(sampling: ArmSamplingConfig) -> dict:
    return {
        "schema_version": 1,
        "status": "ready",
        "ready_unix_ns": 123,
        "node_name": "relay",
        "rate_hz": sampling.rate_hz,
        "routes": [
            {
                "source_topic": route.source_topic,
                "destination_topic": route.recorded_topic,
                "message_type": "example_msgs/msg/StampedState",
            }
            for route in sampling.routes
        ],
    }


class _RelayManager:
    def __init__(self, sampling: ArmSamplingConfig, lifecycle: list[str]) -> None:
        self.sampling = sampling
        self.lifecycle = lifecycle
        self.process = None
        self.command = ("python", "-m", "franka_duo_tele_data.arm_rate_relay")
        self.error: str | None = None
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1
        if self.process is None:
            self.lifecycle.append("relay_start")
            self.process = _Process()
        return _ready_payload(self.sampling)

    def health_error(self):
        return self.error

    def stop(self):
        self.stop_calls += 1
        self.lifecycle.append("relay_stop")
        process = self.process
        self.process = None
        if process is not None:
            process.send_signal(signal.SIGINT)
            return process.wait(timeout=1.0)
        return None


def _successful_run(_argv, **_kwargs):
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_tmr_config_records_all_four_arm_routes_at_100_hz() -> None:
    config = load_config(REPO_ROOT / "configs" / "tmr_mcap.yaml")

    assert config.arm_sampling is not None
    assert config.arm_sampling.rate_hz == 100.0
    assert len(config.arm_sampling.routes) == 4
    assert len(config.topics) == 15
    assert len(config.recorded_topics) == 16
    assert {route.source_topic for route in config.arm_sampling.routes}.isdisjoint(config.topics)
    assert {route.recorded_topic for route in config.arm_sampling.routes}.issubset(config.topics)
    for side in ("left", "right"):
        for stream in (
            "current_pose",
            "measured_joint_states",
        ):
            source = f"/{side}/franka_robot_state_broadcaster/{stream}"
            destination = f"/franka_duo_tele_data/rate100{source}"
            assert ArmSamplingRoute(source, destination) in config.arm_sampling.routes


@pytest.mark.parametrize("case", ("duplicate_source", "missing_output", "direct_source", "bad_rate"))
def test_arm_sampling_contract_rejects_invalid_routes(tmp_path: Path, case: str) -> None:
    config = _config(tmp_path)
    sampling = config.arm_sampling
    assert sampling is not None

    if case == "duplicate_source":
        routes = list(sampling.routes)
        routes[1] = dataclasses.replace(routes[1], source_topic=routes[0].source_topic)
        config = dataclasses.replace(config, arm_sampling=dataclasses.replace(sampling, routes=tuple(routes)))
    elif case == "missing_output":
        config = dataclasses.replace(config, topics=config.topics[1:])
    elif case == "direct_source":
        config = dataclasses.replace(config, topics=(*config.topics, sampling.routes[0].source_topic))
    else:
        config = dataclasses.replace(config, arm_sampling=dataclasses.replace(sampling, rate_hz=float("nan")))

    with pytest.raises(ValueError):
        validate_config(config)


def test_relay_supervisor_requires_ready_handshake_and_stops_cleanly(tmp_path: Path) -> None:
    sampling = _sampling()
    ready_file = tmp_path / ARM_RELAY_READY_NAME
    popen_calls: list[tuple[list[str], dict]] = []
    process = _Process()

    def popen(argv, **kwargs):
        popen_calls.append((list(argv), kwargs))
        ready_file.write_text(json.dumps(_ready_payload(sampling)), encoding="utf-8")
        return process

    supervisor = ArmRateRelayProcess(
        sampling,
        ready_file=ready_file,
        startup_timeout_s=1.0,
        shutdown_timeout_s=1.0,
        popen_factory=popen,
    )

    assert supervisor.start()["status"] == "ready"
    assert popen_calls[0][1]["stdin"] == subprocess.DEVNULL
    assert popen_calls[0][1]["start_new_session"] is True
    assert popen_calls[0][0].count("--topic") == 4
    assert supervisor.stop() == 0
    assert process.signals == [signal.SIGINT]
    assert not ready_file.exists()


def test_missing_relay_topics_fail_before_rosbag_is_started(tmp_path: Path) -> None:
    config = _config(tmp_path)
    lifecycle: list[str] = []

    class FailingRelay(_RelayManager):
        def start(self):
            lifecycle.append("relay_failed")
            raise McapRecorderError("missing=/left/source/0")

    relay = FailingRelay(config.arm_sampling, lifecycle)
    bag_starts = 0

    def popen(_argv, **_kwargs):
        nonlocal bag_starts
        bag_starts += 1
        return _Process()

    recorder = RawMcapRecorder(config, "ros2", popen_factory=popen, relay_manager=relay)
    with pytest.raises(McapRecorderError, match="missing=/left/source/0"):
        recorder.start_episode()

    assert lifecycle == ["relay_failed"]
    assert bag_starts == 0
    assert recorder.active is None


def test_relay_is_started_before_rosbag_and_reused_across_episodes(tmp_path: Path) -> None:
    config = _config(tmp_path, max_episodes=2)
    lifecycle: list[str] = []
    relay = _RelayManager(config.arm_sampling, lifecycle)

    def popen(argv, **_kwargs):
        lifecycle.append("bag_start")
        Path(argv[argv.index("--output") + 1]).mkdir()
        return _Process()

    recorder = RawMcapRecorder(
        config,
        "ros2",
        popen_factory=popen,
        run_fn=_successful_run,
        relay_manager=relay,
    )
    recorder.start_episode()
    recorder.save_episode(None)
    assert relay.start_calls == 1
    assert relay.stop_calls == 0

    recorder.start_episode()
    recorder.save_episode(None)

    assert lifecycle == ["relay_start", "bag_start", "bag_start", "relay_stop"]
    assert relay.start_calls == 2
    assert relay.stop_calls == 1

    manifest = json.loads((recorder.dataset_path / DATASET_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["message_handling"]["online_resampling"] is True
    assert manifest["arm_sampling"]["selection_policy"] == "latest_unseen"
    assert manifest["arm_sampling"]["source_subscription_qos"] == "best_effort_keep_last_depth_1"
    assert manifest["arm_sampling"]["recorded_output_qos"] == "reliable_keep_last_depth_10"
    assert manifest["arm_sampling"]["bag_receipt_timestamp"] == "relay_publish_time"
    assert manifest["arm_sampling"]["source_header_stamp"] == "preserved_when_present"
    assert manifest["arm_sampling"]["serialized_byte_identity"] == "not_guaranteed"
    assert manifest["arm_sampling"]["runtime"]["shutdown_exit_code"] == 0


def test_relay_failure_marks_active_episode_incomplete(tmp_path: Path) -> None:
    config = _config(tmp_path)
    relay = _RelayManager(config.arm_sampling, [])

    def popen(argv, **_kwargs):
        Path(argv[argv.index("--output") + 1]).mkdir()
        return _Process()

    recorder = RawMcapRecorder(
        config,
        "ros2",
        popen_factory=popen,
        run_fn=_successful_run,
        relay_manager=relay,
    )
    active = recorder.start_episode()
    relay.error = "arm relay exited with code 7"

    with pytest.raises(McapRecorderError, match="retained as incomplete"):
        recorder.save_episode(None)

    episode = json.loads((active.path / EPISODE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert episode["status"] == "incomplete"
    assert episode["arm_relay_error"] == "arm relay exited with code 7"
