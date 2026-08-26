from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from franka_duo_tele_data.mcap_recorder import (
    DATASET_MANIFEST_NAME,
    EPISODE_MANIFEST_NAME,
    McapRecorderConfig,
    McapRecorderError,
    RawMcapRecorder,
    allocate_dataset_version,
    build_record_command,
    configured_raw_topics,
    parse_reward,
    preflight_ros2,
    publish_event,
    stop_process_gracefully,
)


class FakeProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode: int | None = None
        self.final_returncode = returncode
        self.signals: list[int] = []
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, value: int) -> None:
        self.signals.append(value)

    def wait(self, timeout: float) -> int:
        del timeout
        self.returncode = self.final_returncode
        return self.final_returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _config(tmp_path: Path, *, rewarded: bool = True) -> McapRecorderConfig:
    return McapRecorderConfig(
        output_root=tmp_path,
        dataset_name="tmr_raw",
        topics=("/left/measured", "/head/rgb", "/head/depth"),
        rewarded=rewarded,
    )


def _successful_run(calls: list[list[str]]):
    def run(argv, **_kwargs):
        calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def test_existing_tmr_mapping_extracts_raw_robot_camera_and_depth_topics() -> None:
    data = {
        "topics": {
            "left_measured_joint_states": "/left/measured",
            "right_desired_joint_states": "/right/desired",
        },
        "cameras": {
            "head": {
                "image": "/head/rgb",
                "depth": "/head/depth",
                "camera_info": "/head/info",
                "width": 1280,
            },
            "wrist_left": {"image": "/left/rgb", "camera_info": "/left/info"},
        },
    }

    assert configured_raw_topics(data) == (
        "/left/measured",
        "/right/desired",
        "/head/rgb",
        "/head/depth",
        "/head/info",
        "/left/rgb",
        "/left/info",
    )


def test_explicit_mcap_topics_are_authoritative_and_deduplicated() -> None:
    data = {
        "topics": {"ignored": "/ignored"},
        "mcap": {"topics": ["/raw/a", "/raw/a", "/raw/b"], "extra_topics": ["/tf"]},
    }
    assert configured_raw_topics(data) == ("/raw/a", "/raw/b", "/tf")


def test_record_command_uses_mcap_zstd_fast_and_explicit_topic_argv(tmp_path: Path) -> None:
    command = build_record_command(
        "/opt/ros/bin/ros2",
        tmp_path / "episode_000000",
        ("/camera/image_raw", "/arm/joint_states"),
    )
    assert command == [
        "/opt/ros/bin/ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--storage-preset-profile",
        "zstd_fast",
        "--output",
        str(tmp_path / "episode_000000"),
        "/camera/image_raw",
        "/arm/joint_states",
    ]


def test_preflight_fails_when_ros2_or_mcap_plugin_is_missing() -> None:
    with pytest.raises(McapRecorderError, match="Cannot find"):
        preflight_ros2(which_fn=lambda _name: None)

    def no_mcap(_argv, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="sqlite3\n", stderr="")

    with pytest.raises(McapRecorderError, match="MCAP storage plugin"):
        preflight_ros2(which_fn=lambda _name: "/usr/bin/ros2", run_fn=no_mcap)


def test_preflight_checks_preset_option_and_std_msgs_type() -> None:
    calls: list[list[str]] = []

    def run(argv, **_kwargs):
        calls.append(list(argv))
        if argv[1:4] == ["bag", "list", "storage"]:
            return SimpleNamespace(returncode=0, stdout="rosbag2_storage_mcap\n", stderr="")
        if argv[1:3] == ["bag", "record"]:
            return SimpleNamespace(returncode=0, stdout="--storage-preset-profile", stderr="")
        return SimpleNamespace(returncode=0, stdout="string data", stderr="")

    assert preflight_ros2(which_fn=lambda _name: "/usr/bin/ros2", run_fn=run) == "/usr/bin/ros2"
    assert calls[-1] == ["/usr/bin/ros2", "interface", "show", "std_msgs/msg/String"]


def test_publish_event_passes_json_without_shell_interpolation() -> None:
    calls: list[list[str]] = []
    payload = {"event": "end", "reward": 1.25, "label": "quote: ' preserved"}
    publish_event("ros2", "/events", payload, run_fn=_successful_run(calls))

    assert calls[0][:6] == ["ros2", "topic", "pub", "--once", "/events", "std_msgs/msg/String"]
    outer = json.loads(calls[0][6])
    assert json.loads(outer["data"]) == payload


def test_allocate_dataset_version_never_reuses_lower_gaps(tmp_path: Path) -> None:
    (tmp_path / "demo_v1").mkdir()
    (tmp_path / "demo_v3").mkdir()
    path, version = allocate_dataset_version(tmp_path, "demo")
    assert version == 4
    assert path.name == "demo_v4"


def test_episode_stops_before_reward_and_saves_sidecar_manifest(tmp_path: Path) -> None:
    run_calls: list[list[str]] = []
    popen_calls: list[tuple[list[str], dict]] = []
    processes: list[FakeProcess] = []

    def popen(argv, **kwargs):
        popen_calls.append((list(argv), kwargs))
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir()
        process = FakeProcess()
        processes.append(process)
        return process

    ticks = iter([100, 110, 120, 130])
    recorder = RawMcapRecorder(
        _config(tmp_path),
        "/usr/bin/ros2",
        popen_factory=popen,
        run_fn=_successful_run(run_calls),
        time_ns_fn=lambda: next(ticks),
    )
    recorder.start_episode(requested_unix_ns=105)

    def reward_after_stop() -> float:
        assert processes[0].signals == [signal.SIGINT]
        assert recorder.active is None
        return 2.5

    episode_path = recorder.save_episode(
        None,
        requested_unix_ns=125,
        reward_provider=reward_after_stop,
    )

    assert popen_calls[0][1]["stdin"] == subprocess.DEVNULL
    assert popen_calls[0][1]["start_new_session"] is True
    command = popen_calls[0][0]
    assert command[command.index("--storage-preset-profile") + 1] == "zstd_fast"
    assert command[-1] == "/franka_duo_tele_data/episode_event"
    assert processes[0].signals == [signal.SIGINT]

    event_payloads = [json.loads(json.loads(call[6])["data"]) for call in run_calls]
    assert [payload["event"] for payload in event_payloads] == ["start"]

    episode_manifest = json.loads((episode_path / EPISODE_MANIFEST_NAME).read_text())
    assert episode_manifest["status"] == "complete"
    assert episode_manifest["reward"] == 2.5
    assert episode_manifest["end_requested_unix_ns"] == 125
    assert episode_manifest["recording_stopped_unix_ns"] == 120
    assert episode_manifest["reward_recorded_unix_ns"] == 130
    assert episode_manifest["metadata_events_recorded"] == ["start"]
    assert episode_manifest["reward_storage"] == "episode_manifest"
    assert episode_manifest["topics"] == ["/left/measured", "/head/rgb", "/head/depth"]

    dataset_manifest = json.loads((recorder.dataset_path / DATASET_MANIFEST_NAME).read_text())
    assert dataset_manifest["storage_preset_profile"] == "zstd_fast"
    assert dataset_manifest["message_handling"] == {
        "subscription_and_serialization": "rosbag2",
        "online_decoding": False,
        "online_typed_deserialization": False,
        "online_synchronization": False,
        "online_resampling": False,
        "online_aggregation": False,
        "recording_stop_before_reward": True,
        "reward_storage": "sidecar_manifest_only",
        "metadata_event_scope": "start_only",
    }
    assert dataset_manifest["episodes"][0]["status"] == "complete"


def test_discard_stops_immediately_and_removes_bag(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def popen(argv, **_kwargs):
        Path(argv[argv.index("--output") + 1]).mkdir()
        return FakeProcess()

    recorder = RawMcapRecorder(
        _config(tmp_path),
        "ros2",
        popen_factory=popen,
        run_fn=_successful_run(calls),
    )
    active = recorder.start_episode()
    recorder.discard_episode(reason="operator_discard")

    assert not active.path.exists()
    event_payloads = [json.loads(json.loads(call[6])["data"]) for call in calls]
    assert [payload["event"] for payload in event_payloads] == ["start"]


def test_reward_input_error_keeps_stopped_episode_incomplete(tmp_path: Path) -> None:
    process = FakeProcess()

    def popen(argv, **_kwargs):
        Path(argv[argv.index("--output") + 1]).mkdir()
        return process

    recorder = RawMcapRecorder(
        _config(tmp_path),
        "ros2",
        popen_factory=popen,
        run_fn=_successful_run([]),
    )
    recorder.start_episode()

    def interrupted_reward() -> float:
        assert process.signals == [signal.SIGINT]
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        recorder.save_episode(None, reward_provider=interrupted_reward)

    episode_path = recorder.dataset_path / "episode_000000"
    manifest = json.loads((episode_path / EPISODE_MANIFEST_NAME).read_text())
    assert recorder.active is None
    assert manifest["status"] == "incomplete"
    assert manifest["outcome"] == "reward_input_error"
    assert manifest["reward"] is None


def test_rewarded_save_returns_idle_and_allows_next_episode(tmp_path: Path) -> None:
    def popen(argv, **_kwargs):
        Path(argv[argv.index("--output") + 1]).mkdir()
        return FakeProcess()

    recorder = RawMcapRecorder(
        _config(tmp_path),
        "ros2",
        popen_factory=popen,
        run_fn=_successful_run([]),
    )

    first = recorder.start_episode()
    recorder.save_episode(None, reward_provider=lambda: 1.0)
    assert first.index == 0
    assert recorder.active is None

    second = recorder.start_episode()
    assert second.index == 1
    assert second.path.name == "episode_000001"
    recorder.discard_episode()


def test_reward_parser_rejects_non_finite_values() -> None:
    assert parse_reward(" -3.5 ") == -3.5
    for value in ("nan", "inf", "-inf", "bad"):
        with pytest.raises(ValueError, match="finite"):
            parse_reward(value)


def test_stop_escalates_only_after_sigint_timeout() -> None:
    class StubbornProcess(FakeProcess):
        def __init__(self) -> None:
            super().__init__()
            self.wait_count = 0

        def wait(self, timeout: float) -> int:
            del timeout
            self.wait_count += 1
            if self.wait_count < 3:
                raise subprocess.TimeoutExpired("ros2", 1)
            return 9

    process = StubbornProcess()
    assert stop_process_gracefully(process, timeout_s=1.0) == 9
    assert process.signals == [signal.SIGINT]
    assert process.terminated
    assert process.killed


def test_config_can_be_loaded_without_ros_or_lerobot(tmp_path: Path) -> None:
    config_path = tmp_path / "record.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_root": str(tmp_path / "bags"),
                "dataset_name": "raw_demo",
                "topics": {"arm": "/arm/state"},
                "cameras": {"head": {"image": "/head/rgb", "depth": "/head/depth"}},
                "mcap": {"rewarded": False},
            }
        )
    )
    from franka_duo_tele_data.mcap_recorder import load_config

    config = load_config(config_path)
    assert config.topics == ("/arm/state", "/head/rgb", "/head/depth")
    assert config.rewarded is False
