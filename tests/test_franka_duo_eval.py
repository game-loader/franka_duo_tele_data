from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from franka_duo_tele_data import eval_franka_duo
from franka_duo_tele_data.action_spec import (
    ACTION_DIM,
    FrankaDuoActionSpec,
    matrix_to_rot6d,
    rot6d_to_matrix,
)
from franka_duo_tele_data.eval_franka_duo import (
    build_eval_mcap_config,
    build_policy_observation,
    load_eval_config,
)
from franka_duo_tele_data.export_rl100_eval_bundle import main as export_bundle_main
from franka_duo_tele_data.franka_duo_eval_io import (
    EvalCameraConfig,
    EvalObservationCache,
    PointCloudConfig,
    SynchronizedObservationReader,
    make_point_cloud,
)
from franka_duo_tele_data.rl100_eval_policy import (
    _prepare_native_input,
    load_policy_bundle,
)


def _stamp(stamp_ns: int) -> SimpleNamespace:
    return SimpleNamespace(
        sec=stamp_ns // 1_000_000_000,
        nanosec=stamp_ns % 1_000_000_000,
    )


def _image(value: int, stamp_ns: int, width: int, height: int) -> SimpleNamespace:
    pixels = np.full((height, width, 3), value, dtype=np.uint8)
    return SimpleNamespace(
        height=height,
        width=width,
        step=width * 3,
        encoding="rgb8",
        is_bigendian=False,
        data=pixels.tobytes(),
        header=SimpleNamespace(stamp=_stamp(stamp_ns)),
    )


def _depth(value: float, stamp_ns: int, width: int, height: int) -> SimpleNamespace:
    pixels = np.full((height, width), value, dtype=np.float32)
    return SimpleNamespace(
        height=height,
        width=width,
        step=width * 4,
        encoding="32FC1",
        is_bigendian=False,
        data=pixels.tobytes(),
        header=SimpleNamespace(stamp=_stamp(stamp_ns)),
    )


def _camera_info(width: int, height: int, stamp_ns: int) -> SimpleNamespace:
    return SimpleNamespace(
        width=width,
        height=height,
        k=[100.0, 0.0, width / 2, 0.0, 100.0, height / 2, 0.0, 0.0, 1.0],
        header=SimpleNamespace(stamp=_stamp(stamp_ns)),
    )


def test_rot6d_contract_matches_rl100_rows():
    matrix = np.eye(3, dtype=np.float32)
    six = matrix_to_rot6d(matrix)
    assert six.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    np.testing.assert_allclose(rot6d_to_matrix(six), matrix, atol=1e-6)


def test_action_spec_rejects_bad_rotation_and_workspace():
    spec = FrankaDuoActionSpec(workspace_min=(-1.0, -1.0, 0.0), workspace_max=(1.0, 1.0, 1.0))
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[3:9] = [1, 0, 0, 0, 1, 0]
    action[12:18] = [1, 0, 0, 0, 1, 0]
    action[0] = 2.0
    with pytest.raises(ValueError, match="outside workspace"):
        spec.validate(action)
    action[0] = 0.0
    action[3:9] = 0.0
    with pytest.raises(ValueError, match="degenerate"):
        spec.validate(action)


def test_pointcloud_supports_xyz_and_xyzrgb_and_fps():
    width, height = 8, 6
    depth = np.ones((height, width), dtype=np.float32)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(width, dtype=np.uint8)
    info = _camera_info(width, height, 1)
    for channels in (3, 6):
        points = make_point_cloud(
            depth,
            rgb,
            info,
            PointCloudConfig(num_points=12, channels=channels, sampling="fps", min_depth=0.1),
        )
        assert points.shape == (12, channels)
        assert np.isfinite(points).all()
    with pytest.raises(ValueError, match="shapes differ"):
        make_point_cloud(depth, rgb[:-1], info, PointCloudConfig(num_points=4))


def test_ros_reader_anchors_head_stamp_and_matches_depth_and_wrists():
    width, height = 8, 6
    cameras = {
        "head": EvalCameraConfig(
            "head",
            "/head",
            width,
            height,
            depth_topic="/depth",
            camera_info_topic="/info",
        ),
        "wrist_left": EvalCameraConfig("wrist_left", "/left", width, height),
        "wrist_right": EvalCameraConfig("wrist_right", "/right", width, height),
    }
    cache = EvalObservationCache(history_size=4)
    reader = SynchronizedObservationReader(
        cache,
        cameras,
        PointCloudConfig(num_points=8),
        rgb_tolerance_ms=2.0,
        depth_tolerance_ms=2.0,
        sync_wait_timeout_ms=2.0,
    )
    stamp = 1_000_000_000
    cache.store_camera_info(_camera_info(width, height, stamp))
    cache.store_depth(_depth(1.0, stamp + 100_000, width, height))
    cache.store_image("wrist_left", _image(2, stamp - 100_000, width, height))
    cache.store_image("wrist_right", _image(3, stamp + 100_000, width, height))
    cache.store_image("head", _image(1, stamp, width, height))
    observation = reader.next(timeout_s=0.1)
    assert observation.stamp_ns == stamp
    assert observation.point_cloud.shape == (8, 3)
    assert observation.source_stamps_ns["depth"] == stamp + 100_000
    # A new frame with a stale depth must be rejected instead of reusing the old pair.
    cache.store_image("head", _image(4, stamp + 33_333_333, width, height))
    with pytest.raises(TimeoutError, match="skew"):
        reader.next(timeout_s=0.01)


def _native_manifest() -> dict:
    return {
        "manifest_version": 1,
        "backend": "rl100_native",
        "action_dim": 20,
        "action_spec": {
            "dimension": 20,
            "ee_dimension": 9,
            "ee_rotation": "rot6d_rows",
            "gripper_range": [0.0, 1.0],
        },
        "pointcloud": {"num_points": 8, "channels": 3, "sampling": "random"},
        "inputs": {
            "point_cloud_key": "point_cloud",
            "state_key": None,
            "image_keys": {
                "wrist_left": "wrist_left",
                "wrist_right": "wrist_right",
            },
        },
        "native": {"factory": "factory_mod:make", "python_root": "."},
    }


def test_native_bundle_requires_manifest_factory_and_preserves_contract(tmp_path):
    (tmp_path / "factory_mod.py").write_text(
        """
class Model:
    def predict(self, batch):
        assert batch['point_cloud'].shape == (1, 1, 8, 3)
        assert batch['wrist_left'].shape[1] == 1
        assert batch['wrist_left'].shape[2] == 3
        assert batch['wrist_right'].shape[1] == 1
        assert batch['wrist_right'].shape[2] == 3
        return [0.0] * 20
def make(bundle_dir, device):
    return Model()
""",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(json.dumps(_native_manifest()), encoding="utf-8")
    bundle = load_policy_bundle(tmp_path, device="cpu")
    assert bundle.action_spec.dimension == ACTION_DIM
    assert bundle.required_observation_keys == (
        "point_cloud",
        "wrist_left",
        "wrist_right",
    )
    observation = {
        "point_cloud": np.zeros((8, 3), dtype=np.float32),
        "wrist_left": np.zeros((6, 16, 3), dtype=np.uint8),
        "wrist_right": np.zeros((6, 16, 3), dtype=np.uint8),
    }
    np.testing.assert_equal(bundle.predict(observation), np.zeros(20, dtype=np.float32))


def test_checkpoint_can_use_external_manifest(tmp_path):
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    (tmp_path / "external_factory.py").write_text(
        """
class Model:
    def select_action(self, batch):
        return [0.0] * 20
def make(bundle_dir, device):
    assert bundle_dir.name == 'pretrained_model'
    return Model()
""",
        encoding="utf-8",
    )
    manifest = _native_manifest()
    manifest["native"] = {"factory": "external_factory:make", "python_root": ".."}
    manifest_path = tmp_path / "franka_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle = load_policy_bundle(checkpoint, device="cpu", manifest_path=manifest_path)
    assert bundle.predict(
        {
            "point_cloud": np.zeros((8, 3), dtype=np.float32),
            "wrist_left": np.zeros((2, 2, 3), np.uint8),
            "wrist_right": np.zeros((2, 2, 3), np.uint8),
        }
    ).shape == (20,)


def test_native_input_uses_manifest_image_keys_for_custom_names():
    prepared = _prepare_native_input(
        {"wrist_left": np.zeros((6, 8, 3), dtype=np.uint8)},
        "cpu",
        image_keys=("wrist_left",),
    )
    assert tuple(prepared["wrist_left"].shape) == (1, 3, 6, 8)
    assert prepared["wrist_left"].dtype == torch.float32


def test_native_bundle_rejects_missing_required_observation(tmp_path):
    (tmp_path / "factory_mod.py").write_text(
        """
class Model:
    def predict(self, batch):
        return [0.0] * 20
def make(bundle_dir, device):
    return Model()
""",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(json.dumps(_native_manifest()), encoding="utf-8")
    bundle = load_policy_bundle(tmp_path, device="cpu")
    with pytest.raises(ValueError, match="missing required"):
        bundle.predict({"point_cloud": np.zeros((8, 3), dtype=np.float32)})


def test_existing_checkpoint_without_bundle_manifest_is_rejected(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"not a self describing export")
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_policy_bundle(tmp_path, device="cpu")


def test_export_bundle_copies_native_weights_and_writes_manifest(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"model")
    (checkpoint / "encoder.pt").write_bytes(b"encoder")
    output = tmp_path / "bundle"
    assert (
        export_bundle_main(
            [
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--factory",
                "factory_mod:make",
                "--workspace-min=-1,-1,0",
                "--workspace-max=1,1,1",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == "rl100_native"
    assert manifest["action_spec"]["dimension"] == ACTION_DIM
    assert (output / "checkpoint" / "encoder.pt").is_file()


def test_policy_observation_can_pack_both_wrist_images():
    observation = SimpleNamespace(
        head_rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        wrist_left_rgb=np.ones((2, 3, 3), dtype=np.uint8),
        wrist_right_rgb=np.full((2, 3, 3), 2, dtype=np.uint8),
        point_cloud=np.zeros((4, 3), dtype=np.float32),
        state=None,
    )
    values = build_policy_observation(
        observation,
        {
            "inputs": {
                "point_cloud_key": "point_cloud",
                "image_keys": {"image": ["wrist_left", "wrist_right"]},
            }
        },
    )
    assert values["image"].shape == (2, 6, 3)
    assert values["image"][0, 0, 0] == 1 and values["image"][0, -1, 0] == 2


def _write_eval_mcap_config(tmp_path: Path, topics: list[str]) -> Path:
    path = tmp_path / "raw_mcap.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "output_root": str(tmp_path / "bags"),
                "dataset_name": "raw_inputs",
                "mcap": {"rewarded": False, "topics": topics},
            }
        ),
        encoding="utf-8",
    )
    return path


def _default_eval_input_topics() -> list[str]:
    return [
        "/isaac/head_camera/image_raw",
        "/isaac/head_camera/depth",
        "/isaac/head_camera/camera_info",
        "/isaac/left_wrist_camera/image_raw",
        "/isaac/right_wrist_camera/image_raw",
    ]


def test_eval_mcap_config_records_all_raw_topics_and_action_trace(tmp_path):
    raw_config = _write_eval_mcap_config(tmp_path, [*_default_eval_input_topics(), "/control/raw"])
    config = load_eval_config(None)
    config.mcap_config = raw_config
    args = SimpleNamespace(
        mcap_config=None,
        mcap_output_root=tmp_path / "override",
        mcap_dataset_name="checkpoint_eval",
        reward=None,
        prompt_reward=True,
    )

    capture = build_eval_mcap_config(config, args, require_state=False)

    assert capture.output_root == tmp_path / "override"
    assert capture.dataset_name == "checkpoint_eval"
    assert capture.max_episodes == 1
    assert capture.rewarded is True
    assert capture.topics == (
        *_default_eval_input_topics(),
        "/control/raw",
        "/franka_duo/eval/action_trace",
    )


def test_eval_mcap_config_preserves_arm_sampling_contract(tmp_path):
    routes = [
        {
            "source_topic": f"/{side}/franka_robot_state_broadcaster/{stream}",
            "recorded_topic": (
                f"/franka_duo_tele_data/rate100/{side}/franka_robot_state_broadcaster/{stream}"
            ),
        }
        for side in ("left", "right")
        for stream in (
            "current_pose",
            "measured_joint_states",
        )
    ]
    for side in ("left", "right"):
        routes.append(
            {
                "source_topic": f"/{side}/gripper/joint_states",
                "recorded_topic": f"/franka_duo_tele_data/rate100/{side}/gripper/joint_states",
            }
        )
    raw_config = tmp_path / "raw_mcap.yaml"
    raw_config.write_text(
        yaml.safe_dump(
            {
                "output_root": str(tmp_path / "bags"),
                "dataset_name": "raw_inputs",
                "arm_sampling": {"rate_hz": 100, "routes": routes},
                "mcap": {
                    "rewarded": False,
                    "topics": [
                        *_default_eval_input_topics(),
                        *(route["recorded_topic"] for route in routes),
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_eval_config(None)
    config.mcap_config = raw_config
    args = SimpleNamespace(
        mcap_config=None,
        mcap_output_root=None,
        mcap_dataset_name=None,
        reward=None,
        prompt_reward=False,
    )

    capture = build_eval_mcap_config(config, args, require_state=False)

    assert capture.arm_sampling is not None
    assert capture.arm_sampling.rate_hz == 100.0
    assert len(capture.arm_sampling.routes) == 6
    assert capture.topics[-1] == "/franka_duo/eval/action_trace"


def test_eval_mcap_config_rejects_an_unrecorded_input_topic(tmp_path):
    topics = _default_eval_input_topics()
    topics.remove("/isaac/head_camera/depth")
    raw_config = _write_eval_mcap_config(tmp_path, topics)
    config = load_eval_config(None)
    config.mcap_config = raw_config
    args = SimpleNamespace(
        mcap_config=None,
        mcap_output_root=None,
        mcap_dataset_name=None,
        reward=None,
        prompt_reward=False,
    )

    with pytest.raises(ValueError, match="does not record eval input.*head_camera/depth"):
        build_eval_mcap_config(config, args, require_state=False)


class _FakeEvalMcapRecorder:
    instances: list[_FakeEvalMcapRecorder] = []
    lifecycle: list[str] = []

    def __init__(self, config, ros2_executable):
        self.config = config
        self.ros2_executable = ros2_executable
        self.dataset_path = config.output_root / f"{config.dataset_name}_v1"
        self.active = None
        self.saved_rewards: list[float | None] = []
        self.saved_requested_unix_ns: list[int | None] = []
        self.preserved_reasons: list[str] = []
        self.__class__.instances.append(self)

    def start_episode(self):
        self.__class__.lifecycle.append("mcap_start")
        self.active = SimpleNamespace(index=0)

    def save_episode(self, reward, *, requested_unix_ns=None, reward_provider=None):
        self.__class__.lifecycle.append("mcap_stop")
        self.active = None
        if reward_provider is not None:
            reward = reward_provider()
        self.__class__.lifecycle.append("mcap_save")
        self.saved_rewards.append(reward)
        self.saved_requested_unix_ns.append(requested_unix_ns)
        return self.dataset_path / "episode_000000"

    def preserve_interrupted_episode(self, *, reason):
        self.__class__.lifecycle.append(f"mcap_incomplete:{reason}")
        self.preserved_reasons.append(reason)
        self.active = None
        return self.dataset_path / "episode_000000"


class _FakePublisher:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.messages: list[object] = []

    def publish(self, message) -> None:
        self.messages.append(message)

    @staticmethod
    def get_subscription_count() -> int:
        return 1


class _FakeNode:
    def __init__(self) -> None:
        self.publishers: dict[str, _FakePublisher] = {}
        self.destroyed = False

    def create_publisher(self, _message_type, topic, _depth):
        publisher = _FakePublisher(topic)
        self.publishers[topic] = publisher
        return publisher

    def destroy_node(self) -> None:
        self.destroyed = True


class _FakeRclpy:
    def __init__(self) -> None:
        self.shutdown_called = False

    @staticmethod
    def spin(_node) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeEvalBundle:
    def __init__(self) -> None:
        self.action_spec = SimpleNamespace(workspace_min=None, validate=lambda action: action)
        self.pointcloud_config = SimpleNamespace()
        self.requires_state = False
        self.manifest = {"inputs": {"point_cloud_key": "point_cloud", "state_key": None, "image_keys": {}}}
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    @staticmethod
    def predict(_observation):
        return np.zeros(20, dtype=np.float32)


def _eval_args(eval_config: Path, *, prompt_reward: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        bundle=Path("/trusted/bundle"),
        checkpoint=None,
        manifest=None,
        config=eval_config,
        device="cpu",
        publish=False,
        enable_robot=False,
        once=True,
        max_steps=None,
        output_jsonl=None,
        mcap_config=None,
        mcap_output_root=None,
        mcap_dataset_name=None,
        reward=None,
        prompt_reward=prompt_reward,
    )


def _install_fake_eval_runtime(monkeypatch, tmp_path: Path, *, reader_error=None):
    raw_config = _write_eval_mcap_config(tmp_path, [*_default_eval_input_topics(), "/control/raw"])
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text(
        yaml.safe_dump(
            {
                "mcap": {"config": raw_config.name, "dataset_name": "eval_capture"},
                "trace_topic": "/eval/action_trace",
            }
        ),
        encoding="utf-8",
    )
    fake_messages = ModuleType("std_msgs.msg")

    class String:
        def __init__(self) -> None:
            self.data = ""

    fake_messages.String = String
    fake_std_msgs = ModuleType("std_msgs")
    fake_std_msgs.msg = fake_messages
    monkeypatch.setitem(sys.modules, "std_msgs", fake_std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", fake_messages)

    bundle = _FakeEvalBundle()
    node = _FakeNode()
    rclpy = _FakeRclpy()
    _FakeEvalMcapRecorder.instances = []
    _FakeEvalMcapRecorder.lifecycle = []
    monkeypatch.setattr(eval_franka_duo, "load_policy_bundle", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(eval_franka_duo, "preflight_ros2", lambda: "/usr/bin/ros2")
    monkeypatch.setattr(eval_franka_duo, "RawMcapRecorder", _FakeEvalMcapRecorder)
    monkeypatch.setattr(eval_franka_duo, "_ros_node", lambda *_args, **_kwargs: (rclpy, node))

    observation = SimpleNamespace(
        stamp_ns=123,
        source_stamps_ns={"head": 123},
        head_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        wrist_left_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        wrist_right_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        point_cloud=np.zeros((4, 3), dtype=np.float32),
        state=None,
    )

    class Reader:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def next(self, **_kwargs):
            _FakeEvalMcapRecorder.lifecycle.append("reader_next")
            if reader_error is not None:
                raise reader_error
            return observation

    monkeypatch.setattr(eval_franka_duo, "SynchronizedObservationReader", Reader)
    return eval_config, bundle, node, rclpy


def test_eval_automatically_records_one_complete_episode_with_prompted_reward(monkeypatch, tmp_path):
    eval_config, bundle, node, rclpy = _install_fake_eval_runtime(monkeypatch, tmp_path)

    def prompt_after_stop() -> float:
        assert _FakeEvalMcapRecorder.instances[0].active is None
        _FakeEvalMcapRecorder.lifecycle.append("reward_prompt")
        return 4.25

    monkeypatch.setattr(eval_franka_duo, "prompt_reward", prompt_after_stop)

    assert eval_franka_duo.run(_eval_args(eval_config, prompt_reward=True)) == 0

    recorder = _FakeEvalMcapRecorder.instances[0]
    assert recorder.config.topics[-2:] == ("/control/raw", "/eval/action_trace")
    assert recorder.saved_rewards == [4.25]
    assert len(recorder.saved_requested_unix_ns) == 1
    assert recorder.saved_requested_unix_ns[0] is not None
    assert recorder.preserved_reasons == []
    assert _FakeEvalMcapRecorder.lifecycle == [
        "mcap_start",
        "reader_next",
        "mcap_stop",
        "reward_prompt",
        "mcap_save",
    ]
    assert len(node.publishers["/eval/action_trace"].messages) == 1
    trace = json.loads(node.publishers["/eval/action_trace"].messages[0].data)
    assert trace["action"] == [0.0] * 20
    assert bundle.reset_calls == 2
    assert node.destroyed and rclpy.shutdown_called


def test_eval_exception_retains_incomplete_mcap(monkeypatch, tmp_path):
    eval_config, _bundle, _node, _rclpy = _install_fake_eval_runtime(
        monkeypatch, tmp_path, reader_error=RuntimeError("inference input failed")
    )

    with pytest.raises(RuntimeError, match="inference input failed"):
        eval_franka_duo.run(_eval_args(eval_config))

    recorder = _FakeEvalMcapRecorder.instances[0]
    assert recorder.saved_rewards == []
    assert recorder.preserved_reasons == ["eval_error"]


def test_eval_keyboard_interrupt_retains_incomplete_mcap_and_returns_130(monkeypatch, tmp_path):
    eval_config, _bundle, _node, _rclpy = _install_fake_eval_runtime(
        monkeypatch, tmp_path, reader_error=KeyboardInterrupt()
    )

    assert eval_franka_duo.run(_eval_args(eval_config)) == 130
    assert _FakeEvalMcapRecorder.instances[0].preserved_reasons == ["keyboard_interrupt"]


def test_publish_without_second_gate_fails_before_mcap_start(monkeypatch, tmp_path):
    eval_config, _bundle, _node, _rclpy = _install_fake_eval_runtime(monkeypatch, tmp_path)
    args = _eval_args(eval_config)
    args.publish = True
    args.enable_robot = False

    with pytest.raises(ValueError, match="explicit second safety gate"):
        eval_franka_duo.run(args)
    assert _FakeEvalMcapRecorder.instances == []
