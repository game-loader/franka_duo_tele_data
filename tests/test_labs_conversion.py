from pathlib import Path

import numpy as np
import pytest

from franka_duo_tele_data.action_spec import (
    FrankaDuoActionSpec,
    matrix_to_rot6d,
    rot6d_to_matrix,
)
from franka_duo_tele_data.labs_kinematics import URDFFK, joint_positions
from franka_duo_tele_data.labs_mcap_to_lerobot import (
    STATE_NAMES,
    NumericSeries,
    binary_gripper,
)


def test_columns_have_explicit_non_symmetric_order():
    matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    np.testing.assert_array_equal(matrix_to_rot6d(matrix), [0, 1, 0, -1, 0, 0])
    np.testing.assert_allclose(rot6d_to_matrix([0, 1, 0, -1, 0, 0]), matrix)
    batch = np.stack([matrix, np.eye(3)])
    np.testing.assert_allclose(rot6d_to_matrix(matrix_to_rot6d(batch)), batch)
    with pytest.raises(ValueError):
        FrankaDuoActionSpec(ee_rotation="rot6d_" + "rows")


def test_target_selection_is_causal_and_bounded():
    series = NumericSeries([(100, 1), (200, 2), (300, 3)])
    assert series.match(190, 100, previous=True) == (1, -90)
    assert series.match(190, 100) == (2, 10)
    assert series.match(99, 100, previous=True) is None
    assert series.match(450, 100, previous=True) is None


def test_joint_names_not_silently_reordered():
    names = [f"fr3_joint{i}" for i in range(7, 0, -1)]
    np.testing.assert_array_equal(joint_positions(names, list(range(7, 0, -1)), "left"), range(1, 8))
    with pytest.raises(ValueError):
        joint_positions(["wrong"] * 7, [0] * 7, "left")
    assert len(STATE_NAMES) == 34


def test_gripper_binary_endpoints():
    assert binary_gripper(0) == 1
    assert binary_gripper(0.8) == 0
    assert binary_gripper(0.4) == 1
    assert binary_gripper(0, target=True) == 0
    assert binary_gripper(1, target=True) == 1


def test_gripper_accepts_full_hardware_feedback_range():
    from franka_duo_tele_data.labs_mcap_to_lerobot import validate_knuckle

    for feedback in (0, 3, 230, 248, 255):
        validate_knuckle(0.7929 * (feedback - 3) / 227)
    validate_knuckle(0.855773127753304)
    assert binary_gripper(0.855773127753304) == 0
    for invalid in (-0.1, 0.9, np.nan, np.inf):
        with pytest.raises(ValueError):
            validate_knuckle(invalid)


def test_checkpoint_reuse_rejects_changed_conversion_settings():
    from franka_duo_tele_data.labs_mcap_to_lerobot import compatible_producer_hash

    previous = {"fps": 30, "code_sha256": {"converter": "old"}}
    current = {"fps": 30, "code_sha256": {"converter": "new"}}
    assert len(compatible_producer_hash(previous, current)) == 64
    with pytest.raises(ValueError, match="fps"):
        compatible_producer_hash(previous, {**current, "fps": 15})


def test_fk_zero_configuration_includes_link8_fixed_joint():
    path = Path(__file__).resolve().parents[1] / "configs/labs_fr3_31/left.urdf"
    fk = URDFFK(path, "left")
    t = fk(np.zeros(7))
    np.testing.assert_allclose(t[:3, 3], [0.088, 0, 0.333 + 0.316 + 0.384 - 0.107], atol=1e-12)
    np.testing.assert_allclose(t[:3, :3], np.diag([1, -1, -1]), atol=1e-12)


def test_head_grid_tolerates_camera_jitter_without_discarding_frames():
    from franka_duo_tele_data.labs_mcap_to_lerobot import HeadFrameGate

    gate = HeadFrameGate(30)
    times = [round(i * 1e9 / 30) + (100_000 if i % 2 else -100_000) for i in range(300)]
    assert all(gate.keep(t) for t in times)
    assert not gate.keep(times[-1])


def test_image_statistics_include_within_frame_variation():
    from franka_duo_tele_data.mcap_to_lerobot import ImageStatsAccumulator

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[8:, :, :] = 255
    stats = ImageStatsAccumulator()
    stats.update(image)
    result = stats.finish()
    np.testing.assert_allclose(result["mean"], np.full((3, 1, 1), 0.5))
    np.testing.assert_allclose(result["std"], np.full((3, 1, 1), 0.5))
    assert result["count"] == [1]
