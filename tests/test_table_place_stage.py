"""ROS-free tests for placement: the mirror image of the grasp."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("cv2")

from franka_duo_tele_data.action_spec import FrankaDuoActionSpec, rot6d_to_matrix  # noqa: E402
from franka_duo_tele_data.grasp_cup_bowl import (  # noqa: E402
    ARM_SLICE,
    DATASET_GRASP_ROT6D,
    GRIPPER_INDEX,
    build_place_rows,
    release_clearance_m,
)
from franka_duo_tele_data.table_grasp_stage import ApproachLimits  # noqa: E402
from franka_duo_tele_data.table_place_stage import plan_placement  # noqa: E402

TABLE_Z = -0.220
CUP_HEIGHT = 0.080
CUP_DEPTH = 0.055


def held_state(arm: str = "right", ee_z: float = -0.05) -> np.ndarray:
    """A state with the given arm holding something (gripper closed)."""
    state = np.zeros(20, np.float32)
    state[0:3] = (0.40, 0.20, ee_z)
    state[3:9] = DATASET_GRASP_ROT6D["left"]
    state[9:12] = (0.40, -0.15, ee_z)
    state[12:18] = DATASET_GRASP_ROT6D["right"]
    state[18:] = 1.0
    state[GRIPPER_INDEX[arm]] = 0.0  # closed: holding
    return state


def test_release_clearance_is_the_object_below_the_ee_origin():
    # Grasped 5.5 cm below an 8 cm rim, so 2.5 cm of cup hangs below the EE.
    assert release_clearance_m(CUP_HEIGHT, CUP_DEPTH) == pytest.approx(0.025)
    assert release_clearance_m(0.045, 0.025) == pytest.approx(0.020)
    with pytest.raises(ValueError):
        release_clearance_m(0.05, 0.05)
    with pytest.raises(ValueError):
        release_clearance_m(0.02, 0.05)


def test_place_rows_descend_straight_down_and_keep_xy():
    """X/Y must come from the current pose: the base has moved since the grasp."""
    state = held_state("right", ee_z=0.05)
    rows = build_place_rows(state, "right", table_z=TABLE_Z, clearance_m=0.025, lift_m=0.10)
    sl = ARM_SLICE["right"]
    start_xy = state[sl][:2]
    for row in rows:
        assert np.allclose(row[sl][:2], start_xy, atol=1e-6)
        # Orientation is held throughout: the grasp already aligned the object.
        assert np.allclose(row[sl][3:9], state[sl][3:9], atol=1e-3)
    # The lowest row puts the object base on the table.
    lowest = min(float(row[sl][2]) for row in rows)
    assert lowest == pytest.approx(TABLE_Z + 0.025, abs=1e-6)
    # The other arm is untouched.
    assert np.allclose(rows[:, ARM_SLICE["left"]], state[ARM_SLICE["left"]], atol=1e-6)


def test_place_rows_respect_step_limits_and_the_contract():
    state = held_state("left", ee_z=0.08)
    rows = build_place_rows(state, "left", table_z=TABLE_Z, clearance_m=0.02, lift_m=0.10)
    spec = FrankaDuoActionSpec()
    previous = state
    for row in rows:
        spec.validate(row)
        for offset in (0, 9):
            assert np.linalg.norm(row[offset : offset + 3] - previous[offset : offset + 3]) <= 0.0101
            delta = (
                rot6d_to_matrix(previous[offset + 3 : offset + 9]).T
                @ rot6d_to_matrix(row[offset + 3 : offset + 9])
            )
            assert math.acos(min(1.0, (np.trace(delta) - 1) / 2)) <= 0.081
        previous = row


def test_final_mode_opens_and_leaves_the_object():
    state = held_state("right", ee_z=0.05)
    rows = build_place_rows(
        state, "right", table_z=TABLE_Z, clearance_m=0.025, regrasp=False, open_rows=14
    )
    grip = rows[:, GRIPPER_INDEX["right"]]
    # Closed while descending, then opens and never closes again.
    assert grip[0] == 0.0
    assert grip[-1] == 1.0
    opened = np.where(grip == 1.0)[0]
    assert len(opened) >= 14
    assert np.all(grip[opened[0] :] == 1.0), "final placement must not re-close"


def test_test_mode_opens_then_closes_and_carries_the_object_up():
    state = held_state("right", ee_z=0.05)
    rows = build_place_rows(
        state, "right", table_z=TABLE_Z, clearance_m=0.025, regrasp=True, open_rows=14, close_rows=14
    )
    grip = rows[:, GRIPPER_INDEX["right"]]
    sl = ARM_SLICE["right"]
    assert grip[0] == 0.0
    opened = np.where(grip == 1.0)[0]
    assert len(opened) >= 14
    # After opening it closes again and stays closed while lifting.
    assert grip[-1] == 0.0, "test placement must re-grasp before lifting"
    assert np.all(grip[opened[-1] + 1 :] == 0.0)
    # The gripper only ever opens at the table plane, never in mid-air.
    place_z = TABLE_Z + 0.025
    for index in opened:
        assert rows[index][sl][2] == pytest.approx(place_z, abs=1e-6)
    # It ends lifted back up, carrying the object.
    assert float(rows[-1][sl][2]) > place_z + 0.05


def test_place_rows_reject_a_target_above_the_current_pose():
    """Descending to a plane above the hand would be a jump upward, not a place."""
    state = held_state("right", ee_z=-0.30)
    with pytest.raises(ValueError, match="above the current EE z"):
        build_place_rows(state, "right", table_z=TABLE_Z, clearance_m=0.025)


def test_plan_placement_refuses_an_empty_gripper():
    state = held_state("right")
    state[GRIPPER_INDEX["right"]] = 1.0  # open: holding nothing
    with pytest.raises(ValueError, match="nothing to place"):
        plan_placement(
            state,
            "right",
            mode="final",
            table_z=TABLE_Z,
            object_height_m=CUP_HEIGHT,
            grasp_depth_m=CUP_DEPTH,
            lift_m=0.10,
            limits=ApproachLimits(),
        )


def test_plan_placement_reports_geometry_and_honours_margin():
    state = held_state("right", ee_z=0.05)
    rows, geometry = plan_placement(
        state,
        "right",
        mode="test",
        table_z=TABLE_Z,
        object_height_m=CUP_HEIGHT,
        grasp_depth_m=CUP_DEPTH,
        lift_m=0.10,
        limits=ApproachLimits(),
        margin_m=0.005,
    )
    assert geometry["regrasp"] is True
    assert geometry["release_clearance_m"] == pytest.approx(0.025)
    # Table + object-below-EE + margin.
    assert geometry["placement_ee_z"] == pytest.approx(TABLE_Z + 0.025 + 0.005)
    assert geometry["descent_m"] == pytest.approx(0.05 - (TABLE_Z + 0.030))
    lowest = min(float(row[ARM_SLICE["right"]][2]) for row in rows)
    assert lowest == pytest.approx(TABLE_Z + 0.030, abs=1e-6)

    _rows, final_geometry = plan_placement(
        state,
        "right",
        mode="final",
        table_z=TABLE_Z,
        object_height_m=CUP_HEIGHT,
        grasp_depth_m=CUP_DEPTH,
        lift_m=0.10,
        limits=ApproachLimits(),
    )
    assert final_geometry["regrasp"] is False


def test_plan_placement_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        plan_placement(
            held_state("right"),
            "right",
            mode="somewhere",
            table_z=TABLE_Z,
            object_height_m=CUP_HEIGHT,
            grasp_depth_m=CUP_DEPTH,
            lift_m=0.10,
            limits=ApproachLimits(),
        )
