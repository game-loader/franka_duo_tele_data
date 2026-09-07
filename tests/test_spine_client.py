"""ROS-free tests for the spine height client contract."""

from __future__ import annotations

import pytest

from franka_duo_tele_data.spine_client import (
    GRASP_HEIGHT_M,
    SPINE_MAX_M,
    TRAVEL_HEIGHT_M,
    height_reached,
    plan_motion,
    report_is_stable,
)


def test_plan_motion_accepts_the_two_mission_heights():
    for target in (TRAVEL_HEIGHT_M, GRASP_HEIGHT_M):
        motion = plan_motion(target)
        assert motion.position_m == target
        assert motion.velocity_mps > 0
        assert motion.acceleration_mps2 > 0
        assert motion.deceleration_mps2 > 0


def test_plan_motion_rejects_out_of_range_and_nonpositive_rates():
    # URDF prismatic limit of franka_spine_vertical_joint is [0.0, 0.85].
    for target in (-0.01, SPINE_MAX_M + 0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            plan_motion(target)
    for kwargs in (
        {"velocity_mps": 0.0},
        {"velocity_mps": -0.05},
        {"acceleration_mps2": 0.0},
        {"deceleration_mps2": -1.0},
    ):
        with pytest.raises(ValueError):
            plan_motion(GRASP_HEIGHT_M, **kwargs)


def test_height_reached_uses_the_three_millimetre_tolerance():
    assert height_reached(0.468, 0.468)
    assert height_reached(0.4703, 0.468)
    assert not height_reached(0.4715, 0.468)
    assert not height_reached(float("nan"), 0.468)
    with pytest.raises(ValueError):
        height_reached(0.468, 0.468, tolerance_m=0.0)


def test_report_is_stable_requires_success_and_the_requested_target():
    good = {
        "status": "success",
        "target_position_m": GRASP_HEIGHT_M,
        "measured_position_m": GRASP_HEIGHT_M + 0.001,
    }
    assert report_is_stable(good, GRASP_HEIGHT_M)
    # A report proving a different height must not satisfy this request.
    assert not report_is_stable(good, TRAVEL_HEIGHT_M)
    assert not report_is_stable({**good, "status": "failed"}, GRASP_HEIGHT_M)
    assert not report_is_stable({**good, "measured_position_m": GRASP_HEIGHT_M + 0.02}, GRASP_HEIGHT_M)
    assert not report_is_stable({"status": "success"}, GRASP_HEIGHT_M)
    assert not report_is_stable(None, GRASP_HEIGHT_M)
