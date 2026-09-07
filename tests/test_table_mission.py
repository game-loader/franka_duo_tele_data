"""ROS-free tests for the table mission coordinator: phases, reports, locking."""

from __future__ import annotations

import json

import pytest

from franka_duo_tele_data.spine_client import GRASP_HEIGHT_M, TRAVEL_HEIGHT_M
from franka_duo_tele_data.table_mission import (
    DEPARTED_PHASES,
    LETTER_SIDE_RIGHT_M,
    MissionConfig,
    MissionError,
    MissionRunLock,
    Phase,
    atomic_write_json,
    base_report_is_stable,
    build_base_argv,
    build_place_argv,
    build_placement_route_argv,
    build_post_grasp_argv,
    build_return_from_letter_argv,
    build_spine_argv,
    build_stage_argv,
    build_stow_argv,
    extract_last_json_object,
    load_checkpoint,
    main,
    place_report_is_stable,
    placement_route_is_waiting,
    return_report_is_stable,
    route_report_is_stable,
    stage_report_is_stable,
    strategy,
)


def config(**overrides) -> MissionConfig:
    values = {
        "base_host": "tmr-user@172.16.0.50",
        "base_root": "/home/tmr-user/tmr_cycle",
        "arm_root": "/home/aup/franka_duo_tele_data",
        "arm_env": "/home/aup/tmr_env.sh",
        "dataset": "datasets/franka_duo_lerobot_rgb20d_v1",
        "speed": 0.1,
        "init_timeout_s": 180.0,
        "outbound_timeout_s": 420.0,
        "stage_timeout_s": 300.0,
        "transition_settle_s": 0.5,
    }
    values.update(overrides)
    return MissionConfig(**values)


def test_phase_order_puts_stow_and_spine_before_departure():
    order = [phase.value for phase in Phase]
    assert order.index("INITIALIZING_ARMS") < order.index("INITIALIZING_SPINE")
    assert order.index("INITIALIZING_SPINE") < order.index("OUTBOUND_BASE_RUNNING")
    assert order.index("OUTBOUND_BASE_RUNNING") < order.index("AT_PICKUP_TABLE")
    assert order.index("AT_PICKUP_TABLE") < order.index("CUP_STAGE_RUNNING")
    assert order.index("CUP_STAGE_RUNNING") < order.index("BOWL_STAGE_RUNNING")


def test_phase_order_covers_the_whole_round_trip():
    order = [phase.value for phase in Phase]
    sequence = [
        "INITIALIZING_ARMS",
        "INITIALIZING_SPINE",
        "OUTBOUND_BASE_RUNNING",
        "AT_PICKUP_TABLE",
        "CUP_STAGE_RUNNING",
        "BOWL_STAGE_RUNNING",
        "OBJECTS_HELD",
        "RAISING_SPINE_FOR_LETTER",
        "POST_GRASP_ROUTE_RUNNING",
        "LOWERING_SPINE_AT_LETTER",
        "AT_LETTER_TABLE",
        "TEST_PLACE_RUNNING",
        "RAISING_SPINE_FOR_RETURN",
        "RETURN_ROUTE_RUNNING",
        "PLACEMENT_ROUTE_RUNNING",
        "LOWERING_SPINE_AT_PLACEMENT",
        "AT_PLACEMENT_TABLE",
        "FINAL_PLACE_RUNNING",
        "COMPLETE",
    ]
    positions = [order.index(name) for name in sequence]
    assert positions == sorted(positions)


def test_every_drive_is_bracketed_by_a_spine_change():
    """The arms ride the spine, so no drive may start at the grasp height."""
    order = [phase.value for phase in Phase]
    for raise_phase, drive in (
        ("INITIALIZING_SPINE", "OUTBOUND_BASE_RUNNING"),
        ("RAISING_SPINE_FOR_LETTER", "POST_GRASP_ROUTE_RUNNING"),
        ("RAISING_SPINE_FOR_RETURN", "RETURN_ROUTE_RUNNING"),
    ):
        assert order.index(raise_phase) < order.index(drive)
    # And every table stop lowers again before an arm touches anything.
    for drive, lower, work in (
        ("POST_GRASP_ROUTE_RUNNING", "LOWERING_SPINE_AT_LETTER", "TEST_PLACE_RUNNING"),
        ("PLACEMENT_ROUTE_RUNNING", "LOWERING_SPINE_AT_PLACEMENT", "FINAL_PLACE_RUNNING"),
    ):
        assert order.index(drive) < order.index(lower) < order.index(work)


def test_departed_phases_include_every_phase_after_the_drive():
    """Resuming past any of these must not replay a route."""
    departed = {phase.value for phase in DEPARTED_PHASES}
    order = [phase.value for phase in Phase]
    first = order.index("AT_PICKUP_TABLE")
    terminal = {"FAILED", "INTERRUPTED"}
    for name in order[first:]:
        if name not in terminal:
            assert name in departed, f"{name} happens after the drive but is not departed"


def test_post_grasp_and_return_legs_target_the_right_scripts():
    post_grasp = build_post_grasp_argv(config(), "abc123")[-1]
    assert "13_post_grasp_route.py" in post_grasp
    assert "--execute" in post_grasp

    back = build_return_from_letter_argv(config(), "abc123")[-1]
    assert "15_return_from_letter.py" in back
    # Script 15 needs the outbound right-shift distance to undo it.
    assert f"--left-m {LETTER_SIDE_RIGHT_M:.2f}" in back

    placement = build_placement_route_argv(config(), "abc123")[-1]
    assert "20_after_return_placement.py outbound" in placement
    # Step 20 reads the return leg's own state file.
    assert "tmr_table_mission_return_abc123.json" in placement


def test_place_argv_carries_mode_arm_and_table_height():
    command = build_place_argv(config(), "test", "cup", "right", -0.2205, "abc123")[-1]
    assert "franka_duo_tele_data.table_place_stage" in command
    assert "--mode test" in command
    assert "--arm right" in command
    assert "--target cup" in command
    assert "--table-z -0.2205" in command
    assert "--publish" in command and "--enable-robot" in command


def test_route_and_placement_validators_match_each_script_contract():
    # Script 13 writes a flat complete/latched state, with no doorway report.
    assert route_report_is_stable({"status": "complete", "zero_command_latched": True})
    assert not route_report_is_stable({"status": "complete", "zero_command_latched": False})
    assert not route_report_is_stable({"status": "failed", "zero_command_latched": True})

    # Script 15 nests its doorway result, so the flat outbound shape must not pass.
    fifteen = {
        "status": "complete",
        "phase": "COMPLETE",
        "zero_command_latched": True,
        "door_report": {
            "status": "success",
            "final_state": "FINAL_STOP",
            "zero_command_latched": True,
            "final_stationary": {"confirmed": True},
        },
    }
    assert return_report_is_stable(fifteen)
    assert not return_report_is_stable({**fifteen, "door_report": {}})
    assert not base_report_is_stable(fifteen), "the outbound shape must not accept script 15"

    waiting = {
        "phase": "WAITING_FOR_PLACEMENT",
        "zero_command_latched": True,
        "placement_completed": False,
    }
    assert placement_route_is_waiting(waiting)
    assert not placement_route_is_waiting({**waiting, "placement_completed": True})
    assert not placement_route_is_waiting({**waiting, "phase": "COMPLETE"})


def test_place_report_must_match_the_requested_mode():
    """A test placement keeps the object; a final one gives it up."""
    carried = {"status": "success", "place": {"rows": 40}, "regrasp": True}
    released = {"status": "success", "place": {"rows": 40}, "regrasp": False}
    assert place_report_is_stable(carried, "test")
    assert place_report_is_stable(released, "final")
    # Reporting the wrong one means the object was left behind, or not left.
    assert not place_report_is_stable(released, "test")
    assert not place_report_is_stable(carried, "final")
    assert not place_report_is_stable({**carried, "status": "failed"}, "test")
    assert not place_report_is_stable(None, "final")


def test_departed_phases_cover_everything_after_the_drive():
    # Resuming past any of these must not replay the route from the start.
    assert Phase.AT_PICKUP_TABLE in DEPARTED_PHASES
    assert Phase.COMPLETE in DEPARTED_PHASES
    assert Phase.READY_TO_DEPART not in DEPARTED_PHASES
    assert Phase.CREATED not in DEPARTED_PHASES


def test_base_argv_runs_the_route_under_a_lock_on_the_base_host():
    argv = build_base_argv(config(), "abc123")
    assert argv[0] == "ssh"
    assert "tmr-user@172.16.0.50" in argv
    remote = argv[-1]
    assert "07_start_to_pickup.py" in remote
    assert "--execute" in remote
    assert "flock -n 9" in remote
    # The base host runs Humble on its own domain; graphs must not be mixed.
    assert "/opt/ros/humble/setup.bash" in remote
    assert "ROS_DOMAIN_ID=${TMR_CYCLE_ROS_DOMAIN_ID:-97}" in remote
    # nounset only after ROS overlays, which legitimately probe unset variables.
    assert remote.index("set -eo pipefail") < remote.index("set -u")
    assert remote.index("/opt/ros/humble/setup.bash") < remote.index("set -u")


def test_stage_argv_requests_the_grasp_height_and_both_robot_gates():
    command = build_stage_argv(config(), "cup", "right", "abc123")[-1]
    assert "franka_duo_tele_data.table_grasp_stage" in command
    assert "--target cup" in command
    assert "--arm right" in command
    assert f"--spine-target-m {GRASP_HEIGHT_M:.3f}" in command
    # Robot output needs both gates, per the repository's safety contract.
    assert "--publish" in command and "--enable-robot" in command
    # PYTHONPATH is appended; replacing it hides the host ROS packages.
    assert 'export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"' in command


def test_stow_argv_reaches_the_travel_posture_without_spine_or_grasp():
    command = build_stow_argv(config())[-1]
    assert "--pose-only travel_stow" in command
    assert "--spine-target-m" not in command
    assert "--target" not in command


def test_spine_argv_uses_the_requested_height_and_execute_gate():
    command = build_spine_argv(config(), TRAVEL_HEIGHT_M)[-1]
    assert "franka_duo_tele_data.spine_client" in command
    assert f"--target-m {TRAVEL_HEIGHT_M:.3f}" in command
    assert "--execute" in command


def test_base_report_is_stable_requires_a_confirmed_latched_stop():
    good = {
        "status": "success",
        "final_state": "FINAL_STOP",
        "final_stationary": {"confirmed": True},
        "zero_command_latched": True,
    }
    assert base_report_is_stable(good)
    assert not base_report_is_stable({**good, "status": "failed"})
    assert not base_report_is_stable({**good, "final_state": "ABORT"})
    assert not base_report_is_stable({**good, "zero_command_latched": False})
    assert not base_report_is_stable({**good, "final_stationary": {"confirmed": False}})
    assert not base_report_is_stable({**good, "final_stationary": None})
    assert not base_report_is_stable(None)


def test_stage_report_is_stable_requires_view_clearing_and_a_grasp():
    good = {
        "status": "success",
        "stages": [{"stage": "camera_clear"}, {"stage": "grasp_ready"}],
        "grasp": {"rows": 58},
    }
    assert stage_report_is_stable(good)
    # An untaught posture reports itself as skipped; the grasp still has to happen.
    skipped = {
        "status": "success",
        "stages": [
            {"stage": "camera_clear", "skipped": "not taught"},
            {"stage": "grasp_ready", "skipped": "not taught"},
        ],
        "grasp": {"rows": 58},
    }
    assert stage_report_is_stable(skipped)
    # A stage that never reached either posture cannot vouch for itself.
    assert not stage_report_is_stable({**good, "stages": [{"stage": "grasp_ready"}]})
    assert not stage_report_is_stable({**good, "grasp": None})
    assert not stage_report_is_stable({**good, "status": "failed"})
    assert not stage_report_is_stable(None)


def test_extract_last_json_object_prefers_the_mission_report():
    text = (
        'noise {"a": 1}\n'
        '{"status": "success", "final_state": "FINAL_STOP"}\n'
        '{"trailing": true}\n'
    )
    assert extract_last_json_object(text) == {"status": "success", "final_state": "FINAL_STOP"}
    # ANSI colouring from remote tools must not defeat parsing.
    assert extract_last_json_object('\x1b[32m{"b": 2}\x1b[0m') == {"b": 2}
    assert extract_last_json_object("nothing here") is None


def test_checkpoint_round_trip_and_rejection(tmp_path):
    path = tmp_path / "mission.json"
    assert load_checkpoint(path) is None
    atomic_write_json(path, {"version": 1, "phase": Phase.AT_PICKUP_TABLE.value})
    assert load_checkpoint(path)["phase"] == "AT_PICKUP_TABLE"
    path.write_text(json.dumps({"no_phase": True}), encoding="utf-8")
    with pytest.raises(MissionError, match="invalid mission checkpoint"):
        load_checkpoint(path)


def test_mission_run_lock_is_exclusive(tmp_path):
    path = tmp_path / ".lock"
    held = MissionRunLock(path)
    held.__enter__()
    try:
        # A second run must refuse while the first still holds the lock.
        with pytest.raises(MissionError, match="another mission run"), MissionRunLock(path):
            pass
    finally:
        held.__exit__(None, None, None)
    # Released on exit, so the next run can take it.
    with MissionRunLock(path):
        pass


def test_strategy_reports_dry_run_and_both_heights(tmp_path):
    plan = strategy(config(), tmp_path / "absent.json")
    assert plan["status"] == "dry_run" and plan["motion_enabled"] is False
    assert plan["travel_spine_m"] == TRAVEL_HEIGHT_M
    assert plan["grasp_spine_m"] == GRASP_HEIGHT_M
    assert plan["existing_checkpoint"] is None
    joined = " | ".join(plan["steps"])
    # The whole round trip, in order: grasp, carry out, touch, come back, leave.
    milestones = [
        "cup stage",
        "bowl stage",
        "carry both objects to the letter side",
        "lower spine to 0.468 m at the letter table",
        "test placement",
        "drive back to the pickup side",
        "step-20 route",
        "lower spine to 0.468 m at the placement table",
        "final placement",
    ]
    positions = [joined.index(text) for text in milestones]
    assert positions == sorted(positions), "strategy steps are out of order"
    # The spine must go up for every drive: once to depart, once per return leg.
    assert joined.count(f"raise spine to {TRAVEL_HEIGHT_M:.3f} m") == 3


def test_main_without_execute_starts_nothing(tmp_path, capsys):
    code = main(["--checkpoint", str(tmp_path / "c.json"), "--log-dir", str(tmp_path)])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "dry_run"


def test_main_rejects_the_same_arm_for_both_objects(tmp_path):
    with pytest.raises(SystemExit):
        main(["--cup-arm", "left", "--bowl-arm", "left"])
