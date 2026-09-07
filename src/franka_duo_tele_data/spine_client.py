"""Franka Spine height client: switch on, move to an absolute height, verify it.

The spine carries both arms and the head ZED (``mobile_fr3_duo_v0_2.xacro``
mounts ``fr3_duo`` on ``franka_spine_mounting_point``), so its height decides
both the driving envelope and the grasp geometry.  Two heights matter here: a
raised travel height while the base navigates, and the calibrated grasp height
the ZED extrinsics were solved at.

Motion goes through ``/franka_spine_node/move_absolute`` and is always verified
by reading ``get_position`` back, because a goal that reports success but stops
short would silently move the whole grasp frame.  Dry-run by default; motion
needs ``--execute``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

SWITCH_ON_SERVICE = "/franka_spine_node/switch_on"
GET_POSITION_SERVICE = "/franka_spine_node/get_position"
MOVE_ABSOLUTE_ACTION = "/franka_spine_node/move_absolute"

# URDF prismatic limit of franka_spine_vertical_joint (lower 0.0, upper 0.85).
SPINE_MIN_M = 0.0
SPINE_MAX_M = 0.85
# initialize_spine_height.py on the arm host verifies against this same bound.
POSITION_TOLERANCE_M = 0.003

TRAVEL_HEIGHT_M = 0.700
GRASP_HEIGHT_M = 0.468


@dataclass(frozen=True)
class SpineMotion:
    """A validated absolute spine move."""

    position_m: float
    velocity_mps: float
    acceleration_mps2: float
    deceleration_mps2: float


def plan_motion(
    position_m: float,
    *,
    velocity_mps: float = 0.05,
    acceleration_mps2: float = 0.1,
    deceleration_mps2: float = 0.1,
) -> SpineMotion:
    """Validate an absolute spine target against the joint limit and the action contract."""
    values = {
        "position": position_m,
        "velocity": velocity_mps,
        "acceleration": acceleration_mps2,
        "deceleration": deceleration_mps2,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"spine {name} must be finite")
    if not SPINE_MIN_M <= float(position_m) <= SPINE_MAX_M:
        raise ValueError(f"spine position must be within [{SPINE_MIN_M}, {SPINE_MAX_M}] m")
    # MoveAbsolute.action documents all three as strictly positive.
    for name in ("velocity", "acceleration", "deceleration"):
        if float(values[name]) <= 0:
            raise ValueError(f"spine {name} must be positive")
    return SpineMotion(
        float(position_m), float(velocity_mps), float(acceleration_mps2), float(deceleration_mps2)
    )


def height_reached(measured_m: float, target_m: float, tolerance_m: float = POSITION_TOLERANCE_M) -> bool:
    """True when the measured spine height proves the commanded target."""
    if not math.isfinite(float(measured_m)) or not math.isfinite(float(target_m)):
        return False
    if float(tolerance_m) <= 0:
        raise ValueError("tolerance must be positive")
    return abs(float(measured_m) - float(target_m)) <= float(tolerance_m)


def report_is_stable(
    report: dict | None, target_m: float, tolerance_m: float = POSITION_TOLERANCE_M
) -> bool:
    """True when a spine report proves the requested height was actually reached."""
    if not isinstance(report, dict) or report.get("status") != "success":
        return False
    try:
        commanded = float(report["target_position_m"])
        measured = float(report["measured_position_m"])
    except (KeyError, TypeError, ValueError):
        return False
    if not height_reached(commanded, target_m, 1e-9):
        return False
    return height_reached(measured, target_m, tolerance_m)


# --------------------------------------------------------------------------- ROS runtime


def move_spine(node, motion: SpineMotion, *, tolerance_m: float = POSITION_TOLERANCE_M) -> dict:
    """Switch on, move to ``motion.position_m`` and prove the height by reading it back."""
    import rclpy
    from franka_spine_msgs.action import MoveAbsolute
    from franka_spine_msgs.srv import GetPosition, SwitchOn
    from rclpy.action import ActionClient

    switch_on = node.create_client(SwitchOn, SWITCH_ON_SERVICE)
    get_position = node.create_client(GetPosition, GET_POSITION_SERVICE)
    move = ActionClient(node, MoveAbsolute, MOVE_ABSOLUTE_ACTION)

    def call(client, request, name, timeout_s=5.0):
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError(f"spine service unavailable: {name}")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"spine service timeout: {name}")
        return future.result()

    def position() -> float:
        response = call(get_position, GetPosition.Request(), GET_POSITION_SERVICE)
        if not response.success:
            raise RuntimeError("spine position query failed")
        return float(response.position)

    start = position()
    switched = call(switch_on, SwitchOn.Request(), SWITCH_ON_SERVICE)
    if not switched.success:
        raise RuntimeError(f"spine switch-on failed: {switched.message}")
    if not move.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("spine move action unavailable")

    goal = MoveAbsolute.Goal()
    goal.position = motion.position_m
    goal.velocity = motion.velocity_mps
    goal.acceleration = motion.acceleration_mps2
    goal.deceleration = motion.deceleration_mps2
    send = move.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send, timeout_sec=5.0)
    handle = send.result() if send.done() else None
    if handle is None or not handle.accepted:
        raise RuntimeError("spine height goal rejected")

    # A full-stroke move at the slowest useful speed still finishes well inside
    # this bound; a longer wait would only hide a stuck carriage.
    travel_s = abs(motion.position_m - start) / motion.velocity_mps
    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=travel_s + 20.0)
    if not result_future.done() or result_future.result() is None:
        raise RuntimeError("spine height goal timeout")
    result = result_future.result().result
    if not result.success:
        raise RuntimeError(f"spine move failed: {result.error}")

    measured = position()
    if not height_reached(measured, motion.position_m, tolerance_m):
        raise RuntimeError(
            f"spine stopped at {measured:.6f} m, commanded {motion.position_m:.6f} m"
        )
    return {
        "status": "success",
        "moved": True,
        "start_position_m": start,
        "target_position_m": motion.position_m,
        "measured_position_m": measured,
        "position_error_m": measured - motion.position_m,
        "stop_by": result.stop_by,
    }


def run(args) -> int:
    motion = plan_motion(
        args.target_m,
        velocity_mps=args.velocity,
        acceleration_mps2=args.acceleration,
        deceleration_mps2=args.deceleration,
    )
    if not args.execute:
        # Validate and report without importing ROS, so the plan can be checked
        # anywhere.
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "motion_enabled": False,
                    "target_position_m": motion.position_m,
                    "velocity_mps": motion.velocity_mps,
                },
                indent=1,
            ),
            flush=True,
        )
        return 0

    import rclpy

    rclpy.init()
    node = rclpy.create_node("franka_duo_spine_height")
    try:
        report = move_spine(node, motion, tolerance_m=args.tolerance_m)
        print(json.dumps(report, indent=1), flush=True)
        return 0
    except BaseException as exc:  # noqa: BLE001 - report and fail, never move on
        print(json.dumps({"status": "failed", "error": repr(exc)}, indent=1), flush=True)
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target-m",
        type=float,
        required=True,
        help=f"absolute height; travel {TRAVEL_HEIGHT_M}, grasp {GRASP_HEIGHT_M}",
    )
    parser.add_argument("--velocity", type=float, default=0.05)
    parser.add_argument("--acceleration", type=float, default=0.1)
    parser.add_argument("--deceleration", type=float, default=0.1)
    parser.add_argument("--tolerance-m", type=float, default=POSITION_TOLERANCE_M)
    parser.add_argument("--execute", action="store_true", help="required to move the spine")
    args = parser.parse_args(argv)
    if args.tolerance_m <= 0:
        parser.error("tolerance must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
