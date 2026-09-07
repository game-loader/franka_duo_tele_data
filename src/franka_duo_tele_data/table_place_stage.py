"""Put a held cup or bowl down on a table, as the mirror image of the grasp.

Two modes, both driven through the joint servo into the site
``joint_impedance_controller`` -- the same path the grasp uses, never PTP:

``--mode test``
    Touch the table, open, close again and lift the object back up.  Used at the
    letter-search end point, where the object must reach the table but is then
    carried onward.

``--mode final``
    Touch the table, open and lift the empty gripper away, leaving the object
    behind.  Used at the step-20 placement stop.

No camera is involved.  X/Y come from the arm's current pose: the base has
driven somewhere else since the grasp, so an earlier grasp point would send the
arm to the wrong place.  Only z changes on the way down, which is what
``move_vertical`` does on the arm host today.  Dry-run by default; robot output
needs ``--publish --enable-robot``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .cartesian_chunk import pose_distance
from .config_io import load_mapping
from .grasp_cup_bowl import ARM_SLICE, GRIPPER_INDEX, SIDES, build_place_rows, check_rows, release_clearance_m
from .joint_servo_client import parse_status
from .rgb20d_io import RGB20DContract
from .table_grasp_stage import ApproachLimits, StageRunner, load_stage_poses

MODES = ("test", "final")


def plan_placement(
    state: np.ndarray,
    arm: str,
    *,
    mode: str,
    table_z: float,
    object_height_m: float,
    grasp_depth_m: float,
    lift_m: float,
    limits: ApproachLimits,
    margin_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rows for one placement plus the geometry that produced them."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if state[GRIPPER_INDEX[arm]] > 0.5:
        # 0 is closed, 1 is open in the 20D contract: an open gripper holds nothing.
        raise ValueError(
            f"{arm} gripper reads open ({state[GRIPPER_INDEX[arm]]:.2f}); "
            f"there is nothing to place"
        )
    clearance = release_clearance_m(object_height_m, grasp_depth_m)
    rows = build_place_rows(
        state,
        arm,
        table_z=table_z,
        clearance_m=clearance,
        lift_m=lift_m,
        step_m=limits.step_m,
        step_rad=limits.step_rad,
        settle_rows=limits.settle_rows,
        regrasp=mode == "test",
        margin_m=margin_m,
    )
    start_z = float(state[ARM_SLICE[arm]][2])
    place_z = float(rows[0][ARM_SLICE[arm]][2]) if len(rows) else start_z
    return rows, {
        "mode": mode,
        "arm": arm,
        "table_z": float(table_z),
        "object_height_m": float(object_height_m),
        "grasp_depth_m": float(grasp_depth_m),
        "release_clearance_m": clearance,
        "start_ee_z": start_z,
        "placement_ee_z": float(table_z) + clearance + float(margin_m),
        "descent_m": start_z - (float(table_z) + clearance + float(margin_m)),
        "regrasp": mode == "test",
        "first_row_ee_z": place_z,
    }


def run(args) -> int:
    if args.publish != args.enable_robot:
        raise ValueError("robot publication requires both --publish and --enable-robot")
    config = load_mapping(args.config)
    _poses, limits, _spine = load_stage_poses(args.stage_poses)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32MultiArray, String

    from .replay_rgb20d import check_joint_servo_controllers
    from .rgb20d_io import RGB20DCache, RobotStateReader

    contract = RGB20DContract(args.dataset)
    contract.action_spec = dataclasses.replace(
        contract.action_spec,
        workspace_min=tuple(config["workspace_min"]),
        workspace_max=tuple(config["workspace_max"]),
    )
    latest: dict[str, Any] = {"status": None, "status_ns": 0}
    lock = threading.Lock()
    rclpy.init()
    node = rclpy.create_node("franka_duo_table_place_stage")
    cache = RGB20DCache(history_size=120)
    reader = RobotStateReader(cache, contract, max_age_ms=config["max_input_age_ms"])
    for side in SIDES:
        node.create_subscription(
            PoseStamped,
            config["topics"][f"{side}_pose"],
            getattr(cache, f"store_{side}_pose"),
            qos_profile_sensor_data,
        )
        node.create_subscription(
            JointState,
            config["topics"][f"{side}_gripper"],
            getattr(cache, f"store_{side}_gripper_states"),
            qos_profile_sensor_data,
        )

    def on_status(msg):
        with lock:
            latest["status"], latest["status_ns"] = msg.data, time.monotonic_ns()

    node.create_subscription(String, config["joint_servo_status_topic"], on_status, 10)
    publisher = (
        node.create_publisher(Float32MultiArray, config["joint_servo_chunk_topic"], 10)
        if args.publish
        else None
    )
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    def status():
        with lock:
            text, ns = latest["status"], latest["status_ns"]
        if text is None:
            return None
        if time.monotonic_ns() - ns > 500_000_000:
            raise TimeoutError("joint servo status expired")
        value = parse_status(text)
        if value.fault:
            raise RuntimeError(f"joint servo fault: {value.fault_reason}")
        return value

    runner = StageRunner(node, config, contract, publisher, status, args.speed, args.append_lead_steps)
    record: dict[str, Any] = {
        "schema": "franka_duo_table_place_stage_v1",
        "published": bool(publisher),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if publisher is not None:
            check_joint_servo_controllers(node, config, args.speed)
        state = reader.next(timeout_s=3).state
        rows, geometry = plan_placement(
            state,
            args.arm,
            mode=args.mode,
            table_z=args.table_z,
            object_height_m=args.object_height_m,
            grasp_depth_m=args.grasp_depth_m,
            lift_m=args.lift_m,
            limits=limits,
            margin_m=args.margin_m,
        )
        record |= geometry
        stats = check_rows(
            rows,
            state,
            contract,
            max_step_m=config["max_target_step_m"],
            max_step_rad=config["max_target_step_rad"],
            first_offset_m=limits.max_first_offset_m,
            first_offset_rad=limits.max_first_offset_rad,
        )
        record |= stats
        print(json.dumps({"stage": "plan", **geometry, **stats}), flush=True)
        record["place"] = runner.execute(rows, f"place_{args.mode}")
        record["rows"] = rows.tolist()
        if publisher is not None:
            final = reader.next(timeout_s=3)
            end_pos, end_rot = pose_distance(final.state, rows[-1])
            record["end_position_error_m"] = end_pos.tolist()
            record["end_rotation_error_rad"] = end_rot.tolist()
            record["grippers"] = final.state[18:].tolist()
            held = final.state[GRIPPER_INDEX[args.arm]]
            if args.mode == "test" and held > 0.5:
                raise RuntimeError(
                    f"test placement ended with the {args.arm} gripper open ({held:.2f}); "
                    f"the object was left behind instead of being carried on"
                )
        record["status"] = "success"
        return 0
    except KeyboardInterrupt:
        record["status"] = "stopped_by_user"
        print(
            json.dumps({"stopped_by_user": True, "note": "accepted plan finishes; servo keeps holding"}),
            flush=True,
        )
        return 130
    except BaseException as exc:  # noqa: BLE001 - always leave a report behind
        record["status"] = "failed"
        record["error"] = repr(exc)
        print(json.dumps({"status": "failed", "error": repr(exc)}), flush=True)
        return 1
    finally:
        args.output.write_text(json.dumps(record, indent=1) + "\n")
        if rclpy.ok():
            rclpy.shutdown()
        thread.join(timeout=2)
        node.destroy_node()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/tmr_rgb20d.yaml"))
    parser.add_argument("--stage-poses", type=Path, default=Path("configs/grasp_stage_poses.json"))
    parser.add_argument(
        "--mode",
        choices=MODES,
        required=True,
        help="test: touch, open, close, carry on. final: touch, open, leave it",
    )
    parser.add_argument("--arm", choices=SIDES, required=True)
    parser.add_argument("--target", choices=("cup", "bowl"), default="cup")
    parser.add_argument(
        "--table-z",
        type=float,
        required=True,
        help="destination table top in the midpoint base frame; measure it at the "
        "grasp spine height, it is not the same table as the pick",
    )
    parser.add_argument("--object-height-m", type=float, default=None)
    parser.add_argument("--grasp-depth-m", type=float, default=None)
    parser.add_argument("--lift-m", type=float, default=0.10)
    parser.add_argument(
        "--margin-m",
        type=float,
        default=0.005,
        help="extra height left above the table so the object is set down, not pressed",
    )
    parser.add_argument("--speed", type=float, default=0.1, help="must equal the servo playback_speed")
    parser.add_argument("--append-lead-steps", type=int, default=3)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/table_place_stage.json"))
    args = parser.parse_args(argv)
    if not 0 < args.speed <= 1:
        parser.error("speed must be in (0,1]")
    if args.margin_m < 0:
        parser.error("margin must not be negative")
    # Same defaults as the grasp, so a placement matches the pick it followed.
    if args.object_height_m is None:
        args.object_height_m = 0.080 if args.target == "cup" else 0.045
    if args.grasp_depth_m is None:
        args.grasp_depth_m = 0.055 if args.target == "cup" else 0.025
    if not 0 < args.grasp_depth_m < args.object_height_m:
        parser.error("grasp depth must be positive and less than the object height")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
