"""Table grasp stage: lower the spine, clear the view, detect, pose up, grasp.

This is the segment that runs after the base has stopped at the table.  Every
arm motion in it goes through the joint servo into the site
``joint_impedance_controller`` -- the same path the grasp chunk uses.  PTP is
deliberately not used: the impedance controller is what the dataset was
recorded under, and mixing a position motion generator into the sequence would
reintroduce the discontinuity reflexes that path was chosen to avoid.

Stage order::

    spine -> 0.468 m (verified)  ->  camera_clear pose  ->  YOLO detect
      ->  grasp_ready pose  ->  grasp chunk

Detection failure stops the stage; it never falls through to a grasp with no
target.  Dry-run by default; robot output needs ``--publish --enable-robot``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .cartesian_chunk import pose_distance
from .config_io import load_mapping
from .grasp_cup_bowl import (
    ARM_SLICE,
    DATASET_GRASP_ROT6D,
    GRIPPER_INDEX,
    SIDES,
    _segment,
    build_grasp_rows,
    check_rows,
    locate_target,
    pick_cup_and_bowl,
    run_yolo,
    spoon_centroid,
    summarize_detections,
)
from .joint_servo_client import chunk_payload, parse_status
from .rgb20d_io import RGB20DContract
from .spine_client import GRASP_HEIGHT_M, height_reached, move_spine, plan_motion
from .zed_pnp_calib import (
    extrinsics_for_arm,
    intrinsics_from_camera_info,
    load_calibration,
)

STAGE_SCHEMA = "franka_duo_grasp_stage_poses_v1"
STAGE_NAMES = ("travel_stow", "camera_clear", "grasp_ready")


# --------------------------------------------------------------------------- stage poses (ROS-free)


@dataclasses.dataclass(frozen=True)
class StagePose:
    name: str
    action: np.ndarray  # 18D: left xyz+rot6d, right xyz+rot6d
    grippers: tuple[float, float]

    def to_state(self) -> np.ndarray:
        """The 20D contract row this pose commands."""
        return np.concatenate([self.action, np.asarray(self.grippers, dtype=np.float32)]).astype(
            np.float32
        )


@dataclasses.dataclass(frozen=True)
class ApproachLimits:
    step_m: float = 0.01
    step_rad: float = 0.08
    settle_rows: int = 6
    max_first_offset_m: float = 0.03
    max_first_offset_rad: float = 0.3


def load_stage_poses(path: Path) -> tuple[dict[str, StagePose | None], ApproachLimits, float]:
    """Load stage poses. Untaught entries stay ``None`` so a stage that does not
    need them still runs; :func:`require_pose` rejects using one."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != STAGE_SCHEMA:
        raise ValueError("expected a grasp stage pose document")
    spine_height_m = float(document["spine_height_m"])
    if not math.isfinite(spine_height_m):
        raise ValueError("stage spine height must be finite")
    poses: dict[str, StagePose | None] = {}
    for name in STAGE_NAMES:
        entry = document["poses"][name]
        action = entry.get("action")
        if action is None:
            poses[name] = None
            continue
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (18,) or not np.isfinite(values).all():
            raise ValueError(f"stage pose '{name}' must be a finite 18D action")
        grippers = tuple(float(v) for v in entry.get("grippers", (1.0, 1.0)))
        if len(grippers) != 2 or not all(0.0 <= v <= 1.0 for v in grippers):
            raise ValueError(f"stage pose '{name}' grippers must be two values in [0, 1]")
        poses[name] = StagePose(name, values, grippers)
    approach = document.get("approach", {})
    limits = ApproachLimits(
        step_m=float(approach.get("step_m", 0.01)),
        step_rad=float(approach.get("step_rad", 0.08)),
        settle_rows=int(approach.get("settle_rows", 6)),
        max_first_offset_m=float(approach.get("max_first_offset_m", 0.03)),
        max_first_offset_rad=float(approach.get("max_first_offset_rad", 0.3)),
    )
    if limits.step_m <= 0 or limits.step_rad <= 0 or limits.settle_rows < 0:
        raise ValueError("approach limits must be positive")
    return poses, limits, spine_height_m


def require_pose(poses: dict[str, StagePose | None], name: str) -> StagePose:
    """The taught pose, or a refusal: an untaught posture must never be guessed."""
    pose = poses.get(name)
    if pose is None:
        raise ValueError(
            f"stage pose '{name}' has not been taught yet; record it on the robot "
            f"and write it into the stage pose file before running this stage"
        )
    return pose


def build_transfer_rows(state: np.ndarray, pose: StagePose, limits: ApproachLimits) -> np.ndarray:
    """20D rows moving both arms from the current state to a stage pose.

    Both arms are interpolated together with the same bounded step the grasp
    uses, so the impedance controller never sees a jump at the hand-off.
    """
    target = pose.to_state()
    per_arm: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for side in SIDES:
        sl = ARM_SLICE[side]
        per_arm[side] = _segment(
            state[sl][:3].astype(np.float64),
            target[sl][:3].astype(np.float64),
            state[sl][3:9].astype(np.float64),
            target[sl][3:9].astype(np.float64),
            limits.step_m,
            limits.step_rad,
        )
    count = max(len(per_arm[side]) for side in SIDES)
    rows = np.tile(state.astype(np.float32), (count + limits.settle_rows, 1))
    for side in SIDES:
        sl = ARM_SLICE[side]
        steps = per_arm[side]
        for index in range(len(rows)):
            # A shorter arm segment holds its endpoint while the other finishes.
            position, rotation = steps[min(index, len(steps) - 1)]
            rows[index, sl] = np.concatenate([position, rotation]).astype(np.float32)
        rows[:, GRIPPER_INDEX[side]] = pose.grippers[SIDES.index(side)]
    return rows


# --------------------------------------------------------------------------- ROS runtime


class StageRunner:
    """Publishes chunks to the joint servo and waits for them to finish."""

    def __init__(self, node, config, contract, publisher, status, speed: float, lead_steps: int):
        self.node = node
        self.config = config
        self.contract = contract
        self.publisher = publisher
        self.status = status
        self.speed = float(speed)
        self.lead_steps = int(lead_steps)

    def execute(self, rows: np.ndarray, label: str) -> dict:
        """Send one absolute-step chunk and wait until the servo holds at its end."""
        from std_msgs.msg import Float32MultiArray, MultiArrayDimension

        if self.publisher is None:
            return {"stage": label, "rows": int(len(rows)), "published": False}
        current = self.status()
        if current is None:
            raise RuntimeError(f"{label}: joint servo status unavailable")
        if current.started and not current.holding:
            raise RuntimeError(f"{label}: joint servo is still executing a chunk")
        start_step = int(math.ceil(current.step)) + self.lead_steps if current.started else 0
        link0 = np.stack([self.contract.action_spec.to_link0_action(row) for row in rows])
        data, dims, offset = chunk_payload(link0, start_step)
        message = Float32MultiArray(data=data)
        message.layout.dim = [
            MultiArrayDimension(label="rows", size=dims[0], stride=dims[0] * dims[1]),
            MultiArrayDimension(label="action", size=dims[1], stride=dims[1]),
        ]
        message.layout.data_offset = offset
        before = current.chunks
        self.publisher.publish(message)
        end_step = start_step + len(rows) - 1

        deadline = time.monotonic() + 3
        while True:
            current = self.status()
            if current is not None and current.chunks > before:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"{label}: servo did not accept the chunk; inspect its IK log")
            time.sleep(0.02)

        deadline = time.monotonic() + 20 + len(rows) / (self.contract.fps * self.speed) * 1.5
        max_track = 0.0
        while True:
            time.sleep(0.05)
            current = self.status()
            if current is not None:
                max_track = max(max_track, current.tracking_error_rad)
                if current.holding and current.last_step >= end_step and current.step >= end_step:
                    break
            if time.monotonic() > deadline:
                raise TimeoutError(f"{label}: servo did not finish the chunk in time")
        return {
            "stage": label,
            "rows": int(len(rows)),
            "published": True,
            "start_step": start_step,
            "end_step": end_step,
            "max_servo_tracking_error_rad": max_track,
        }


def prepare(args) -> tuple[dict[str, StagePose | None], Mapping[str, Any], np.ndarray, ApproachLimits, dict[str, Any]]:
    """Validate every input before ROS is touched, so mistakes surface anywhere."""
    if args.publish != args.enable_robot:
        raise ValueError("robot publication requires both --publish and --enable-robot")

    poses, limits, stage_spine_m = load_stage_poses(args.stage_poses)
    if args.pose_only is not None:
        require_pose(poses, args.pose_only)
    else:
        # camera_clear and grasp_ready are optional. Until they are taught the
        # stage detects and grasps from whatever pose the arms are already in --
        # build_grasp_rows starts from the current pose anyway. An untaught
        # camera_clear only means the view is not actively cleared first, so the
        # detection report must be checked for a blocked tray.
        if not height_reached(args.spine_target_m, stage_spine_m, 1e-6):
            raise ValueError(
                f"stage poses were taught at spine {stage_spine_m} m but the target is "
                f"{args.spine_target_m} m; retune the poses or pass the matching height"
            )
        plan_motion(args.spine_target_m, velocity_mps=args.spine_velocity)
    config = load_mapping(args.config)
    calibration = load_calibration(args.calibration)
    t_base_from_cam = extrinsics_for_arm(calibration, args.arm)
    return poses, config, t_base_from_cam, limits, calibration


def run(args) -> int:
    poses, config, t_base_from_cam, limits, calibration = prepare(args)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from std_msgs.msg import Float32MultiArray, String

    from .replay_rgb20d import check_joint_servo_controllers
    from .rgb20d_io import RGB20DCache, RobotStateReader
    from .ros_utils import image_msg_to_rgb

    spine_motion = plan_motion(args.spine_target_m, velocity_mps=args.spine_velocity)
    contract = RGB20DContract(args.dataset)
    contract.action_spec = dataclasses.replace(
        contract.action_spec,
        workspace_min=tuple(config["workspace_min"]),
        workspace_max=tuple(config["workspace_max"]),
    )

    latest: dict[str, Any] = {"image": None, "info": None, "status": None, "status_ns": 0}
    lock = threading.Lock()
    rclpy.init()
    node = rclpy.create_node("franka_duo_table_grasp_stage")
    cache = RGB20DCache(history_size=120)
    reader = RobotStateReader(cache, contract, max_age_ms=config["max_input_age_ms"])

    def store(key):
        def cb(msg):
            with lock:
                latest[key] = (time.monotonic(), msg)

        return cb

    node.create_subscription(Image, config["topics"]["head"], store("image"), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, args.camera_info_topic, store("info"), qos_profile_sensor_data)
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

    runner = StageRunner(
        node, config, contract, publisher, status, args.speed, args.append_lead_steps
    )
    record: dict[str, Any] = {
        "schema": "franka_duo_table_grasp_stage_v1",
        "target": args.target,
        "arm": args.arm,
        "spine_target_m": spine_motion.position_m,
        "stages": [],
        "published": bool(publisher),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def transfer(name: str) -> dict | None:
        """Move both arms to a taught posture, or skip it when untaught."""
        if poses.get(name) is None:
            report = {"stage": name, "skipped": "not taught"}
            record["stages"].append(report)
            print(json.dumps(report), flush=True)
            return None
        observation = reader.next(timeout_s=3)
        rows = build_transfer_rows(observation.state, require_pose(poses, name), limits)
        stats = check_rows(
            rows,
            observation.state,
            contract,
            max_step_m=config["max_target_step_m"],
            max_step_rad=config["max_target_step_rad"],
            first_offset_m=limits.max_first_offset_m,
            first_offset_rad=limits.max_first_offset_rad,
        )
        report = runner.execute(rows, name) | stats
        record["stages"].append(report)
        print(json.dumps(report), flush=True)
        return report

    try:
        if publisher is not None:
            check_joint_servo_controllers(node, config, args.speed)

        if args.pose_only is not None:
            # Reach one taught posture and stop: used to stow the arms inside the
            # navigation footprint before the base drives.
            record["pose_only"] = args.pose_only
            transfer(args.pose_only)
            record["status"] = "success"
            return 0

        # 1. Lower the spine. Both arms and the head camera ride on its carriage,
        #    so nothing below may run until the height is proven.
        if args.publish:
            spine_report = move_spine(node, spine_motion, tolerance_m=args.spine_tolerance_m)
        else:
            spine_report = {"status": "dry_run", "target_position_m": spine_motion.position_m}
        record["spine"] = spine_report
        print(json.dumps({"stage": "spine", **spine_report}), flush=True)

        # 2. Clear the camera view before trusting any detection, when taught.
        transfer("camera_clear")

        # 3. Detect on the head ZED at the calibrated height.
        if args.image is not None:
            import cv2

            bgr = cv2.imread(str(args.image))
            if bgr is None:
                raise ValueError(f"cannot read {args.image}")
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            k_matrix = np.asarray(calibration["camera"]["K"], dtype=np.float64)
        else:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and (latest["image"] is None or latest["info"] is None):
                time.sleep(0.05)
            if latest["image"] is None or latest["info"] is None:
                raise TimeoutError("no ZED image/camera_info; check whether the camera host is up")
            k_matrix, _dist, _size = intrinsics_from_camera_info(latest["info"][1])
            if not np.allclose(k_matrix, np.asarray(calibration["camera"]["K"]), atol=1e-3):
                raise ValueError("live intrinsics differ from the calibration")
            with lock:
                rgb = image_msg_to_rgb(latest["image"][1])

        detections = run_yolo(rgb, args.weights, args.conf)
        objects = summarize_detections(rgb, detections)
        picked = pick_cup_and_bowl(objects)
        record["detections"] = [
            dataclasses.asdict(o) | {"rim_pixel": list(o.rim_pixel), "rim_axes_px": list(o.rim_axes_px)}
            for o in objects
        ]
        if args.target not in picked:
            # Stop here: a stage pose change is worthless without a target, and
            # descending on a guess is how the gripper hits the tray.
            raise RuntimeError(
                f"{args.target} not detected at the table; "
                f"detected: {[(o.cls, round(o.conf, 2)) for o in objects]}"
            )

        rim_z = (
            args.table_z
            + args.tray_thickness_m
            + (args.cup_height_m if args.target == "cup" else args.bowl_height_m)
        )
        state_for_base = reader.next(timeout_s=3).state
        grasp = locate_target(
            args.target,
            args.arm,
            picked,
            k_matrix=k_matrix,
            t_base_from_cam=t_base_from_cam,
            rim_z=rim_z,
            spoon_pixel=spoon_centroid(rgb, detections),
            arm_base_xy=state_for_base[ARM_SLICE[args.arm]][:2],
            bowl_rim_radius_m=args.bowl_rim_radius_m,
        )
        record["rim_centre_base"] = grasp.rim_centre_base.tolist()
        record["grasp_point_base"] = grasp.grasp_point_base.tolist()
        record["rim_z"] = rim_z
        print(
            json.dumps(
                {
                    "stage": "detect",
                    "target": args.target,
                    "rim_centre_base": [round(v, 4) for v in grasp.rim_centre_base.tolist()],
                    "grasp_point_base": [round(v, 4) for v in grasp.grasp_point_base.tolist()],
                }
            ),
            flush=True,
        )

        # 4. Grasp start posture, then the grasp itself.
        transfer("grasp_ready")

        state = reader.next(timeout_s=3).state
        rows = build_grasp_rows(
            state,
            grasp,
            grasp_rot6d=np.asarray(DATASET_GRASP_ROT6D[args.arm]),
            approach_height_m=args.approach_height_m,
            grasp_depth_below_rim_m=args.grasp_depth_m,
            lift_m=args.lift_m,
        )
        stats = check_rows(
            rows,
            state,
            contract,
            max_step_m=config["max_target_step_m"],
            max_step_rad=config["max_target_step_rad"],
            first_offset_m=limits.max_first_offset_m,
            first_offset_rad=limits.max_first_offset_rad,
        )
        record["grasp"] = runner.execute(rows, "grasp") | stats
        record["rows"] = rows.tolist()

        if publisher is not None:
            final = reader.next(timeout_s=3)
            end_pos, end_rot = pose_distance(final.state, rows[-1])
            record["end_position_error_m"] = end_pos.tolist()
            record["end_rotation_error_rad"] = end_rot.tolist()
            record["grippers"] = final.state[18:].tolist()
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
    parser.add_argument("--calibration", type=Path, default=Path("configs/zed_pnp_calibration.json"))
    parser.add_argument("--stage-poses", type=Path, default=Path("configs/grasp_stage_poses.json"))
    parser.add_argument("--camera-info-topic", default="/head_camera/zed/rgb/color/rect/camera_info")
    parser.add_argument("--weights", default="outputs/zed_pnp/yolo11n-seg.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--image", type=Path, help="saved frame instead of the live ZED image")
    parser.add_argument("--target", choices=("cup", "bowl"), default="cup")
    parser.add_argument("--arm", choices=SIDES, default="right")
    parser.add_argument(
        "--pose-only",
        choices=STAGE_NAMES,
        help="reach one taught posture and stop, skipping spine, detection and grasp",
    )
    parser.add_argument("--spine-target-m", type=float, default=GRASP_HEIGHT_M)
    parser.add_argument("--spine-velocity", type=float, default=0.05)
    parser.add_argument("--spine-tolerance-m", type=float, default=0.003)
    parser.add_argument("--table-z", type=float, default=-0.220, help="table top in the midpoint frame")
    parser.add_argument("--tray-thickness-m", type=float, default=0.010)
    parser.add_argument("--cup-height-m", type=float, default=0.080)
    parser.add_argument("--bowl-height-m", type=float, default=0.045)
    parser.add_argument("--bowl-rim-radius-m", type=float, default=0.0575)
    parser.add_argument("--approach-height-m", type=float, default=0.10)
    parser.add_argument("--grasp-depth-m", type=float, default=None)
    parser.add_argument("--lift-m", type=float, default=0.10)
    parser.add_argument("--speed", type=float, default=0.1, help="must equal the servo playback_speed")
    parser.add_argument("--append-lead-steps", type=int, default=3)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/table_grasp_stage.json"))
    args = parser.parse_args(argv)
    if not 0 < args.speed <= 1:
        parser.error("speed must be in (0,1]")
    if args.grasp_depth_m is None:
        args.grasp_depth_m = 0.055 if args.target == "cup" else 0.025
    height = args.cup_height_m if args.target == "cup" else args.bowl_height_m
    if not 0 < args.grasp_depth_m < height:
        parser.error("grasp depth must be positive and less than the object height")
    if args.image is not None and args.publish:
        parser.error("--image is a dry-run aid and cannot be combined with --publish")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
