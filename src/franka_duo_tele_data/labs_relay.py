"""Site-owned Labs link8 IK relay to the existing joint follower controllers.

No controller activation. Both publication gates are required. Defaults to a
read-only status monitor. Accepted chunks finish then hold; faults stop output.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import labs_joint_inference as joint_policy
from .labs_action_delta import absolute20_to_delta14
from .labs_episode import REPLAY_SCHEMA, load_episode
from .labs_ik import MoveItKDL
from .labs_inference import (
    CHUNK_REFERENCE,
    COMMAND_SCHEMA,
    COMMAND_TOPIC,
    RETURN_SCHEMA,
    SIDES,
    STATUS_SCHEMA,
    STATUS_TOPIC,
    TOPICS,
    LabsContract,
    episode_start,
    start_identity,
    validate_target,
)
from .labs_kinematics import joint_positions
from .labs_mcap_to_lerobot import validate_knuckle
from .labs_task_starts import load_task_starts
from .labs_tracking import track_chunk
from .ros_utils import _stamp_ns


def admit_command(command, contract, measured, *, now_ns=None, planning_complete=False):
    now = time.time_ns() if now_ns is None else now_ns
    returning = command.get("schema") in (RETURN_SCHEMA, REPLAY_SCHEMA)
    if command.get("model_hashes") != contract.model_hashes:
        raise ValueError("Incompatible Labs command/URDF/reference")
    if not isinstance(command.get("command_id"), str) or not 1 <= len(command["command_id"]) <= 128:
        raise ValueError("Invalid command id")
    if returning:
        stamp = command.get("created_ns")
        limit = 10_000_000_000 if planning_complete else 1_000_000_000
        if not isinstance(stamp, int) or not 0 <= now - stamp <= limit:
            raise ValueError("Stale or future created_ns")
        if command.get("schema") == REPLAY_SCHEMA:
            reference = np.asarray(command.get("reference_joints"), dtype=float)
            actual = np.asarray(measured, dtype=float)
            if (
                reference.shape != (14,)
                or not np.isfinite(reference).all()
                or actual.shape != (14,)
                or not np.isfinite(actual).all()
                or np.max(np.abs(reference - actual)) > 0.02
            ):
                raise ValueError("Robot moved since replay request")
        return
    if (
        command.get("schema") not in (COMMAND_SCHEMA, joint_policy.COMMAND_SCHEMA)
        or command.get("chunk_reference") != CHUNK_REFERENCE
        or command.get("mode", "policy_chunk") != "policy_chunk"
    ):
        raise ValueError("Incompatible Labs command/URDF/reference")
    if command["schema"] == joint_policy.COMMAND_SCHEMA and command.get("chunk_integration") != joint_policy.INTEGRATION:
        raise ValueError("Expected absolute joint16 integration")
    command_age = 10_000_000_000 if planning_complete else 1_000_000_000
    for name, limit in (("created_ns", command_age), ("observation_ns", 10_000_000_000)):
        stamp = command.get(name)
        if not isinstance(stamp, int) or not 0 <= now - stamp <= limit:
            raise ValueError(f"Stale or future {name}")
    reference = contract.validate_state(command["reference_state"])
    measured = np.asarray(measured, dtype=float)
    if measured.shape != (14,) or not np.isfinite(measured).all():
        raise ValueError("Missing measured joints")
    if np.max(np.abs(reference[20:] - measured)) > 0.02:
        raise ValueError("Robot moved since the request observation")


@dataclass
class JointPlan:
    joints: np.ndarray  # initial measured joints followed by every target row
    grippers: np.ndarray
    durations: np.ndarray
    preserve_grippers: bool = False
    diagnostics: dict | None = None

    @property
    def duration(self):
        return float(self.durations.sum())

    def sample(self, elapsed):
        ends = np.cumsum(self.durations)
        index = min(int(np.searchsorted(ends, max(0, elapsed), side="right")), len(ends) - 1)
        start = 0 if index == 0 else ends[index - 1]
        u = float(np.clip((elapsed - start) / self.durations[index], 0, 1))
        blend = 10 * u**3 - 15 * u**4 + 6 * u**5
        q = self.joints[index] + blend * (self.joints[index + 1] - self.joints[index])
        return q, self.grippers[index], elapsed >= self.duration


def stationary(velocity):
    value = np.asarray(velocity, dtype=float)
    return value.shape == (14,) and np.isfinite(value).all() and np.max(np.abs(value)) <= 0.02


def return_at_goal(plan, actual, velocity, *, position_tolerance=0.05):
    return np.max(np.abs(np.asarray(actual) - plan.joints[-1])) <= position_tolerance and stationary(velocity)


def settle_update(since, close, now, dwell):
    """Require continuous measured arrival; any excursion resets the dwell."""
    if not close:
        return None, False
    if since is None:
        return now, False
    return since, now - since >= dwell


def build_hold_plan(contract, measured, velocity):
    """Latch one stationary measured pose for follower activation; keep grippers untouched."""
    q = np.asarray(measured, dtype=float)
    bounds = np.concatenate([contract.fk[s].bounds for s in SIDES])
    if (
        q.shape != (14,)
        or not np.isfinite(q).all()
        or np.any(q < bounds[:, 0])
        or np.any(q > bounds[:, 1])
        or not stationary(velocity)
    ):
        raise ValueError("Activation hold requires stationary in-bounds joint feedback")
    return JointPlan(np.stack([q, q]), np.zeros((1, 2)), np.array([1.0]), preserve_grippers=True)


def controller_set_ready(active, *, allow_inactive=False):
    broadcasters = {"joint_state_broadcaster", "franka_robot_state_broadcaster"}
    return active == broadcasters | {"joint_follower_controller"} or (
        allow_inactive and active == broadcasters
    )


def build_return_plan(command, contract, measured, velocity, start_state, start_episode):
    """Reach the recorded q and open both grippers; never solve Cartesian IK."""
    if (
        command.get("schema") != RETURN_SCHEMA
        or command.get("mode") != "return_to_start"
        or start_state is None
        or command.get("start_episode") != start_episode
        or command.get("start_identity") != start_identity(start_state, start_episode, contract)
    ):
        raise ValueError("Return target does not match configured dataset episode start")
    if command.get("gripper_targets") != [1.0, 1.0]:
        raise ValueError("Return requires both gripper targets fully open: [1.0, 1.0]")
    initial = np.asarray(measured, dtype=float)
    target = contract.validate_state(start_state)[20:].astype(float)
    bounds = np.concatenate([contract.fk[s].bounds for s in SIDES])
    if (
        initial.shape != (14,)
        or not np.isfinite(initial).all()
        or np.any(initial < bounds[:, 0])
        or np.any(initial > bounds[:, 1])
    ):
        raise ValueError("Invalid measured joints for return")
    if not stationary(velocity):
        raise ValueError("Return requires fresh stationary joint velocity feedback")
    speed = command.get("speed")
    if (
        isinstance(speed, bool)
        or not isinstance(speed, (int, float))
        or not 0 < speed <= 1
        or command.get("execution_rate_hz") != 30 * speed
    ):
        raise ValueError("Return requires 0 < speed <= 1 and matching execution_rate_hz")
    distance = float(np.max(np.abs(target - initial)))
    # Return has no dataset row clock. Scale its whole quintic duration by the
    # same speed ratio as policy/replay; keep smooth 100 Hz target publication.
    base_duration = max(1.0, 1.875 * distance / 0.25, math.sqrt(10 / math.sqrt(3) * distance / 0.5))
    duration = base_duration / speed
    if duration > 90:
        raise ValueError("Return needs more than 90 s")
    return JointPlan(
        np.stack([initial, target]),
        np.ones((1, 2)),
        np.array([duration]),
        preserve_grippers=False,
        diagnostics={
            "tracking": "synchronized_quintic_return_v2",
            "gripper_targets": [1.0, 1.0],
            "speed": speed,
            "execution_rate_hz": 30 * speed,
            "base_duration_s": base_duration,
            "command_duration_s": duration,
            "velocity_limit_rad_s": 0.25 * speed,
            "acceleration_limit_rad_s2": 0.5 * speed**2,
            "max_velocity_rad_s": 1.875 * distance / duration,
            "max_acceleration_rad_s2": 10 / math.sqrt(3) * distance / duration**2,
        },
    )


def build_plan(command, contract, solvers, measured, *, commanded_start=None):
    """Validate and solve the ENTIRE chunk before any joint target is published."""
    if command.get("schema") == RETURN_SCHEMA or command.get("mode", "policy_chunk") != "policy_chunk":
        raise ValueError("Joint return must use the separate return planner")
    reference = contract.validate_state(command["reference_state"])
    targets = np.asarray(command["targets"], dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 20 or not 1 <= len(targets) <= 256:
        raise ValueError("Expected [H,20] Labs targets")
    speed = command.get("speed", 0)
    if not isinstance(speed, (int, float)) or not 0 < speed <= 1 or command.get("action_rate_hz") != 30:
        raise ValueError("Require 30 Hz source and 0 < speed <= 1")
    for target in targets:
        validate_target(target)
    initial = np.asarray(measured, dtype=float)
    if initial.shape != (14,) or not np.isfinite(initial).all():
        raise ValueError("Expected 14 finite measured joints")
    current_pose = contract.state({"left": initial[:7], "right": initial[7:]}, reference[18:20])[:20]
    steps = absolute20_to_delta14(targets, np.vstack((current_pose, targets[:-1])))
    extent = absolute20_to_delta14(targets, np.broadcast_to(reference, (len(targets), 34)))
    violations = []
    for side, offset in (("left", 0), ("right", 6)):
        # Seed-relative rotation covers all principal angles [0, pi], with
        # float32 tolerance. Per-step rotation is checked separately.
        for name, values, start, limit, unit in (
            ("step translation", steps, offset, 0.04, "m"),
            ("step rotation", steps, offset + 3, 0.35, "rad"),
            ("chunk translation extent", extent, offset, 0.60, "m"),
            ("chunk rotation extent", extent, offset + 3, math.pi + 1e-6, "rad"),
        ):
            magnitudes = np.linalg.norm(values[:, start : start + 3], axis=1)
            exceeded = np.flatnonzero(magnitudes > limit)
            if len(exceeded):
                peak = int(np.argmax(magnitudes))
                violations.append(
                    f"{side} {name}: max={magnitudes[peak]:.6f} {unit}, "
                    f"limit={limit:.6f} {unit}, first_row={int(exceeded[0])}, max_row={peak}"
                )
    if violations:
        raise ValueError(
            "Chunk Cartesian jump/extent exceeds Labs limits (zero-based rows): "
            + "; ".join(violations)
        )
    return solve_and_track(targets, contract, solvers, initial, commanded_start, 1 / (30 * speed))


def solve_and_track(
    targets,
    contract,
    solvers,
    initial,
    commanded_start,
    row_period,
    *,
    max_reference_lag=0.3,
    max_velocity=0.8,
    max_acceleration=2.0,
    max_jerk=20.0,
    target_velocity_weight=0.0,
):
    """Shared site dynamics for both policy clients and recorded episode replay."""
    start = initial if commanded_start is None else np.asarray(commanded_start, dtype=float)
    if start.shape != (14,) or not np.isfinite(start).all() or np.max(np.abs(start - initial)) > 0.15:
        raise ValueError("Invalid initial commanded hold for continuous tracking")
    joints = [start.copy()]
    for target in targets:
        row = []
        for side, p, q in (("left", 0, 0), ("right", 9, 7)):
            solution = solvers[side].solve(target[p : p + 9], joints[-1][q : q + 7])
            if not solution["success"]:
                raise ValueError(f"{side} IK failed: {solution.get('reason')}")
            result = np.asarray(solution["joint_positions"], dtype=float)
            bounds = np.asarray(contract.fk[side].bounds)
            if (
                result.shape != (7,)
                or not np.isfinite(result).all()
                or np.any(result < bounds[:, 0])
                or np.any(result > bounds[:, 1])
                or np.max(np.abs(result - joints[-1][q : q + 7])) > 0.25
            ):
                raise ValueError(f"{side} IK bounds/joint jump")
            from .labs_kinematics import pose_vector

            residual = absolute20_to_delta14(
                np.r_[target[p : p + 9], target[p : p + 9], [0, 0]],
                np.r_[pose_vector(contract.fk[side](result)), pose_vector(contract.fk[side](result)), [0, 0]],
            )
            if np.linalg.norm(residual[:3]) > 1e-4 or np.linalg.norm(residual[3:6]) > 1e-3:
                raise ValueError(f"{side} IK FK residual")
            row.extend(result)
        joints.append(np.asarray(row))
    return track_chunk(
        np.asarray(joints),
        targets[:, 18:].copy(),
        row_period,
        np.concatenate([contract.fk[s].bounds for s in SIDES]),
        max_reference_lag=max_reference_lag,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        max_jerk=max_jerk,
        target_velocity_weight=target_velocity_weight,
    )


def build_episode_plan(episode, contract, solvers, measured, commanded_start, rate=9.0):
    if not np.isfinite(rate) or not 0 < rate <= 30 or len(episode.targets) / rate > 90:
        raise ValueError("Episode playback requires 0 < rate <= 30 and duration <=90 s")
    initial = np.asarray(measured, dtype=float)
    if initial.shape != (14,) or not np.isfinite(initial).all():
        raise ValueError("Invalid measured joints")
    if np.max(np.abs(initial - episode.states[0, 20:])) > 0.05:
        raise ValueError("Return to the selected episode start before replay")
    current_pose = contract.state({"left": initial[:7], "right": initial[7:]}, episode.states[0, 18:20])[:20]
    steps = absolute20_to_delta14(episode.targets, np.vstack([current_pose, episode.targets[:-1]]))
    # An entire recording can move farther than a single inference chunk;
    # retain adjacent-step limits and check label deltas against each own state.
    for offset in (0, 6):
        if (
            np.any(np.linalg.norm(steps[:, offset : offset + 3], axis=1) > 0.04)
            or np.any(np.linalg.norm(steps[:, offset + 3 : offset + 6], axis=1) > 0.35)
            or np.any(np.linalg.norm(episode.actions[:, offset : offset + 3], axis=1) > 0.25)
            or np.any(np.linalg.norm(episode.actions[:, offset + 3 : offset + 6], axis=1) > 1.0)
        ):
            raise ValueError("Recorded episode Cartesian step/delta exceeds limits")
    return solve_and_track(
        episode.targets,
        contract,
        solvers,
        initial,
        commanded_start,
        1 / rate,
    )


def run(args):
    import rclpy
    from controller_manager_msgs.srv import ListControllers
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32, String

    contract = LabsContract(args.config, args.dataset)
    start_state = episode_start(args.dataset, args.start_episode, contract) if args.dataset else None
    task_starts = load_task_starts(args.config)
    enabled = args.publish and args.enable_robot
    rclpy.init()
    node = rclpy.create_node("labs_policy_relay")
    lock = threading.RLock()
    state = {
        "phase": "arming" if args.hold_current_on_start else "holding",
        "command_id": None,
        "fault": "",
        "pending": None,
        "plan": None,
        "start": 0.0,
        "last_q": None,
        "settled": None,
        "last_tick": time.monotonic(),
        "completed_start_identity": None,
        "active_start_identity": None,
    }
    feedback, gripper_feedback, controllers, futures = {}, {}, {}, {}
    velocities = {}
    follower_states = {}
    seen = set()
    outputs = {}
    for side in SIDES:
        if enabled:
            outputs[side] = (
                node.create_publisher(JointState, f"/{side}/follower/gello/joint_states", 1),
                node.create_publisher(
                    Float32, f"/{side}/follower/gripper/gripper_client/target_gripper_width_percent", 1
                ),
            )

        def on_joint(message, s=side):
            try:
                q = joint_positions(message.name, message.position, s)
                stamp = _stamp_ns(message)
                with lock:
                    feedback[s] = (q, stamp, time.monotonic())
                    velocities[s] = (
                        joint_positions(message.name, message.velocity, s)
                        if len(message.velocity) == len(message.name)
                        else None
                    )
            except Exception as exc:
                with lock:
                    state["fault"] = str(exc)

        node.create_subscription(JointState, TOPICS[f"{side}_q"], on_joint, qos_profile_sensor_data)

        def on_gripper(message, s=side):
            try:
                candidates = [
                    i for i, name in enumerate(message.name) if name.endswith("robotiq_85_left_knuckle_joint")
                ]
                if len(candidates) != 1 or len(message.name) != len(message.position):
                    raise ValueError(f"Invalid {s} gripper feedback")
                value = float(message.position[candidates[0]])
                validate_knuckle(value)
                with lock:
                    gripper_feedback[s] = (
                        float(np.clip(1 - value / 0.8, 0, 1)),
                        _stamp_ns(message),
                        time.monotonic(),
                    )
            except Exception as exc:
                with lock:
                    state["fault"] = str(exc)

        node.create_subscription(JointState, TOPICS[f"{side}_gripper"], on_gripper, qos_profile_sensor_data)
    for side in SIDES:

        def on_follower_state(message, s=side):
            with lock:
                follower_states[s] = message.data

        node.create_subscription(
            String,
            f"/{side}/joint_follower_controller/state",
            on_follower_state,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

    clients = {
        s: node.create_client(ListControllers, f"/{s}/controller_manager/list_controllers") for s in SIDES
    }
    status_pub = node.create_publisher(String, STATUS_TOPIC, 10)

    def poll_controllers():
        for side in SIDES:
            previous = futures.get(side)
            if previous is not None and not previous.done():
                continue
            if clients[side].service_is_ready():
                future = clients[side].call_async(ListControllers.Request())
                futures[side] = future

                def on_result(result, s=side):
                    try:
                        active = {c.name for c in result.result().controller if c.state == "active"}
                        with lock:
                            controllers[s] = (active, time.monotonic())
                    except Exception as exc:
                        with lock:
                            state["fault"] = str(exc)

                future.add_done_callback(on_result)

    def measured():
        now = time.monotonic()
        values = []
        for side in SIDES:
            if side not in feedback:
                raise RuntimeError(f"Missing {side} feedback")
            q, stamp, receipt = feedback[side]
            if stamp is None or not 0 <= time.time_ns() - stamp <= 150_000_000 or now - receipt > 0.15:
                raise RuntimeError(f"Stale {side} feedback")
            values.extend(q)
        return np.asarray(values)

    def measured_velocity():
        measured()  # q and dq came from the same fresh message.
        if any(velocities.get(s) is None for s in SIDES):
            raise RuntimeError("Missing measured joint velocity")
        return np.concatenate([velocities[s] for s in SIDES])

    def readiness(*, allow_inactive=False):
        if not enabled:
            return "robot output disabled"
        measured()
        if node.count_publishers(COMMAND_TOPIC) > 1:
            return "multiple inference clients"
        for side in SIDES:
            _, stamp, received = gripper_feedback.get(side, (None, None, 0))
            if (
                stamp is None
                or not 0 <= time.time_ns() - stamp <= 150_000_000
                or time.monotonic() - received > 0.15
            ):
                return f"{side}: missing/stale gripper feedback"
            active, received = controllers.get(side, (set(), 0))
            if time.monotonic() - received > 0.75 or not controller_set_ready(
                active, allow_inactive=allow_inactive
            ):
                return f"{side}: require only joint follower and state broadcasters active"
            for pub, topic in zip(
                outputs[side],
                (
                    f"/{side}/follower/gello/joint_states",
                    f"/{side}/follower/gripper/gripper_client/target_gripper_width_percent",
                ),
                strict=True,
            ):
                if node.count_publishers(topic) != 1 or pub.get_subscription_count() != 1:
                    return f"{side}: require exclusive output ownership and one subscriber: {topic}"
        return ""

    def on_command(message):
        with lock:
            try:
                if not enabled:
                    raise ValueError("Relay disabled")
                if state["fault"] or state["phase"] != "holding" or state["pending"] is not None:
                    raise ValueError("Relay is not ready for another chunk")
                command = json.loads(message.data)
                if command.get("command_id") in seen:
                    raise ValueError("Duplicate command id")
                reason = readiness()
                if reason:
                    raise ValueError(reason)
                admit_command(command, contract, measured())
                seen.add(command["command_id"])
                state.update(pending=command, phase="planning", command_id=command["command_id"])
            except Exception as exc:
                state["fault"] = str(exc)

    node.create_subscription(String, COMMAND_TOPIC, on_command, 1)

    def tick():
        with lock:
            try:
                arming = state["phase"] == "arming"
                reason = readiness(allow_inactive=arming)
                if (
                    arming
                    and state["plan"] is None
                    and not reason
                    and not state["fault"]
                    and stationary(measured_velocity())
                ):
                    q = measured()
                    plan = build_hold_plan(contract, q, measured_velocity())
                    state.update(
                        plan=plan, start=time.monotonic(), last_q=q, last_tick=time.monotonic(), settled=None
                    )
                active = state["plan"] is not None
                if active and reason:
                    raise RuntimeError(reason)
                if active and not state["fault"]:
                    if state["phase"] == "executing" and time.monotonic() - state["last_tick"] > 0.05:
                        raise RuntimeError("Relay scheduling gap exceeds 50 ms")
                    plan = state["plan"]
                    elapsed = time.monotonic() - state["start"]
                    q, grippers, ended = plan.sample(elapsed)
                    actual = measured()
                    if state["last_q"] is not None and np.max(np.abs(actual - state["last_q"])) > 0.15:
                        raise RuntimeError("Joint tracking error exceeds 0.15 rad")
                    for side, offset, grip in (("left", 0, 0), ("right", 7, 1)):
                        msg = JointState()
                        msg.header.stamp = node.get_clock().now().to_msg()
                        msg.name = [f"{side}_fr3_joint{i}" for i in range(1, 8)]
                        msg.position = q[offset : offset + 7].tolist()
                        outputs[side][0].publish(msg)
                        if not plan.preserve_grippers:
                            outputs[side][1].publish(Float32(data=float(grippers[grip])))
                    state["last_q"] = q
                    state["last_tick"] = time.monotonic()
                    if arming:
                        close = (
                            not readiness()
                            and all(follower_states.get(s) == "FOLLOWING" for s in SIDES)
                            and return_at_goal(plan, actual, measured_velocity(), position_tolerance=0.01)
                        )
                        state["settled"], complete = settle_update(
                            state["settled"], close, time.monotonic(), 0.5
                        )
                        if complete:
                            state["phase"] = "holding"
                    if ended and state["phase"] == "executing":
                        # Policy/replay endpoints allow steady-state position error.
                        # A recorded-start return still verifies that pose before
                        # acknowledging completed_start_identity.
                        returning = state["active_start_identity"] is not None
                        velocity = measured_velocity()
                        close = return_at_goal(plan, actual, velocity) if returning else stationary(velocity)
                        state["settled"], complete = settle_update(
                            state["settled"], close, time.monotonic(), 0.5
                        )
                        if complete:
                            state["phase"] = "holding"
                            state["completed_start_identity"] = state["active_start_identity"]
                        if elapsed > plan.duration + 3 and state["phase"] != "holding":
                            if returning:
                                raise RuntimeError("Robot did not settle at recorded return target")
                            raise RuntimeError("Robot did not stop at chunk endpoint")
            except Exception as exc:
                # DDS discovery/missing feedback before the first output is a
                # readiness condition; faults latch once a hold/trajectory exists.
                if state["plan"] is not None:
                    state["fault"] = str(exc)

    def publish_status():
        with lock:
            try:
                reason = readiness()
            except Exception as exc:
                reason = str(exc)
            message = {
                "schema": STATUS_SCHEMA,
                "completed_start_identity": state["completed_start_identity"],
                "follower_states": follower_states.copy(),
                "holding_target": state["last_q"].tolist() if state["last_q"] is not None else None,
                "enabled": enabled,
                "model_hashes": contract.model_hashes,
                "ready": not reason and not state["fault"],
                "reason": reason,
                "phase": state["phase"],
                "fault": state["fault"],
                "supported_command_schemas": [COMMAND_SCHEMA, joint_policy.COMMAND_SCHEMA, RETURN_SCHEMA, REPLAY_SCHEMA],
                "task_start_identities": {str(k): v.identity for k, v in task_starts.items()},
                "command_id": state["command_id"],
                "rows": len(state["plan"].grippers) if state["plan"] else 0,
                "duration_s": state["plan"].duration if state["plan"] else 0,
                "tracking": getattr(state["plan"], "diagnostics", None),
            }
            status_pub.publish(String(data=json.dumps(message)))

    node.create_timer(0.25, poll_controllers)
    node.create_timer(0.01, tick)
    node.create_timer(0.05, publish_status)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    try:
        with contextlib.ExitStack() as stack:
            solvers = {}  # Initialize IK lazily for policy chunks, never for joint return.
            while rclpy.ok():
                with lock:
                    pending = state["pending"]
                    state["pending"] = None
                if pending is None:
                    time.sleep(0.01)
                    continue
                try:
                    with lock:
                        q = measured()
                        velocity = measured_velocity()
                        if not stationary(velocity):
                            raise ValueError("Chunk startup requires stationary measured joints")
                        commanded_start = state["last_q"].copy() if state["last_q"] is not None else q.copy()
                    returning = pending["schema"] == RETURN_SCHEMA
                    if returning:
                        return_state, return_episode = start_state, args.start_episode
                        if "task_id" in pending:
                            task_id = pending["task_id"]
                            if type(task_id) is not int or task_id not in task_starts:
                                raise ValueError("Return task_id is not in the station task-start catalog")
                            return_state, return_episode = task_starts[task_id].state, 0
                        plan = build_return_plan(
                            pending, contract, q, velocity, return_state, return_episode
                        )
                    elif pending["schema"] == joint_policy.COMMAND_SCHEMA:
                        plan = joint_policy.build_plan(pending, contract, q, commanded_start=commanded_start)
                    else:
                        if args.ik is None or not args.ik.is_file():
                            raise RuntimeError("Policy execution requires a relay with --ik PATH")
                        if not solvers:
                            solvers = {
                                s: stack.enter_context(MoveItKDL(args.ik, args.config, s)) for s in SIDES
                            }
                        if pending["schema"] == REPLAY_SCHEMA:
                            episode = load_episode(args.dataset, pending.get("episode"), contract)
                            if pending.get("episode_identity") != episode.identity:
                                raise ValueError("Recorded episode identity mismatch")
                            plan = build_episode_plan(
                                episode,
                                contract,
                                solvers,
                                q,
                                commanded_start,
                                pending.get("execution_rate_hz", 9),
                            )
                        else:
                            plan = build_plan(pending, contract, solvers, q, commanded_start=commanded_start)
                    with lock:
                        if state["fault"]:
                            continue
                        # Check again after IK; do not use moved/stale request state.
                        admit_command(pending, contract, measured(), planning_complete=True)
                        if returning and not stationary(measured_velocity()):
                            raise RuntimeError("Robot started moving during return planning")
                        reason = readiness()
                        if reason:
                            raise RuntimeError(reason)
                        state.update(
                            plan=plan,
                            completed_start_identity=None,
                            active_start_identity=pending.get("start_identity"),
                            start=time.monotonic(),
                            last_tick=time.monotonic(),
                            last_q=plan.joints[0].copy(),
                            settled=None,
                            phase="executing",
                        )
                except Exception as exc:
                    with lock:
                        state["fault"] = str(exc)
    finally:
        rclpy.shutdown()
        thread.join(timeout=2)
        node.destroy_node()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/labs_fr3_31"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--ik", type=Path, help="Existing host-built labs_fr3_ik executable")
    parser.add_argument(
        "--hold-current-on-start",
        action="store_true",
        help="Stream a fixed measured pose while followers activate, then hold",
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    args = parser.parse_args(argv)
    if args.publish != args.enable_robot:
        parser.error("Robot publication requires both --publish and --enable-robot")
    if args.hold_current_on_start and not args.publish:
        parser.error("--hold-current-on-start requires both robot publication gates")
    if args.publish and args.dataset is None:
        parser.error("Enabled relay requires --dataset for independent episode-start verification")
    if args.ik is not None and not args.ik.is_file():
        parser.error("--ik must point to the host-built labs_fr3_ik")
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
