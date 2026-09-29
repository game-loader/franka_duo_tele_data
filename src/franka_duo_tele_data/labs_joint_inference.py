"""Absolute joint16 policy contract; no delta accumulation or inverse kinematics."""

from __future__ import annotations

import numpy as np

from .labs_joint_dataset import JOINT_NAMES, SCHEMA

COMMAND_SCHEMA = "labs_fr3_absolute_joint16_command_v1"
INTEGRATION = "absolute_joint16_v1"
JOINT_INDICES = [*range(7), *range(8, 15)]
GRIPPER_INDICES = [7, 15]
FASTWAM_REPRESENTATION = "franka_fr3_duo_joint16_absolute_v1"
NEXT_POLICY = "joint16_next_action_v2_5k"
NEXT_CONTRACT = {
    "state_dim": 16, "action_dim": 16, "horizon": 32, "action_rate_hz": 30,
    "state_layout": JOINT_NAMES, "action_layout": JOINT_NAMES,
    "action_representation": FASTWAM_REPRESENTATION, "joint_units": "radians",
}


def check_next_contract(value):
    mismatch = [key for key, expected in NEXT_CONTRACT.items() if value.get(key) != expected]
    if value.get("normalized") is not False:
        mismatch.append("normalized")
    if mismatch:
        raise ValueError(f"FastWAM next joint16 contract mismatch: {', '.join(mismatch)}")


def validate_next_health(health, task):
    expected = {"ready": True, "busy": False, "model": "FastWAM-FR3-PolicyRegistry", "task": task}
    mismatch = [f"{key}: expected={value!r}, received={health.get(key)!r}"
                for key, value in expected.items()
                if (health.get(key) is not value if key in ("ready", "busy") else health.get(key) != value)]
    if mismatch:
        raise ValueError("FastWAM joint16 health mismatch: " + "; ".join(mismatch))
    policy = health.get("configured_policies", {}).get(NEXT_POLICY, {})
    if policy.get("variant") != "joint16_next_action_v2" or policy.get("checkpoint_step") != 5000:
        raise ValueError(f"FastWAM registry must contain {NEXT_POLICY} checkpoint5000")
    check_next_contract(policy.get("contract", {}))


def adapt_next_response(result, health):
    check_next_contract(result)
    if result.get("policy") != NEXT_POLICY or result.get("variant") != "joint16_next_action_v2":
        raise ValueError("FastWAM next joint16 response has wrong policy/variant")
    policy = health["configured_policies"][NEXT_POLICY]
    for key in ("checkpoint_step", "checkpoint_sha256", "manifest_sha256"):
        if key not in policy or result.get(key) != policy[key]:
            raise ValueError(f"FastWAM next joint16 response {key} disagrees with health")
    adapted = adapt_fastwam_response(result, policy["contract"])
    # These labels predict measured next state, not recorded follower commands.
    adapted["is_recorded_command_action"] = False
    return adapted


def validate_fastwam_health(health, task):
    expected = {
        "model": "FastWAM-FR3-Joint16", "variant": "joint16",
        "state_dim": 16, "action_dim": 16, "horizon": 32, "action_rate_hz": 30,
        "state_layout": JOINT_NAMES, "action_layout": JOINT_NAMES,
        "action_representation": FASTWAM_REPRESENTATION, "joint_units": "radians",
        "task": task,
    }
    mismatch = [k for k, v in expected.items() if health.get(k) != v]
    if health.get("ready") is not True or health.get("busy") is not False:
        mismatch.append("readiness")
    if health.get("normalized") is not False:
        mismatch.append("normalization")
    if mismatch:
        raise ValueError(f"FastWAM joint16 health mismatch: {', '.join(mismatch)}")


def adapt_fastwam_response(result, health):
    """Map the checked FastWAM wire contract onto the shared joint executor."""
    if not isinstance(result, dict) or result.get("normalized") is not False:
        raise ValueError("FastWAM joint16 requires normalized=false")
    if result.get("action_representation") != FASTWAM_REPRESENTATION:
        raise ValueError("FastWAM response must declare absolute joint16")
    if "action_normalized" in result and result["action_normalized"] is not False:
        raise ValueError("Conflicting FastWAM normalization declarations")
    if "action_contract" in result and result["action_contract"] != FASTWAM_REPRESENTATION:
        raise ValueError("Conflicting FastWAM action contract")
    for key, expected in (("horizon", 32), ("action_dim", 16), ("action_rate_hz", 30)):
        if key in result and result[key] != expected:
            raise ValueError(f"FastWAM joint16 response {key} mismatch")
    return validate_response(
        {**result, "action_representation": SCHEMA, "action_contract": SCHEMA,
         "action_normalized": False},
        {"chunk_size": health["horizon"], "prediction_horizon": health["horizon"]},
    )


def model_state(state34):
    value = np.asarray(state34, dtype=np.float32)
    if value.shape != (34,) or not np.isfinite(value).all():
        raise ValueError("Joint16 projection requires finite internal state34")
    return np.r_[value[20:27], value[18], value[27:34], value[19]].tolist()


def validate_info(info, protocol, task):
    expected = {
        "protocol": protocol,
        "state_dim": 16,
        "action_dim": 16,
        "chunk_size": 32,
        "state_names": JOINT_NAMES,
        "action_names": JOINT_NAMES,
        "action_contract": SCHEMA,
        "state_input_normalized": False,
        "action_normalized": False,
        "image_shape_hwc": [480, 640, 3],
        "cameras": {k: f"observation.images.{k}" for k in ("head", "wrist_left", "wrist_right")},
        "default_task": task,
    }
    mismatch = [k for k, v in expected.items() if info.get(k) != v]
    if info.get("state_input_normalized") is not False or info.get("action_normalized") is not False:
        mismatch.append("normalization")
    if mismatch:
        raise ValueError(f"Joint16 info mismatch: {', '.join(mismatch)}")


def validate_response(result, info):
    if not isinstance(result, dict) or result.get("action_normalized") is not False:
        raise ValueError("Joint16 requires action_normalized=false")
    if "normalized" in result and result["normalized"] is not False:
        raise ValueError("Conflicting Joint16 normalization declarations")
    for key in ("action_contract", "action_representation"):
        if key in result and result[key] != SCHEMA:
            raise ValueError("Joint16 response must contain absolute joints")
    actions = np.asarray(result.get("actions"), dtype=float)
    if actions.shape != (32, 16) or not np.isfinite(actions).all():
        raise ValueError("Expected finite absolute actions[32,16]")
    for key in ("chunk_size", "prediction_horizon"):
        if key in result and result[key] != info.get(key):
            raise ValueError(f"Joint16 response {key} disagrees with info")
    if "action" in result and not np.array_equal(np.asarray(result["action"]), actions[0]):
        raise ValueError("Joint16 action must equal actions[0]")
    # Binary openness regression, matching the dataset. No joint clipping.
    targets = actions.copy()
    targets[:, GRIPPER_INDICES] = (actions[:, GRIPPER_INDICES] >= 0.5).astype(float)
    adapted = {
        **result,
        "action_representation": SCHEMA,
        "normalized": False,
        "actions": targets.tolist(),
        "client_gripper_postprocess": "binary_open_ge_0.5_v1",
    }
    if "action" in result:
        adapted["action"] = targets[0].tolist()
    return adapted


def split_targets(targets):
    value = np.asarray(targets, dtype=float)
    if value.ndim != 2 or value.shape[1] != 16 or not 1 <= len(value) <= 256 or not np.isfinite(value).all():
        raise ValueError("Expected finite joint16 target rows")
    if not np.isin(value[:, GRIPPER_INDICES], [0, 1]).all():
        raise ValueError("Joint16 target grippers must be binary")
    return value[:, JOINT_INDICES].copy(), value[:, GRIPPER_INDICES].copy()


def build_plan(command, contract, measured, *, commanded_start=None):
    from .labs_tracking import track_chunk

    if command.get("schema") != COMMAND_SCHEMA or command.get("chunk_integration") != INTEGRATION:
        raise ValueError("Expected absolute joint16 command")
    joints, grippers = split_targets(command.get("targets"))
    initial = np.asarray(measured, dtype=float)
    start = initial if commanded_start is None else np.asarray(commanded_start, dtype=float)
    if (
        initial.shape != (14,)
        or start.shape != (14,)
        or not np.isfinite(initial).all()
        or not np.isfinite(start).all()
        or np.max(np.abs(start - initial)) > 0.15
    ):
        raise ValueError("Invalid initial joint hold")
    speed = command.get("speed")
    if (
        isinstance(speed, bool)
        or not isinstance(speed, (int, float))
        or not 0 < speed <= 1
        or command.get("action_rate_hz") != 30
        or command.get("execution_rate_hz") != 30 * speed
    ):
        raise ValueError("Joint16 requires matching 30 Hz source and speed")
    anchors = np.vstack((start, joints))
    if np.max(np.abs(np.diff(anchors, axis=0))) > 0.25:
        raise ValueError("Joint16 adjacent joint target jump exceeds 0.25 rad")
    bounds = np.concatenate([contract.fk[s].bounds for s in ("left", "right")])
    if np.any(initial < bounds[:, 0]) or np.any(initial > bounds[:, 1]):
        raise ValueError("Measured joints outside URDF bounds")
    return track_chunk(
        anchors,
        grippers,
        1 / (30 * speed),
        bounds,
        max_reference_lag=0.3,
        max_velocity=0.8,
        max_acceleration=2.0,
        max_jerk=20.0,
        target_velocity_weight=0.0,
    )
