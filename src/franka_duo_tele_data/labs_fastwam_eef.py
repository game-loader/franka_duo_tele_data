"""FastWAM registry next_state20_5k: absolute link0/link8 pose targets."""

import numpy as np

from .labs_inference import ABSOLUTE_REPRESENTATION, absolute_targets
from .labs_mcap_to_lerobot import STATE_NAMES

POLICY = "next_state20_5k"
REPRESENTATION = "franka_fr3_duo_link8_next_state20_v1"
REQUEST_FIELDS = {"policy": POLICY, "state_format": "rot6d_cols20", "action_format": "absolute20"}
CONTRACT = {
    "state_dim": 20, "action_dim": 20, "horizon": 32, "action_rate_hz": 30,
    "state_format": "rot6d_cols20", "action_format": "absolute20",
    "state_layout": STATE_NAMES[:20], "action_layout": STATE_NAMES[:20],
    "action_representation": REPRESENTATION, "translation_units": "metres",
}


def check_contract(value):
    mismatch = [key for key, expected in CONTRACT.items() if value.get(key) != expected]
    if value.get("normalized") is not False:
        mismatch.append("normalized")
    if mismatch:
        raise ValueError(f"FastWAM EEF20 contract mismatch: {', '.join(mismatch)}")


def validate_health(health, task):
    if (
        health.get("ready") is not True or health.get("busy") is not False
        or health.get("model") != "FastWAM-FR3-PolicyRegistry" or health.get("task") != task
    ):
        raise ValueError("FastWAM EEF20 registry not ready or task/model mismatch")
    policy = health.get("configured_policies", {}).get(POLICY, {})
    if policy.get("variant") != "next_state20" or policy.get("checkpoint_step") != 5000:
        raise ValueError(f"FastWAM EEF20 registry must contain {POLICY} checkpoint5000")
    check_contract(policy.get("contract", {}))
    return policy


def adapt_response(result, health):
    check_contract(result)
    if result.get("policy") != POLICY or result.get("variant") != "next_state20":
        raise ValueError("FastWAM EEF20 response has wrong policy/variant")
    policy = health["configured_policies"][POLICY]
    for key in ("checkpoint_step", "checkpoint_sha256", "manifest_sha256"):
        if key not in policy or result.get(key) != policy[key]:
            raise ValueError(f"FastWAM EEF20 response {key} disagrees with health")
    if "action_normalized" in result and result["action_normalized"] is not False:
        raise ValueError("Conflicting FastWAM EEF20 normalization declarations")
    actions = np.asarray(result.get("actions"), dtype=float)
    targets = absolute_targets(
        {**result, "action_representation": ABSOLUTE_REPRESENTATION}, rows=32
    )
    if "action" in result and not np.array_equal(np.asarray(result["action"]), actions[0]):
        raise ValueError("FastWAM EEF20 action must equal actions[0]")
    adapted = {
        **result, "action_representation": ABSOLUTE_REPRESENTATION,
        "actions": targets.tolist(), "client_gripper_postprocess": "binary_open_ge_0.5_v1",
        "client_rotation_postprocess": "gram_schmidt_columns_v1", "is_recorded_command_action": False,
    }
    if "action" in result:
        adapted["action"] = targets[0].tolist()
    return adapted
