"""Model-independent physical action contracts, selected explicitly by the user."""

import numpy as np

from . import labs_joint_inference as joint
from .labs_inference import ABSOLUTE_REPRESENTATION, absolute_targets
from .labs_mcap_to_lerobot import STATE_NAMES

PROFILES = ('absolute20', 'absolute_joint16')


def adapt_response(result, profile):
    if profile not in PROFILES:
        raise ValueError(f'Unknown physical action profile: {profile}')
    # The selected profile declares raw physical units. If the server includes
    # normalization/format declarations, never ignore an explicit contradiction.
    for key in ('normalized', 'action_normalized'):
        if key in result and result[key] is not False:
            raise ValueError(f'Physical action contract rejects {key}={result[key]!r}')
    dimension = 20 if profile == 'absolute20' else 16
    names = STATE_NAMES[:20] if dimension == 20 else joint.JOINT_NAMES
    for key in ('state_dim', 'action_dim'):
        if key in result and result[key] != dimension:
            raise ValueError(f'{key} must equal {dimension}')
    for key in ('state_layout', 'action_layout', 'state_names', 'action_names'):
        if key in result and result[key] != names:
            raise ValueError(f'{key} disagrees with selected physical layout')
    if 'absolute_action' in result and result['absolute_action'] is not True:
        raise ValueError('Expected absolute actions')
    for key in ('action_contract', 'action_representation'):
        if 'delta' in str(result.get(key, '')).lower():
            raise ValueError('Absolute action mode cannot execute declared delta actions')
    if dimension == 20:
        if 'translation_units' in result and result['translation_units'] not in ('m', 'metres', 'meters'):
            raise ValueError('Absolute translations must be in metres')
        for key, value in (('state_format', 'rot6d_cols20'), ('action_format', 'absolute20')):
            if key in result and result[key] != value:
                raise ValueError(f'{key} must be {value}')
    else:
        if 'joint_units' in result and result['joint_units'] not in ('rad', 'radians'):
            raise ValueError('Absolute joints must be in radians')
        if 'action_format' in result and result['action_format'] not in ('joint16', 'absolute16', 'absolute_joint16'):
            raise ValueError('Expected absolute joint16 action_format')
    actions = np.asarray(result.get('actions'), dtype=float)
    if actions.ndim != 2 or actions.shape[1] != dimension or not 1 <= len(actions) <= 256 or not np.isfinite(actions).all():
        raise ValueError(f'Expected finite actions[H,{dimension}], 1 <= H <= 256')
    if 'chunk_size' in result and result['chunk_size'] != len(actions):
        raise ValueError('chunk_size disagrees with returned actions')
    if 'action' in result and not np.array_equal(np.asarray(result['action']), actions[0]):
        raise ValueError('action must equal actions[0]')
    if dimension == 20:
        targets = absolute_targets({**result, 'normalized': False, 'action_representation': ABSOLUTE_REPRESENTATION},
                                   rows=len(actions))
        representation = ABSOLUTE_REPRESENTATION
    else:
        targets = actions.copy()
        targets[:, joint.GRIPPER_INDICES] = (actions[:, joint.GRIPPER_INDICES] >= .5).astype(float)
        representation = joint.SCHEMA
    adapted = {**result, 'actions': targets.tolist(), 'normalized': False,
               'action_representation': representation, 'client_action_contract': profile,
               'client_gripper_postprocess': 'binary_open_ge_0.5_v1'}
    if dimension == 20:
        adapted['client_rotation_postprocess'] = 'gram_schmidt_columns_v1'
    if 'action' in result:
        adapted['action'] = targets[0].tolist()
    return adapted
