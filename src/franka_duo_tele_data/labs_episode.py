"""Recorded Labs actions reconstructed against their own same-row states."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labs_action_delta import delta14_to_absolute20
from .labs_inference import validate_target

REPLAY_SCHEMA = "labs_fr3_episode_action_replay_v1"


@dataclass
class Episode:
    index: int
    states: np.ndarray
    actions: np.ndarray
    targets: np.ndarray
    identity: str


def load_episode(dataset, episode, contract):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not isinstance(episode, int) or episode < 0:
        raise ValueError("Episode index must be nonnegative")
    tables = [
        pq.read_table(
            p,
            columns=["frame_index", "timestamp", "observation.state", "action"],
            filters=[("episode_index", "=", episode)],
        )
        for p in sorted((Path(dataset) / "data").rglob("*.parquet"))
    ]
    if not tables:
        raise ValueError("Dataset has no episode data")
    table = pa.concat_tables(tables).sort_by([("frame_index", "ascending")])
    frames = np.asarray(table["frame_index"])
    if not len(frames) or not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("Episode requires unique consecutive frame indices starting at zero")
    times = np.asarray(table["timestamp"], dtype=float)
    if not np.allclose(times, np.arange(len(frames)) / 30, atol=1e-4, rtol=0):
        raise ValueError("Episode timestamp grid must match 30 Hz source")
    states = np.stack([contract.validate_state(row) for row in table["observation.state"].to_pylist()])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if actions.shape != (len(frames), 14) or not np.isfinite(actions).all():
        raise ValueError("Episode actions must be finite [N,14]")
    if not np.isin(actions[:, 12:], [0, 1]).all():
        raise ValueError("Recorded gripper actions must already be binary")
    targets = delta14_to_absolute20(actions, states)
    for target in targets:
        validate_target(target)
    digest = hashlib.sha256(
        json.dumps({"episode": episode, "models": contract.model_hashes}, sort_keys=True).encode()
    )
    digest.update(states.astype("<f4").tobytes())
    digest.update(actions.astype("<f4").tobytes())
    return Episode(episode, states, actions, targets, digest.hexdigest())
