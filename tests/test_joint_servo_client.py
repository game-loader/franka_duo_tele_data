from __future__ import annotations

import json

import numpy as np
import pytest

from franka_duo_tele_data.joint_servo_client import ChunkPacer, ServoStatus, chunk_payload, parse_status


def status(**overrides) -> ServoStatus:
    base = {
        "started": True,
        "fault": False,
        "fault_reason": "",
        "step": 0.0,
        "holding": False,
        "last_step": 0,
        "playback_speed": 0.1,
        "tracking_error_rad": 0.0,
        "chunks": 1,
    }
    base.update(overrides)
    return ServoStatus(**base)


def test_chunk_payload_carries_absolute_step_and_layout():
    actions = np.zeros((5, 20), dtype=np.float32)
    actions[:, 0] = np.arange(5)
    data, dims, offset = chunk_payload(actions, 17)
    assert dims == [5, 20] and offset == 17 and len(data) == 100
    assert data[20] == 1.0
    with pytest.raises(ValueError):
        chunk_payload(np.zeros((0, 20)), 0)
    with pytest.raises(ValueError):
        chunk_payload(np.zeros((2, 19)), 0)
    with pytest.raises(ValueError):
        chunk_payload(np.full((2, 20), np.nan), 0)
    with pytest.raises(ValueError):
        chunk_payload(actions, -1)


def test_parse_status_requires_schema():
    text = json.dumps(
        {
            "schema": "franka_duo_joint_servo_status_v1",
            "started": True,
            "fault": False,
            "fault_reason": "",
            "step": 3.5,
            "holding": False,
            "last_step": 31,
            "playback_speed": 0.1,
            "tracking_error_rad": 0.01,
            "chunks": 1,
            "servo_overruns": 0,
        }
    )
    parsed = parse_status(text)
    assert parsed.step == 3.5 and parsed.last_step == 31 and not parsed.fault
    with pytest.raises(ValueError):
        parse_status(json.dumps({"schema": "other"}))


def test_pacer_requests_first_chunk_once_then_overlapping_futures():
    pacer = ChunkPacer(horizon=32, total_steps=60)
    assert pacer.next_start(None) == 0
    pacer.record(0)
    assert pacer.next_start(None) is None
    assert pacer.next_start(status(started=False)) is None
    # Plenty of plan remaining: no request.
    assert pacer.next_start(status(step=2.0, last_step=31)) is None
    # Half consumed: request from the step after the current one.
    assert pacer.next_start(status(step=15.3, last_step=31)) == 17
    pacer.record(17)
    # Monotonic even if the servo step has not advanced.
    assert pacer.next_start(status(step=15.3, last_step=17)) == 18
    # Never beyond the total.
    assert pacer.next_start(status(step=58.6, last_step=59)) is None
    # Once a chunk covers the end of the recording, stop requesting.
    pacer.record(30)
    assert pacer.next_start(status(step=45.0, last_step=59)) is None
    with pytest.raises(ValueError):
        pacer.record(17)


def test_pacer_fault_and_finish():
    pacer = ChunkPacer(horizon=8, total_steps=10)
    pacer.record(0)
    assert pacer.next_start(status(fault=True, step=1.0, last_step=2)) is None
    assert not pacer.finished(status(step=9.0, last_step=9, holding=False))
    assert pacer.finished(status(step=9.0, last_step=9, holding=True))
    assert not pacer.finished(status(step=9.0, last_step=7, holding=True))
