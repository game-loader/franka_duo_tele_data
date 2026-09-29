import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np

from franka_duo_tele_data.labs_action_recording import ActionRecording, export_journal, load_recording


def begin(recorder, index, command_id):
    recorder.event({'event': 'observation', 'index': index, 'state': [0.] * 34})
    raw = {'actions': [[float(index)] * 14] * 32, 'request_id': str(index), 'normalized': False}
    recorder.event({'event': 'wire_response', 'response': raw})
    command = {'command_id': command_id, 'targets': [[0.] * 20] * 32}
    recorder.event({'event': 'inference', 'command': command, 'response': raw})
    return raw


def test_archive_keeps_every_raw_chunk_reference_and_execution_state(tmp_path):
    r = ActionRecording(tmp_path, {'chunk_reference': 'request_observation'})
    first = begin(r, 0, 'a')
    r.event({'event': 'published', 'command_id': 'a'})
    r.event({'event': 'tracking_sample', 'status': {'command_id': 'a', 'phase': 'executing',
             'holding_target': [0.] * 14}, 'measured_joints': [0.1] * 14})
    r.event({'event': 'completed', 'command_id': 'a', 'status': {'phase': 'holding'}})
    second = begin(r, 1, 'b')
    r.event({'event': 'published', 'command_id': 'b'})
    archive = load_recording(r.close('KeyboardInterrupt'))
    assert [c['raw_response'] for c in archive['chunks']] == [first, second]
    a, b = archive['chunks']
    assert a['execution'] == 'completed'
    assert a['summary']['sampled_max_abs_joint_error_rad'] == 0.1
    assert b['execution'] == 'published_unconfirmed'
    assert b['summary'] == {'samples': 0, 'completed': False}
    assert a['observation']['state'] == [0.] * 34


def test_rejected_reply_and_crash_journal_are_preserved(tmp_path):
    r = ActionRecording(tmp_path, {})
    r.event({'event': 'observation', 'index': 0, 'state': [1.] * 34})
    raw = {'actions': [[float('nan')] * 14], 'extra': b'raw payload'}
    r.event({'event': 'wire_response', 'response': raw})
    r.event({'event': 'stopped', 'error': 'Nonfinite actions'})
    recovered = load_recording(export_journal(tmp_path))
    chunk = recovered['chunks'][0]
    assert np.isnan(chunk['raw_response']['actions'][0][0])
    assert chunk['raw_response']['extra'] == b'raw payload'
    assert chunk['execution'] == 'not_published'
    assert 'command' not in chunk
    r.close('ValueError')


def test_snapshot_followed_by_tail_completion_and_stale_feedback(tmp_path):
    r = ActionRecording(tmp_path, {})
    begin(r, 0, 'a')
    r.event({'event': 'published', 'command_id': 'a'})
    assert load_recording(r.export('interrupt_snapshot'))['chunks'][0]['execution'] == 'published_unconfirmed'
    cache = SimpleNamespace(condition=threading.Lock(), buffers={
        f'{s}_q': deque([SimpleNamespace(stamp_ns=time.time_ns()-1_000_000_000,
                                      received_ns=time.monotonic_ns(), value=np.zeros(7))])
        for s in ('left', 'right')
    })
    r.status({'command_id': 'a', 'phase': 'holding', 'holding_target': [0.] * 14}, cache)
    r.event({'event': 'completed', 'command_id': 'a', 'status': {'phase': 'holding'}})
    a = load_recording(r.close('KeyboardInterrupt'))['chunks'][0]
    assert a['execution'] == 'completed'
    assert a['tracking_samples'][0]['measured_joints'] is None
    assert a['summary'] == {'samples': 0, 'completed': True}
    # ROS callbacks arriving during teardown cannot write to a closed journal.
    r.event({'event': 'tracking_sample'})


def test_endpoint_errors_and_portable_offline_viewer(tmp_path):
    from pathlib import Path

    from franka_duo_tele_data.labs_action_viewer import write_viewer
    from franka_duo_tele_data.labs_inference import LabsContract

    config = Path(__file__).resolve().parents[1] / 'configs/labs_fr3_31'
    contract = LabsContract(config)
    q = np.array([0, -.4, 0, -1.8, 0, 1.5, 0] * 2)
    state = contract.state({'left': q[:7], 'right': q[7:]}, [0, 0])
    r = ActionRecording(tmp_path, {'execution_rate_hz': 9,
                                 'urdf': {s: (config/f'{s}.urdf').read_text() for s in ('left', 'right')}})
    r.event({'event': 'observation', 'index': 0, 'state': state.tolist()})
    r.event({'event': 'wire_response', 'response': {'actions': [[0.] * 14] * 32}})
    r.event({'event': 'inference', 'response': {}, 'command': {
        'command_id': 'a', 'targets': [state[:20].tolist()] * 32, 'execution_rate_hz': 9}})
    r.event({'event': 'published', 'command_id': 'a'})
    r.event({'event': 'tracking_sample', 'measured_joints': q.tolist(),
             'status': {'command_id': 'a', 'phase': 'holding', 'holding_target': q.tolist()}})
    r.event({'event': 'completed', 'command_id': 'a'})
    bundle = load_recording(r.close('normal'))
    assert bundle['chunks'][0]['summary']['model_endpoint_error']['left']['position_m'] < 1e-6
    path = tmp_path / 'viewer.html'
    write_viewer(bundle, path)
    html = path.read_text()
    assert '__PAYLOAD__' not in html
    assert '没有可用记录' in html
    assert 'https://' not in html  # portable: no CDN or network dependency
