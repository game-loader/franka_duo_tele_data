import asyncio
import copy
import io
import time
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
from PIL import Image

from franka_duo_tele_data import labs_fastwam_eef as eef
from franka_duo_tele_data.labs_client import LabsClient, command_for, model_input_state
from franka_duo_tele_data.labs_inference import ABSOLUTE_INTEGRATION, CAMERAS, COMMAND_SCHEMA, LabsContract


@pytest.fixture
def contract():
    return LabsContract(Path(__file__).resolve().parents[1] / 'configs/labs_fr3_31')


def state(contract):
    return contract.state(dict.fromkeys(('left', 'right'), [0, -.4, 0, -1.8, 0, 1.5, 0]), [1, 0])


def health(contract):
    return {
        'ready': True, 'busy': False, 'model': 'FastWAM-FR3-PolicyRegistry', 'task': contract.task,
        'default_policy': 'some_other_policy',
        'configured_policies': {eef.POLICY: {
            'variant': 'next_state20', 'checkpoint_step': 5000,
            'checkpoint_sha256': 'checkpoint', 'manifest_sha256': 'manifest',
            'contract': {**eef.CONTRACT, 'normalized': False},
        }},
    }


def reply(contract):
    actions = np.tile(state(contract)[:20], (32, 1)).astype(float)
    actions[:, 0] += np.arange(32) * .001
    actions[:, 9] -= np.arange(32) * .002
    actions[:, 3:9] = [0, 2, 0, -3, 1, 0]
    actions[:, 18:] = [1.03, -.02]
    return {
        **eef.CONTRACT, 'normalized': False, 'policy': eef.POLICY, 'variant': 'next_state20',
        'checkpoint_step': 5000, 'checkpoint_sha256': 'checkpoint', 'manifest_sha256': 'manifest',
        'actions': actions.tolist(), 'action': actions[0].tolist(), 'request_id': 'r',
    }


@pytest.mark.parametrize('change', [
    {'state_format': 'rot6d_rows20'}, {'action_format': 'delta14'}, {'horizon': 16},
    {'state_dim': 34}, {'action_dim': 16}, {'normalized': 0}, {'action_rate_hz': 9},
    {'state_layout': []}, {'action_layout': []}, {'action_representation': 'wrong'},
])
def test_reject_incompatible_health_and_response(contract, change):
    h = health(contract)
    h['configured_policies'][eef.POLICY]['contract'].update(change)
    with pytest.raises(ValueError):
        eef.validate_health(h, contract.task)
    with pytest.raises(ValueError):
        eef.adapt_response({**reply(contract), **change}, health(contract))


@pytest.mark.parametrize('change', [
    {'policy': 'joint16'}, {'variant': 'joint16'}, {'checkpoint_step': 10000},
    {'checkpoint_sha256': 'other'}, {'manifest_sha256': 'other'}, {'action_normalized': True},
    {'actions': [[0] * 20] * 16}, {'actions': [[0] * 14] * 32},
    {'actions': [[float('nan')] * 20] * 32}, {'action': [0] * 20},
])
def test_reject_wrong_model_or_actions(contract, change):
    with pytest.raises(ValueError):
        eef.adapt_response({**reply(contract), **change}, health(contract))


@pytest.mark.parametrize('columns', [[0] * 6, [1, 0, 0, 2, 0, 0]])
def test_reject_degenerate_rotations(contract, columns):
    r = reply(contract)
    r['actions'][4][12:18] = columns
    with pytest.raises(ValueError):
        eef.adapt_response(r, health(contract))


def test_absolute_columns_xyz_no_accumulation_and_raw_preserved(contract):
    h, r = health(contract), reply(contract)
    eef.validate_health(h, contract.task)
    original = copy.deepcopy(r)
    adapted = eef.adapt_response(r, h)
    xyz = [0, 1, 2, 9, 10, 11]
    for reference in (state(contract), state(contract) + .1):
        cmd = command_for(adapted, SimpleNamespace(state=reference, stamp_ns=time.time_ns()),
                          contract, server_profile='fastwam_eef20')
        targets = np.asarray(cmd['targets'])
        assert targets.shape == (32, 20)
        assert cmd['schema'] == COMMAND_SCHEMA
        assert cmd['chunk_integration'] == ABSOLUTE_INTEGRATION
        np.testing.assert_array_equal(targets[:, xyz], np.asarray(r['actions'])[:, xyz])
        np.testing.assert_array_equal(targets[:, 3:9], np.tile([0, 1, 0, -1, 0, 0], (32, 1)))
        np.testing.assert_array_equal(targets[:, 18:], np.tile([1, 0], (32, 1)))
    assert r == original
    assert adapted['action'] == adapted['actions'][0]
    assert adapted['is_recorded_command_action'] is False


def test_binary_websocket_request_fields_and_state20(contract):
    from aiohttp import web

    async def scenario():
        requests = []

        async def metadata(request):
            return web.json_response(health(contract))

        async def serve(request):
            ws = web.WebSocketResponse(protocols=('fastwam.msgpack.v1',))
            await ws.prepare(request)
            async for message in ws:
                payload = msgpack.unpackb(message.data, raw=False)
                requests.append(payload)
                result = {**reply(contract), 'request_id': payload['request_id']}
                await ws.send_bytes(msgpack.packb(result, use_bin_type=True))
            return ws

        app = web.Application()
        app.router.add_get('/health', metadata)
        app.router.add_get('/infer', serve)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        try:
            image = io.BytesIO()
            Image.new('RGB', (640, 480)).save(image, format='PNG')
            async with LabsClient(f'ws://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/infer',
                                  contract, server_profile='fastwam_eef20') as client:
                raw = []
                client.response_sink = raw.append
                result = await client.infer(state(contract), dict.fromkeys(CAMERAS, image.getvalue()))
                assert len(result['actions']) == 32
                assert raw[0]['action_representation'] == eef.REPRESENTATION
                assert raw[0]['actions'][0][18] == 1.03
            assert {k: requests[0][k] for k in eef.REQUEST_FIELDS} == eef.REQUEST_FIELDS
            assert requests[0]['state'] == model_input_state(state(contract), 'fastwam_eef20')
            assert requests[0]['state'] == state(contract)[:20].tolist()
        finally:
            await runner.cleanup()

    asyncio.run(scenario())
