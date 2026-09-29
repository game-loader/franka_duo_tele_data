"""Start/recover the station-owned drivers, followers and Labs relay at current pose."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from .labs_client import launch_relay
from .labs_inference import COMMAND_TOPIC, SIDES, STATUS_TOPIC, TOPICS
from .labs_kinematics import joint_positions
from .labs_task_starts import load_task_starts
from .ros_utils import _stamp_ns

BROADCASTERS = {'joint_state_broadcaster', 'franka_robot_state_broadcaster'}
FOLLOWER = 'joint_follower_controller'


def container_running(name):
    result = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', name],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip() == 'true'


def relay_pids():
    """Only this checkout's current-user Python relay, never arbitrary cached PIDs."""
    found = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            argv = (path / 'cmdline').read_bytes().split(b'\0')
            if (path.stat().st_uid == os.getuid() and (path / 'cwd').resolve() == Path.cwd()
                    and argv[1:3] == [b'-m', b'franka_duo_tele_data.labs_relay']):
                found.append(int(path.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return found


def hardware_recovery():
    # Use the site's installed message types inside its existing driver container.
    # No dependency installation, driver replacement or robot target publication.
    code = '''
import json,time
import rclpy
from rclpy.action import ActionClient
from franka_msgs.action import ErrorRecovery
rclpy.init();n=rclpy.create_node('labs_hardware_recovery')
def wait(f):
    end=time.monotonic()+20
    while not f.done() and time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.05)
    if not f.done():raise TimeoutError('Franka hardware recovery timeout')
    return f.result()
try:
    for side in ('left','right'):
        c=ActionClient(n,ErrorRecovery,'/'+side+'/action_server/error_recovery')
        if not c.wait_for_server(timeout_sec=5):raise RuntimeError(side+': recovery action missing')
        goal=wait(c.send_goal_async(ErrorRecovery.Goal()))
        if not goal.accepted:raise RuntimeError(side+': recovery rejected')
        result=wait(goal.get_result_async())
        if result.status!=4:raise RuntimeError(side+': recovery failed, status='+str(result.status))
        print(json.dumps({'hardware_recovered':side}),flush=True)
finally:n.destroy_node();rclpy.shutdown()
'''
    subprocess.run(['docker', 'exec', '-i', 'franka-robot', 'bash', '-lc',
                    'source /opt/ros/humble/setup.bash && source /workspace/src/install/setup.bash && python3 -'],
                   input=code, text=True, check=True, timeout=55)


class Control:
    def __init__(self):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        self.ros = rclpy
        self.node = rclpy.create_node('labs_control_service')
        self.feedback = {}
        self.status = {}
        self.status_time = 0
        self.node.create_subscription(String, STATUS_TOPIC, self.on_status, 10)
        for side in SIDES:
            def receive(message, s=side):
                self.feedback[s] = (
                    joint_positions(message.name, message.position, s),
                    joint_positions(message.name, message.velocity, s), _stamp_ns(message),
                )
            self.node.create_subscription(JointState, TOPICS[f'{side}_q'], receive, qos_profile_sensor_data)

    def on_status(self, message):
        self.status = json.loads(message.data)
        self.status_time = time.monotonic()

    def spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.ros.spin_once(self.node, timeout_sec=.05)

    def call(self, kind, name, request):
        client = self.node.create_client(kind, name)
        try:
            if not client.wait_for_service(timeout_sec=5):
                raise RuntimeError(f'Driver service unavailable: {name}')
            future = client.call_async(request)
            self.ros.spin_until_future_complete(self.node, future, timeout_sec=10)
            if not future.done():
                raise TimeoutError(name)
            return future.result()
        finally:
            self.node.destroy_client(client)

    def controllers(self, side):
        from controller_manager_msgs.srv import ListControllers
        result = self.call(ListControllers, f'/{side}/controller_manager/list_controllers', ListControllers.Request())
        return {c.name: c.state for c in result.controller}

    def switch(self, side, activate=(), deactivate=()):
        from controller_manager_msgs.srv import SwitchController
        request = SwitchController.Request()
        request.activate_controllers = list(activate)
        request.deactivate_controllers = list(deactivate)
        request.strictness = 2
        request.timeout.sec = 5
        if not self.call(SwitchController, f'/{side}/controller_manager/switch_controller', request).ok:
            raise RuntimeError(f'{side}: controller switch failed')

    def measured(self):
        if len(self.feedback) != 2 or any(not 0 <= time.time_ns() - v[2] <= 150_000_000
                                          for v in self.feedback.values()):
            raise RuntimeError('Missing/stale arm feedback')
        q = np.concatenate([self.feedback[s][0] for s in SIDES])
        dq = np.concatenate([self.feedback[s][1] for s in SIDES])
        if not np.isfinite(q).all() or not np.isfinite(dq).all() or np.max(np.abs(dq)) > .02:
            raise RuntimeError('Arms are not stationary; cannot establish current-pose hold')
        return q

    def output_owners(self):
        return {topic: [p.node_namespace + '/' + p.node_name
                        for p in self.node.get_publishers_info_by_topic(topic)]
                for s in SIDES for topic in (
                    f'/{s}/follower/gello/joint_states',
                    f'/{s}/follower/gripper/gripper_client/target_gripper_width_percent')}

    def report(self):
        report = {'controllers': {}, 'relay': self.status if time.monotonic() - self.status_time < .5 else None,
                  'relay_pids': relay_pids(), 'output_publishers': self.output_owners(), 'feedback': {}}
        for side in SIDES:
            try:
                report['controllers'][side] = self.controllers(side)
            except Exception as exc:
                report['controllers'][side] = {'error': str(exc)}
            if side in self.feedback:
                report['feedback'][side] = {'age_s': (time.time_ns() - self.feedback[side][2]) / 1e9,
                                            'max_velocity_rad_s': float(np.max(np.abs(self.feedback[side][1])))}
        return report

    def start(self, args):
        from rcl_interfaces.srv import GetParameters

        self.spin(1)
        if self.node.count_publishers(COMMAND_TOPIC):
            raise RuntimeError('Stop inference/replay clients before starting or recovering control')
        if not (args.recover or args.restart_relay) and self.status.get('ready') and self.status.get('phase') == 'holding' and time.monotonic() - self.status_time < .5:
            return {'result': 'already ready', **self.report()}
        # Validate all task data before interrupting a healthy current-pose hold.
        load_task_starts(args.config)
        # Stop the site coordinator before switching controllers so it cannot race recovery.
        if container_running('controller-coordinator'):
            subprocess.run(['docker', 'stop', 'controller-coordinator'], check=True, timeout=30)
        for container in ('franka-robot', 'robotiq-gripper'):
            if not container_running(container):
                subprocess.run(['docker', 'start', container], check=True, timeout=30)
        self.spin(3)
        for side in SIDES:
            controllers = self.controllers(side)
            unexpected = {k for k, v in controllers.items() if v == 'active'} - BROADCASTERS - {FOLLOWER, 'gravity_compensation_controller'}
            if unexpected:
                raise RuntimeError(f'{side}: unexpected active controllers {unexpected}')
            active = [k for k in (FOLLOWER, 'gravity_compensation_controller') if controllers.get(k) == 'active']
            if active:
                self.switch(side, deactivate=active)
            if {k for k, v in self.controllers(side).items() if v == 'active'} != BROADCASTERS:
                raise RuntimeError(f'{side}: could not establish broadcasters-only state')
        # Both followers are confirmed inactive before stopping any relay.
        for pid in relay_pids():
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            self.spin(.2)
            if not relay_pids() and not any(self.output_owners().values()):
                break
        else:
            raise RuntimeError(f'Relay/output publishers remain; followers left inactive: {self.output_owners()}')
        if args.recover:
            hardware_recovery()
        self.spin(1)
        self.measured()
        for side in SIDES:
            request = GetParameters.Request()
            request.names = ['sync_after_activation', 'target_joint_states_topic_name']
            params = self.call(GetParameters, f'/{side}/{FOLLOWER}/get_parameters', request).values
            topic = params[1].string_value
            topic = topic if topic.startswith('/') else f'/{side}/{topic}'
            if not params[0].bool_value or topic != f'/{side}/follower/gello/joint_states':
                raise RuntimeError(f'{side}: invalid follower sync/target-topic configuration')
        self.status = {}
        self.status_time = 0
        process = launch_relay(args)
        activated = []
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                self.spin(.1)
                if self.status.get('fault') or process.poll() is not None:
                    raise RuntimeError(f'Relay startup failed: {self.status}; see outputs/labs_relay/relay.log')
                if self.status.get('phase') == 'arming' and self.status.get('holding_target') is not None:
                    break
            else:
                raise TimeoutError('Relay did not establish current-pose hold')
            if np.max(np.abs(self.measured() - self.status['holding_target'])) > .01:
                raise RuntimeError('Arm moved before follower activation')
            for side in SIDES:
                activated.append(side)
                self.switch(side, activate=[FOLLOWER])
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                self.spin(.1)
                if self.status.get('fault'):
                    raise RuntimeError(self.status['fault'])
                if (time.monotonic() - self.status_time < .5 and self.status.get('ready')
                        and self.status.get('phase') == 'holding'
                        and all(self.status.get('follower_states', {}).get(s) == 'FOLLOWING' for s in SIDES)):
                    error = float(np.max(np.abs(self.measured() - self.status['holding_target'])))
                    if error > .01:
                        raise RuntimeError('Follower activation hold drift')
                    return {'result': 'ready', 'max_hold_error_rad': error, **self.report()}
            raise TimeoutError('Followers did not reach FOLLOWING/holding')
        except BaseException:
            for side in reversed(activated):
                try:
                    self.switch(side, deactivate=[FOLLOWER])
                except Exception as exc:
                    print(json.dumps({'rollback_error': side, 'error': str(exc)}), flush=True)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--status', action='store_true', help='Read-only status (default)')
    mode.add_argument('--start', action='store_true', help='Start drivers/followers/relay holding the current pose')
    mode.add_argument('--recover', action='store_true', help='Clear hardware errors and rebuild current-pose hold')
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--enable-robot', action='store_true')
    parser.add_argument('--restart-relay', action='store_true',
                        help='With --start, reload relay code/task starts and re-establish current-pose hold')
    parser.add_argument('--config', type=Path, default=Path('configs/labs_fr3_31'))
    parser.add_argument('--dataset', type=Path, default=Path('/home/agile/work/labs/data/lerobot/labs_fr3_link8_delta14_20260916'))
    parser.add_argument('--start-episode', type=int, default=0)
    parser.add_argument('--ik', type=Path, default=Path('site/install/labs_fr3_kinematics/lib/labs_fr3_kinematics/labs_fr3_ik'))
    args = parser.parse_args(argv)
    if args.restart_relay and not args.start:
        parser.error('--restart-relay requires --start')
    if (args.start or args.recover) and not (args.publish and args.enable_robot):
        parser.error('--start/--recover requires BOTH --publish --enable-robot')
    if args.publish != args.enable_robot:
        parser.error('Robot publication requires both gates')
    import rclpy

    directory = Path('outputs/labs_control_service')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rclpy.init()
        control = Control()
        try:
            control.spin(2)
            result = control.start(args) if args.start or args.recover else control.report()
            (directory / 'latest.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
        finally:
            control.node.destroy_node()
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
