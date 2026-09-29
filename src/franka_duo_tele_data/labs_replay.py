"""Replay one recorded LeRobot Labs episode through the continuous site relay.

Defaults to offline IK/trajectory validation. Publication requires both gates.
The recorded action is decoded against its OWN row's state, not live state.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
import uuid
from pathlib import Path

import numpy as np

from .labs_client import require_relay, start_command
from .labs_episode import REPLAY_SCHEMA, load_episode
from .labs_ik import MoveItKDL
from .labs_inference import COMMAND_TOPIC, SIDES, STATUS_TOPIC, TOPICS, LabsContract
from .labs_kinematics import joint_positions
from .labs_relay import build_episode_plan
from .ros_utils import _stamp_ns


def run(args):
    if args.publish != args.enable_robot:
        raise ValueError("Robot publication requires both --publish and --enable-robot")
    contract = LabsContract(args.config, args.dataset)
    episode = load_episode(args.dataset, args.episode, contract)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "trace.jsonl").open("x") as log:

        def record(value):
            log.write(json.dumps(value, allow_nan=False) + "\n")
            log.flush()

        record(
            {
                "event": "episode",
                "dataset": str(args.dataset),
                "episode": args.episode,
                "identity": episode.identity,
                "rows": len(episode.targets),
                "source_rate_hz": 30,
                "execution_rate_hz": args.rate,
                "action_reference": "same-row recorded observation.state",
                "model_hashes": contract.model_hashes,
            }
        )
        np.savez(
            args.output / "recorded_actions.npz",
            states=episode.states,
            actions=episode.actions,
            reconstructed_targets=episode.targets,
        )
        with contextlib.ExitStack() as stack:
            solvers = {s: stack.enter_context(MoveItKDL(args.ik, args.config, s)) for s in SIDES}
            plan = build_episode_plan(
                episode, contract, solvers, episode.states[0, 20:], episode.states[0, 20:], args.rate
            )
        record({"event": "preflight", **plan.diagnostics})
        print(
            json.dumps({"preflight": "passed", "rows": len(episode.targets), **plan.diagnostics}), flush=True
        )
        if not args.publish:
            return 0

        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        rclpy.init()
        node = rclpy.create_node("labs_episode_replay")
        status, feedback = {}, {}

        def on_status(message):
            status.update(value=json.loads(message.data), received=time.monotonic())

        node.create_subscription(String, STATUS_TOPIC, on_status, 10)
        for side in SIDES:

            def on_joint(message, s=side):
                feedback[s] = (
                    joint_positions(message.name, message.position, s),
                    _stamp_ns(message),
                    time.monotonic(),
                )

            node.create_subscription(JointState, TOPICS[f"{side}_q"], on_joint, qos_profile_sensor_data)
        publisher = node.create_publisher(String, COMMAND_TOPIC, 1)

        def live_joints():
            if len(feedback) != 2 or any(
                v[1] is None
                or not 0 <= time.time_ns() - v[1] <= 150_000_000
                or time.monotonic() - v[2] > 0.15
                for v in feedback.values()
            ):
                raise RuntimeError("Fresh measured joints required")
            return np.concatenate([feedback[s][0] for s in SIDES])

        def wait(command_id=None, timeout=120):
            deadline = time.monotonic() + timeout
            progress = 0.0
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.02)
                if not status:
                    continue
                if time.monotonic() - status["received"] > 0.5:
                    raise RuntimeError("Replay relay status stale")
                if node.count_publishers(COMMAND_TOPIC) != 1 or publisher.get_subscription_count() != 1:
                    raise RuntimeError("Replay requires exclusive client and one site relay")
                value = status["value"]
                if require_relay(value, contract, command_id=command_id):
                    return value
                if command_id and time.monotonic() - progress > 5:
                    record({"event": "progress", "status": value})
                    print(
                        json.dumps(
                            {
                                "phase": value.get("phase"),
                                "rows": value.get("rows"),
                                "duration_s": value.get("duration_s"),
                            }
                        ),
                        flush=True,
                    )
                    progress = time.monotonic()
            raise TimeoutError(f"Replay did not finish: {status}")

        try:
            ready = wait(timeout=10)
            reset = start_command(episode.states[0], contract, args.episode, speed=args.rate / 30)
            if reset["schema"] not in ready.get("supported_command_schemas", []):
                raise RuntimeError("Reload the Labs relay to support restore with both grippers open")
            publisher.publish(String(data=json.dumps(reset)))
            record({"event": "published_restore", "command": reset})
            complete = wait(reset["command_id"])
            if complete.get("completed_start_identity") != reset["start_identity"]:
                raise RuntimeError("Relay did not confirm the selected episode start")
            record({"event": "restored", "status": complete})
            command = {
                "schema": REPLAY_SCHEMA,
                "command_id": uuid.uuid4().hex,
                "model_hashes": contract.model_hashes,
                "created_ns": time.time_ns(),
                "episode": episode.index,
                "episode_identity": episode.identity,
                "reference_joints": live_joints().tolist(),
                "execution_rate_hz": args.rate,
            }
            publisher.publish(String(data=json.dumps(command)))
            record({"event": "published_replay", "command": command})
            complete = wait(command["command_id"])
            record({"event": "completed", "status": complete, "measured_joints": live_joints().tolist()})
            print(
                json.dumps(
                    {
                        "replay": "completed",
                        "episode": episode.index,
                        "rows": len(episode.targets),
                        "status": complete,
                    }
                ),
                flush=True,
            )
        except BaseException as exc:
            record({"event": "stopped", "error": str(exc), "status": status})
            raise
        finally:
            node.destroy_node()
            rclpy.shutdown()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/labs_fr3_31"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--rate", type=float, default=9)
    parser.add_argument("--ik", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(f"outputs/labs_replay_{time.time_ns()}"))
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    args = parser.parse_args(argv)
    if args.publish != args.enable_robot:
        parser.error("Robot publication requires both --publish and --enable-robot")
    if args.episode < 0 or not np.isfinite(args.rate) or not 0 < args.rate <= 30:
        parser.error("Require episode >=0 and 0 < rate <=30")
    if not args.ik.is_file():
        parser.error("--ik must be the host-built Labs solver")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
