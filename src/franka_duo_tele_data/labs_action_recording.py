"""Durable inference chunks and sampled execution feedback, independent of ROS."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import msgpack
import numpy as np

SCHEMA = "labs_inference_recording_v1"


class ActionRecording:
    """Append as data arrives; atomically export one portable file on exit."""

    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.lock = threading.Lock()
        self.journal = (self.directory / "actions.journal.msgpack").open("xb")
        self.closed = False
        self.event({"event": "configuration", **metadata})

    def event(self, event):
        payload = {"recorded_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(), **event}
        with self.lock:
            if self.closed:
                return
            self.journal.write(msgpack.packb(payload, use_bin_type=True))
            self.journal.flush()
            if event["event"] in ("wire_response", "published", "completed"):
                os.fsync(self.journal.fileno())

    def status(self, status, cache):
        # Status exposes the last published joint goal, so no command-topic
        # subscriber is added (the relay requires exactly one subscriber).
        with cache.condition:
            feedback = {
                side: cache.buffers[f"{side}_q"][-1]
                for side in ("left", "right") if cache.buffers[f"{side}_q"]
            }
        now_ns, mono_ns = time.time_ns(), time.monotonic_ns()
        fresh = len(feedback) == 2 and all(
            0 <= now_ns - sample.stamp_ns <= 150_000_000
            and 0 <= mono_ns - sample.received_ns <= 150_000_000
            for sample in feedback.values()
        )
        self.event({
            "event": "tracking_sample", "status": status,
            "measured_joints": np.concatenate([feedback[s].value for s in ("left", "right")]).tolist()
            if fresh else None,
            "feedback_stamp_ns": {s: v.stamp_ns for s, v in feedback.items()},
            "feedback_received_monotonic_ns": {s: v.received_ns for s, v in feedback.items()},
        })

    def export(self, reason):
        with self.lock:
            self.journal.flush()
            os.fsync(self.journal.fileno())
            return export_journal(self.directory, reason)

    def close(self, reason):
        with self.lock:
            self.closed = True
            self.journal.flush()
            os.fsync(self.journal.fileno())
            self.journal.close()
            return export_journal(self.directory, reason)


def assemble(events, reason):
    chunks, pending, by_command = [], None, {}
    for event in events:
        kind = event["event"]
        if kind == "observation":
            pending = {"index": event["index"], "observation": event, "published": False,
                       "execution": "not_published", "tracking_samples": []}
        elif kind == "wire_response" and pending is not None:
            pending["raw_response"] = event["response"]
            pending["response_received_ns"] = event["recorded_ns"]
            chunks.append(pending)
        elif kind == "inference" and pending is not None:
            # Raw wire replies may be absent in recovered legacy/test traces.
            if "raw_response" not in pending:
                pending["raw_response"] = event["response"]
                chunks.append(pending)
            pending.update(command=event["command"], validated_response=event["response"])
            by_command[event["command"]["command_id"]] = pending
        elif kind in ("published", "completed") and event.get("command_id") in by_command:
            chunk = by_command[event["command_id"]]
            if kind == "published":
                chunk.update(published=True, execution="published_unconfirmed", published_ns=event["recorded_ns"])
            else:
                chunk.update(execution="completed", completion=event)
        elif kind == "tracking_sample":
            status = event["status"]
            chunk = by_command.get(status.get("command_id"))
            if chunk is not None:
                chunk["tracking_samples"].append(event)
                if status.get("fault"):
                    chunk.update(execution="fault", fault=status["fault"])
    for chunk in chunks:
        chunk["summary"] = tracking_summary(chunk)
    metadata = next((e for e in events if e["event"] == "configuration"), {})
    if metadata.get("urdf"):
        add_endpoint_errors(chunks, metadata["urdf"])
    return {
        "schema": SCHEMA, "exit_reason": reason, "exported_ns": time.time_ns(), "chunks": chunks,
        "events": [e for e in events if e["event"] != "tracking_sample"],
        "tracking_note": "20 Hz relay status paired with latest fresh measured joints at receipt; "
        "asynchronous samples, not exact 100 Hz command/feedback pairs. Missing samples are not zero error.",
        "interrupt_note": "Published chunks continue in the relay after client exit; only matching "
        "completion acknowledgments prove completion. Replay must honor the saved command schema/integration "
        "and targets; do not reinterpret archived actions using a newer decoder.",
    }


def add_endpoint_errors(chunks, urdfs):
    from .labs_kinematics import URDFFK, vector_pose

    with tempfile.TemporaryDirectory() as directory:
        solvers = {}
        for side in ("left", "right"):
            path = Path(directory) / f"{side}.urdf"
            path.write_text(urdfs[side])
            solvers[side] = URDFFK(path, side)
        for chunk in chunks:
            settled = [s for s in chunk["tracking_samples"]
                       if s["status"].get("phase") == "holding" and s["measured_joints"] is not None]
            command = chunk.get("command", {})
            targets = command.get("targets", [])
            if chunk["execution"] != "completed" or not settled or not targets:
                continue
            q = settled[-1]["measured_joints"]
            errors = {}
            for side, qi, pi in (("left", 0, 0), ("right", 7, 9)):
                measured = solvers[side](q[qi:qi+7])
                from .labs_joint_inference import COMMAND_SCHEMA as JOINT_SCHEMA, split_targets

                if command.get("schema") == JOINT_SCHEMA:
                    joints, _ = split_targets(targets)
                    desired = solvers[side](joints[-1, qi:qi+7])
                    chunk["summary"]["model_endpoint_joint_error_rad"] = (np.asarray(q) - joints[-1]).tolist()
                else:
                    desired = vector_pose(targets[-1][pi:pi+9])
                cosine = (np.trace(desired[:3, :3] @ measured[:3, :3].T) - 1) / 2
                errors[side] = {
                    "position_m": float(np.linalg.norm(desired[:3, 3] - measured[:3, 3])),
                    "rotation_rad": float(np.arccos(np.clip(cosine, -1, 1))),
                }
            chunk["summary"]["model_endpoint_error"] = errors


def tracking_summary(chunk):
    pairs = []
    for sample in chunk["tracking_samples"]:
        actual, goal = sample["measured_joints"], sample["status"].get("holding_target")
        if actual is None or goal is None or sample["status"].get("phase") not in ("executing", "holding"):
            continue
        actual, goal = np.asarray(actual), np.asarray(goal)
        if actual.shape == goal.shape == (14,) and np.isfinite(actual).all() and np.isfinite(goal).all():
            pairs.append(actual - goal)
    result = {"samples": len(pairs), "completed": chunk["execution"] == "completed"}
    if pairs:
        errors = np.asarray(pairs)
        result.update(
            sampled_max_abs_joint_error_rad=float(np.abs(errors).max()),
            sampled_rms_per_joint_rad=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
            last_sample_error_rad=errors[-1].tolist(),
        )
    return result


def export_journal(directory, reason="recovered_unconfirmed"):
    directory = Path(directory)
    with (directory / "actions.journal.msgpack").open("rb") as stream:
        events = list(msgpack.Unpacker(stream, raw=False, strict_map_key=False))
    bundle = assemble(events, reason)
    target = directory / "actions.msgpack"
    temporary = target.with_suffix(".msgpack.tmp")
    with temporary.open("wb") as stream:
        stream.write(msgpack.packb(bundle, use_bin_type=True))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    return target


def load_recording(path):
    with Path(path).open("rb") as stream:
        bundle = msgpack.unpack(stream, raw=False, strict_map_key=False)
    if bundle.get("schema") != SCHEMA:
        raise ValueError("Expected Labs inference recording v1")
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="actions.msgpack or a run directory with a journal")
    parser.add_argument("--recover", action="store_true", help="Rebuild actions.msgpack from a crash journal")
    parser.add_argument("--html", type=Path, help="Write an offline interactive trajectory/feedback viewer")
    args = parser.parse_args(argv)
    path = export_journal(args.path) if args.recover else args.path
    if path.is_dir():
        path = path / "actions.msgpack"
    bundle = load_recording(path)
    print(json.dumps({"file": str(path), "exit_reason": bundle["exit_reason"],
                      "chunks": [{"index": c["index"], "execution": c["execution"], **c["summary"]}
                                 for c in bundle["chunks"]]}, indent=2))
    if args.html:
        from .labs_action_viewer import write_viewer
        write_viewer(bundle, args.html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
