#!/usr/bin/env python3
"""Export a fixed snapshot of successful Labs episodes with next-measured-state labels.

Reuses the existing MCAP and next-state converters. Source bags remain untouched;
file symlinks select episodes without copying the raw payload. Rerunning the same
command resumes the existing snapshot and completed MCAP conversion checkpoints.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--completed-before", help="Inclusive ISO-8601 recording end cutoff with timezone")
    parser.add_argument("--exclude-snapshot", type=Path, help="Exclude episodes included in a prior source snapshot")
    parser.add_argument("--expected-episodes", type=int, help="Require this exact selected episode count")
    args = parser.parse_args()
    cutoff = None
    if args.completed_before:
        cutoff = datetime.fromisoformat(args.completed_before)
        if cutoff.tzinfo is None:
            parser.error("--completed-before requires an explicit timezone")
    if args.expected_episodes is not None and args.expected_episodes < 1:
        parser.error("--expected-episodes must be positive")
    root, job, output, config = [p.resolve() for p in (args.input, args.job, args.output, args.config)]
    if args.workers < 1 or not args.task.strip():
        parser.error("Positive workers and nonempty task required")
    excluded_ids = set()
    excluded_snapshot = None
    if args.exclude_snapshot:
        exclusion_path = args.exclude_snapshot.resolve()
        exclusion_bytes = exclusion_path.read_bytes()
        previous = json.loads(exclusion_bytes)
        if Path(previous["settings"]["input"]).resolve() != root:
            parser.error("Excluded snapshot must refer to the same raw input root")
        excluded_ids = {e["episode_id"] for e in previous["included"]}
        excluded_snapshot = {"path": str(exclusion_path),
                             "sha256": hashlib.sha256(exclusion_bytes).hexdigest(),
                             "episodes": len(excluded_ids)}
    for destination in (job, output):
        if destination == root or root in destination.parents:
            parser.error("Output/job must be independent of raw input")
    job.mkdir(parents=True, exist_ok=True)
    snapshot_path = job / "source_snapshot.json"
    settings = {
        "input": str(root), "output": str(output), "task": args.task,
        "fps": 30, "urdf_sha256": {
            side: hashlib.sha256((config / f"{side}.urdf").read_bytes()).hexdigest()
            for side in ("left", "right")
        },
    }
    if cutoff is not None:
        settings["completed_before"] = cutoff.isoformat()
    if excluded_snapshot is not None:
        settings["exclude_snapshot"] = excluded_snapshot
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        if snapshot["settings"] != settings:
            raise ValueError("Snapshot settings changed; use a new job/output directory")
    else:
        if output.exists():
            raise FileExistsError(output)
        included, excluded = [], []
        for episode in sorted(p for p in root.iterdir() if p.is_dir()):
            if episode.name in excluded_ids:
                excluded.append({"episode_id": episode.name, "reason": "included in excluded snapshot"})
                continue
            ep = episode / "episode_metadata.json"
            rec = episode / "record_metadata.json"
            metadata = json.loads(ep.read_text()) if ep.exists() else {}
            recording = json.loads(rec.read_text()) if rec.exists() else {}
            status = (metadata.get("status"), metadata.get("label"), recording.get("status"))
            if status != ("SAVED", "REVIEW_SUCCESS", "completed"):
                excluded.append({"episode_id": episode.name, "status": list(status)})
                continue
            if cutoff is not None:
                end = recording.get("end_timestamp")
                if not isinstance(end, (int, float)) or not 0 < end <= cutoff.timestamp():
                    excluded.append({"episode_id": episode.name, "reason": "outside completion cutoff",
                                     "end_timestamp": end})
                    continue
            files = sorted((episode / "mcap").glob("*.mcap"))
            if not files or not (episode / "mcap/metadata.yaml").is_file():
                raise ValueError(f"Successful episode has incomplete bag: {episode}")
            identities = []
            for path in files:
                with path.open("rb") as stream:
                    first = stream.read(8)
                    stream.seek(-8, 2)
                    last = stream.read(8)
                if first != b"\x89MCAP0\r\n" or last != first:
                    raise ValueError(f"Unfinished MCAP file: {path}")
                stat = path.stat()
                identities.append({"path": str(path.relative_to(root)), "size": stat.st_size,
                                   "mtime_ns": stat.st_mtime_ns})
            included.append({"episode_id": episode.name, "episode_metadata": metadata,
                             "record_metadata": recording, "files": identities})
        if not included:
            raise ValueError("No completed REVIEW_SUCCESS episodes")
        if args.expected_episodes is not None and len(included) != args.expected_episodes:
            raise ValueError(f"Expected {args.expected_episodes} episodes, selected {len(included)}")
        snapshot = {"created_at": datetime.now().astimezone().isoformat(), "settings": settings,
                    "selection": "SAVED + REVIEW_SUCCESS + completed, closed MCAP files",
                    "task_authority": "explicit user instruction overrides original episode task text",
                    "included": included, "excluded": excluded}
        save_json(snapshot_path, snapshot)
    if args.expected_episodes is not None and len(snapshot["included"]) != args.expected_episodes:
        raise ValueError("Existing snapshot does not match --expected-episodes")
    selected = job / "selected_raw"
    for episode in snapshot["included"]:
        for identity in episode["files"]:
            src = root / identity["path"]
            stat = src.stat()
            if stat.st_size != identity["size"] or stat.st_mtime_ns != identity["mtime_ns"]:
                raise ValueError(f"Selected raw file changed: {src}")
            dest = selected / identity["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(src)
            elif not os.path.samefile(src, dest):
                raise ValueError(f"Unexpected selection file: {dest}")
    source = job / "source_state34"
    print(json.dumps({"stage": "selection", "episodes": len(snapshot["included"]),
                      "excluded": len(snapshot["excluded"]), "task": args.task}), flush=True)
    if not source.exists():
        subprocess.run([sys.executable, "-m", "franka_duo_tele_data.labs_mcap_to_lerobot",
                        "--input", str(selected), "--output", str(source), "--config", str(config),
                        "--task", args.task, "--workers", str(args.workers), "--fps", "30"], check=True)
    if not output.exists():
        print(json.dumps({"stage": "next_state20"}), flush=True)
        subprocess.run([sys.executable, "-m", "franka_duo_tele_data.labs_next_state_dataset",
                        "--input", str(source), "--output", str(output), "--config", str(config)], check=True)
    import pyarrow.parquet as pq

    info = json.loads((output / "meta/info.json").read_text())
    report = json.loads((output / "meta/next_state_validation.json").read_text())
    manifest = json.loads((output / "meta/conversion_manifest.json").read_text())
    expected = [e["episode_id"] for e in snapshot["included"]]
    actual = [e["original_episode"] for e in manifest["episodes"]]
    if (actual != expected or info["total_episodes"] != len(expected)
            or info["codebase_version"] != "v3.0" or manifest["task"] != args.task
            or report["status"] != "passed"
            or any(info["features"][k]["shape"] != [20] for k in ("observation.state", "action"))):
        raise ValueError("Final output does not match selected episodes/task/contract")
    if pq.read_table(output / "meta/tasks.parquet").to_pandas().index.tolist() != [args.task]:
        raise ValueError("Final LeRobot task table mismatch")
    shutil.copy2(snapshot_path, output / "meta/source_selection.json")
    save_json(job / "completed.json", {"output": str(output), "task": args.task, **report})
    print(json.dumps({"stage": "completed", "output": str(output), **report}), flush=True)


if __name__ == "__main__":
    main()
