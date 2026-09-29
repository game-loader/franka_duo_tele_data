#!/usr/bin/env python3
"""Losslessly archive closed Labs MCAP files, verifying before optional replacement."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def digest(stream):
    result = hashlib.sha256()
    while data := stream.read(8 * 1024 * 1024):
        result.update(data)
    return result.hexdigest()


def sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def verify_archive(path, expected):
    proc = subprocess.Popen(["zstd", "-d", "-q", "-c", str(path)], stdout=subprocess.PIPE)
    try:
        with proc.stdout:
            actual = digest(proc.stdout)
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if proc.wait() != 0 or actual != expected:
        raise ValueError(f"Decompressed SHA-256 mismatch: {path}")


def archive_one(source, threads, replace, level=3):
    archive = source.with_suffix(".mcap.zst")
    report_path = archive.with_suffix(".zst.verification.json")
    original = source.stat()
    started = time.monotonic()
    with source.open("rb") as stream:
        sha = digest(stream)
    if not archive.exists():
        partial = archive.with_name(archive.name + ".partial")
        # Preserve any interrupted archive for inspection instead of overwriting it.
        if partial.exists():
            raise FileExistsError(partial)
        disk = os.statvfs(source.parent)
        available = disk.f_bfree if os.geteuid() == 0 else disk.f_bavail
        if available * disk.f_frsize < original.st_size + 2 * 1024**3:
            raise OSError("Insufficient temporary space; source retained")
        subprocess.run(["nice", "-n", "15", "zstd", "--ultra", f"-{level}", f"-T{threads}",
                        "--check", "-q", str(source), "-o", str(partial)], check=True)
        verify_archive(partial, sha)
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        os.rename(partial, archive)
        sync_directory(source.parent)
    else:
        verify_archive(archive, sha)
    current = source.stat()
    def identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    if identity(current) != identity(original):
        raise ValueError(f"Source changed while archiving: {source}")
    report = {
        "source": str(source), "archive": str(archive), "original_bytes": original.st_size,
        "compressed_bytes": archive.stat().st_size, "sha256_uncompressed": sha,
        "verified_lossless": True, "compression": "zstd", "level": level,
        "threads": threads, "seconds": time.monotonic() - started,
        "source_mtime_ns": original.st_mtime_ns, "source_mode": original.st_mode & 0o7777,
        "original_retained": True, "checked_at": datetime.now().astimezone().isoformat(),
    }
    save_json(report_path, report)
    if replace and archive.stat().st_size < original.st_size:
        source.unlink()
        sync_directory(source.parent)
        report["original_retained"] = False
        save_json(report_path, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, nargs="+", required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--level", type=int, default=3, help="Zstandard level 1..22; default 3 prioritizes speed")
    p.add_argument("--replace-verified", action="store_true")
    args = p.parse_args()
    if not 1 <= args.threads <= 16:
        p.error("threads must be between 1 and 16")
    if not 1 <= args.level <= 22:
        p.error("level must be between 1 and 22")
    args.job.mkdir(parents=True, exist_ok=True)
    with (args.job / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot_path = args.job / "snapshot.json"
        settings = {"roots": [str(root.resolve()) for root in args.input],
                    "replace_verified": args.replace_verified, "level": args.level}
        if snapshot_path.exists():
            snapshot = json.loads(snapshot_path.read_text())
            if snapshot["settings"] != settings:
                raise ValueError("Archive job settings changed")
        else:
            sources = []
            for root in args.input:
                for source in sorted(root.glob("*/mcap/*.mcap")):
                    if source.is_symlink():
                        raise ValueError(f"Expected original file, not symlink: {source}")
                    record = json.loads((source.parent.parent / "record_metadata.json").read_text())
                    if record.get("status") != "completed":
                        raise ValueError(f"Unfinished recording: {source}")
                    sources.append(str(source.resolve()))
            snapshot = {"settings": settings, "files": sources,
                        "created_at": datetime.now().astimezone().isoformat()}
            save_json(snapshot_path, snapshot)
        completed = []
        for index, name in enumerate(snapshot["files"], 1):
            source = Path(name)
            if not source.exists():
                archive = source.with_suffix(".mcap.zst")
                report = json.loads(archive.with_suffix(".zst.verification.json").read_text())
                if report["original_retained"] or not report["verified_lossless"]:
                    raise ValueError(f"Unexpected missing source: {source}")
                verify_archive(archive, report["sha256_uncompressed"])
            else:
                print(json.dumps({"phase": "compressing", "index": index,
                                  "total": len(snapshot["files"]), "source": name}), flush=True)
                report = archive_one(source, args.threads, args.replace_verified, args.level)
            completed.append(report)
            progress = {"phase": "progress", "completed": len(completed), "total": len(snapshot["files"]),
                        "original_bytes": sum(r["original_bytes"] for r in completed),
                        "compressed_bytes": sum(r["compressed_bytes"] for r in completed),
                        "freed_bytes": sum(r["original_bytes"] - r["compressed_bytes"]
                                           for r in completed if not r["original_retained"]),
                        "last": report, "updated_at": datetime.now().astimezone().isoformat()}
            save_json(args.job / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        save_json(args.job / "completed.json", {"files": len(completed), "reports": completed})


if __name__ == "__main__":
    main()
