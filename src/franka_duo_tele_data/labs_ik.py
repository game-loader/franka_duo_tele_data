"""Persistent subprocess interface to the offline MoveIt/KDL executable.

Source ROS and the package install before using. The executable has no robot
command publishers. A successful IK is not a collision or trajectory check.
"""

from __future__ import annotations

import os
import selectors
import subprocess
from pathlib import Path

import numpy as np

from .labs_kinematics import vector_pose


class MoveItKDL:
    def __init__(self, executable: Path, config: Path, side: str):
        if side not in ("left", "right"):
            raise ValueError("Expected left or right")
        env = os.environ.copy()
        env.pop("CYCLONEDDS_URI", None)
        env["ROS_DOMAIN_ID"] = "213"
        env["ROS_LOCALHOST_ONLY"] = "1"
        env["RCUTILS_LOGGING_USE_STDOUT"] = "0"
        self.process = subprocess.Popen(
            [
                str(executable),
                str(config / f"{side}.urdf"),
                str(config / f"{side}.srdf"),
                side,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def solve(self, pose9, seed):
        vector_pose(pose9)  # Validate column-6D orientation and finite values.
        seed = np.asarray(seed, dtype=np.float64)
        if seed.shape != (7,) or not np.isfinite(seed).all():
            raise ValueError("Expected seven finite seed angles")
        if self.process.poll() is not None:
            raise RuntimeError("MoveIt/KDL subprocess exited")
        self.process.stdin.write(" ".join(map(str, np.r_[pose9, seed])) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(timeout=15):
            self.close()
            raise TimeoutError("MoveIt/KDL solver response timeout")
        result = self.process.stdout.readline().split()
        if not result or result[0] not in ("OK", "FAIL"):
            raise RuntimeError(f"Invalid IK response: {result}")
        if result[0] == "FAIL":
            return {"success": False, "reason": " ".join(result[1:])}
        values = np.asarray(result[1:], dtype=float)
        if values.shape != (10,) or not np.isfinite(values).all():
            raise RuntimeError("Invalid IK result values")
        return {
            "success": True,
            "joint_positions": values[:7],
            "position_error_m": float(values[7]),
            "rotation_error_rad": float(values[8]),
            "max_joint_delta_rad": float(values[9]),
        }

    def close(self):
        self.selector.close()
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
