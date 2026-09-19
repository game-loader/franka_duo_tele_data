"""URDF FK and column-based Cartesian encoding for the labs FR3 station."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .action_spec import matrix_to_rot6d, rot6d_to_matrix


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def joint_positions(names, positions, side):
    """Match explicit side-prefixed or bare FR3 names; never guess array order."""
    if (
        side not in ("left", "right")
        or len(names) != len(positions)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Invalid side or joint names")
    values = dict(zip(names, positions, strict=True))
    result = []
    for index in range(1, 8):
        candidates = [
            n for n in (f"{side}_fr3_joint{index}", f"fr3_joint{index}") if n in values
        ]
        if len(candidates) != 1:
            raise ValueError(f"Missing or ambiguous joint {index} for {side}")
        result.append(float(values[candidates[0]]))
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite joint positions")
    return result


class URDFFK:
    """Precompute fixed transforms; return T_link0_link8 for seven radians."""

    def __init__(self, path: Path, side: str):
        self.side = side
        self.base, self.tip = f"{side}_fr3_link0", f"{side}_fr3_link8"
        root = ET.parse(path).getroot()
        parents = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain = []
        tip = self.tip
        while tip != self.base:
            joint = parents[tip]
            chain.append(joint)
            tip = joint.find("parent").get("link")
        self.chain = []
        self.names = []
        self.bounds = []
        for joint in reversed(chain):
            origin = joint.find("origin")
            xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
            roll, pitch, yaw = [float(x) for x in origin.get("rpy", "0 0 0").split()]
            fixed = np.eye(4)
            fixed[:3, :3] = (
                rotation([0, 0, 1], yaw)
                @ rotation([0, 1, 0], pitch)
                @ rotation([1, 0, 0], roll)
            )
            fixed[:3, 3] = xyz
            if joint.get("type") == "fixed":
                self.chain.append((fixed, None))
            elif joint.get("type") == "revolute":
                axis = np.array(
                    [float(x) for x in joint.find("axis").get("xyz").split()]
                )
                self.chain.append((fixed, axis))
                self.names.append(joint.get("name"))
                limit = joint.find("limit")
                self.bounds.append(
                    (float(limit.get("lower")), float(limit.get("upper")))
                )
            else:
                raise ValueError("FR3 chain must have fixed or revolute joints")
        if self.names != [f"{side}_fr3_joint{i}" for i in range(1, 8)]:
            raise ValueError("Unexpected FR3 chain")

    def __call__(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Expected seven finite joint angles in radians")
        transform = np.eye(4)
        index = 0
        for fixed, axis in self.chain:
            transform = transform @ fixed
            if axis is not None:
                transform[:3, :3] = transform[:3, :3] @ rotation(axis, q[index])
                index += 1
        return transform


def pose_vector(transform):
    return np.concatenate(
        (transform[:3, 3], matrix_to_rot6d(transform[:3, :3]))
    ).astype(np.float32)


def vector_pose(vector):
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (9,) or not np.isfinite(vector).all():
        raise ValueError("Expected xyz and two rotation columns")
    transform = np.eye(4)
    transform[:3, 3] = vector[:3]
    transform[:3, :3] = rot6d_to_matrix(vector[3:])
    return transform
