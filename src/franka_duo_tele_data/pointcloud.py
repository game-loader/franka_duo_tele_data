# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Depth-image deprojection and RL100-style point-cloud sampling.

The RGB-D -> camera XYZ -> rigid transform -> XYZ-only spatial sampling order
follows RL100's ``gym_util/mjpc_wrapper.py`` and its
``mujoco_point_cloud.py`` implementation.  The deprojection API is compatible
with LeRobot's DP3 point-cloud utilities.  NumPy keeps this package usable on
the robot host without adding PyTorch/PyTorch3D to the base dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def farthest_point_sample(
    points: NDArray,
    num_points: int,
    *,
    seed: int = 0,
    candidate_limit: int | None = None,
) -> NDArray[np.float32]:
    """Sample a fixed-size point set with deterministic Euclidean FPS.

    The geometry is taken from the first three columns (XYZ); additional
    columns such as RGB follow the selected points.  A deterministic
    subsampling limit keeps the quadratic CPU loop bounded for dense depth
    images, matching the practical RL-100/DP3 preprocessing pattern.
    """

    value = np.asarray(points, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("points must contain only finite values")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if candidate_limit is not None and candidate_limit < num_points:
        raise ValueError("candidate_limit must be >= num_points")
    if value.shape[0] == 0:
        raise ValueError("points must not be empty")

    if candidate_limit is not None and value.shape[0] > candidate_limit:
        candidate_indices = np.linspace(0, value.shape[0] - 1, candidate_limit, dtype=np.int64)
        value = value[candidate_indices]

    if value.shape[0] <= num_points:
        if value.shape[0] == num_points:
            return np.ascontiguousarray(value)
        rng = np.random.default_rng(seed)
        padding = rng.choice(value.shape[0], num_points - value.shape[0], replace=True)
        return np.ascontiguousarray(np.concatenate((value, value[padding]), axis=0))

    rng = np.random.default_rng(seed)
    selected = np.empty(num_points, dtype=np.int64)
    selected[0] = int(rng.integers(value.shape[0]))
    geometry = value[:, :3]
    distances = np.full(value.shape[0], np.inf, dtype=np.float32)
    for index in range(1, num_points):
        current = geometry[selected[index - 1]]
        distances = np.minimum(distances, np.sum((geometry - current) ** 2, axis=1))
        selected[index] = int(np.argmax(distances))
    return np.ascontiguousarray(value[selected])


def _voxel_representatives(
    points: NDArray[np.floating],
    voxel_size: float,
    *,
    origin: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Return one real point nearest the center of each occupied voxel."""

    value = np.asarray(points, dtype=np.float32)
    keys = np.floor((value[:, :3] - origin) / float(voxel_size)).astype(np.int64)
    packed = keys.view(np.dtype((np.void, keys.dtype.itemsize * 3))).reshape(-1)
    unique_packed, inverse = np.unique(packed, return_inverse=True)
    unique_keys = unique_packed.view(keys.dtype).reshape(-1, 3)
    centers = origin + (unique_keys.astype(np.float32) + 0.5) * float(voxel_size)
    distance = np.sum((value[:, :3] - centers[inverse]) ** 2, axis=1)
    # Sorting by voxel id and then distance makes the first entry of each
    # group the nearest *measured* point, rather than a synthetic centroid.
    order = np.lexsort((distance, inverse))
    ordered_groups = inverse[order]
    first = order[np.r_[True, ordered_groups[1:] != ordered_groups[:-1]]]
    return np.ascontiguousarray(value[first])


def _occupied_voxel_count(
    points: NDArray[np.float32],
    voxel_size: float,
    *,
    origin: NDArray[np.float32],
) -> int:
    keys = np.floor((points - origin) / float(voxel_size)).astype(np.int64)
    packed = keys.view(np.dtype((np.void, keys.dtype.itemsize * 3))).reshape(-1)
    return int(np.unique(packed).size)


def adaptive_voxel_sample(
    points: NDArray,
    num_points: int,
    *,
    seed: int = 0,
    max_iterations: int = 4,
) -> NDArray[np.float32]:
    """Downsample XYZ points to exactly ``num_points`` with an adaptive grid.

    The voxel edge length is selected by a few fast surface-density updates
    (``s <- s * sqrt(occupied / target)``), which avoids the many full
    ``np.unique`` passes required by a strict binary search.  One measured
    point nearest each voxel center is retained, then the representatives are
    uniformly thinned if the count is larger than the target.  If the input has
    fewer unique points, deterministic seeded repetition is used as a last
    resort.
    """

    value = np.asarray(points, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("points must contain only finite values")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if value.shape[0] == 0:
        raise ValueError("points must not be empty")

    geometry = np.ascontiguousarray(value[:, :3], dtype=np.float32)
    if geometry.shape[0] <= num_points:
        if geometry.shape[0] == num_points:
            return geometry
        rng = np.random.default_rng(seed)
        padding = rng.choice(geometry.shape[0], num_points - geometry.shape[0], replace=True)
        return np.ascontiguousarray(np.concatenate((geometry, geometry[padding]), axis=0))

    origin = geometry.min(axis=0)
    span = geometry.max(axis=0) - origin
    max_span = float(np.max(span))
    if not np.isfinite(max_span) or max_span <= 1e-8:
        return np.repeat(geometry[:1], num_points, axis=0)

    # For a depth image the occupied cells mostly lie on surfaces, so the
    # occupied count scales approximately with 1 / s² rather than 1 / s³.
    voxel_size = max_span / math.sqrt(float(num_points))
    count = 0
    for _ in range(max_iterations):
        count = _occupied_voxel_count(geometry, voxel_size, origin=origin)
        if count == 0 or count == num_points:
            break
        voxel_size = max(voxel_size * math.sqrt(count / float(num_points)), max_span * 1e-7)
    # The update above changes the edge length; refresh the count so the
    # representative pass and the boundary correction use the same grid.
    count = _occupied_voxel_count(geometry, voxel_size, origin=origin)

    # Ensure the representative set is not below target when the inexpensive
    # fixed iteration budget lands just on the wrong side of a voxel boundary.
    for _ in range(3):
        if count >= num_points:
            break
        voxel_size = max(voxel_size * 0.92, max_span * 1e-7)
        count = _occupied_voxel_count(geometry, voxel_size, origin=origin)

    representatives = _voxel_representatives(geometry, voxel_size, origin=origin)
    if representatives.shape[0] > num_points:
        # The voxel-key order is spatially stable; linspace keeps coverage
        # across the whole workspace without the quadratic cost of FPS.
        indices = (np.arange(num_points, dtype=np.int64) * representatives.shape[0]) // num_points
        representatives = representatives[indices]
    if representatives.shape[0] < num_points:
        rng = np.random.default_rng(seed)
        padding = rng.choice(representatives.shape[0], num_points - representatives.shape[0], replace=True)
        representatives = np.concatenate((representatives, representatives[padding]), axis=0)
    return np.ascontiguousarray(representatives[:, :3], dtype=np.float32)


def _as_vec3(value: Sequence[float] | NDArray[np.floating] | None, name: str) -> NDArray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    return array


def depth_to_point_cloud(
    depth: NDArray,
    camera_matrix: Sequence[float] | NDArray,
    *,
    depth_scale: float = 1.0,
    rgb: NDArray | None = None,
    extrinsics: NDArray | None = None,
    workspace_min: Sequence[float] | NDArray[np.floating] | None = None,
    workspace_max: Sequence[float] | NDArray[np.floating] | None = None,
    min_depth: float = 0.05,
    max_depth: float = 5.0,
    num_points: int | None = 512,
    seed: int | None = None,
) -> NDArray[np.float32]:
    """Deproject one depth image into a fixed-size XYZ or XYZRGB point set.

    ``camera_matrix`` accepts either ROS ``CameraInfo.k`` flattened row-major
    or a 3x3 matrix. ``extrinsics``, when supplied, transforms homogeneous
    camera optical-frame points into the desired stable training frame.
    """

    depth_array = np.asarray(depth)
    if depth_array.ndim != 2:
        raise ValueError(f"depth must have shape (height, width), got {depth_array.shape}.")
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive.")
    if not 0 <= min_depth < max_depth:
        raise ValueError("Expected 0 <= min_depth < max_depth.")
    if num_points is not None and num_points <= 0:
        raise ValueError("num_points must be positive or None.")

    intrinsics = np.asarray(camera_matrix, dtype=np.float32)
    if intrinsics.size != 9:
        raise ValueError(f"camera_matrix must contain 9 values, got shape {intrinsics.shape}.")
    intrinsics = intrinsics.reshape(3, 3)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if not np.isfinite(intrinsics).all() or fx <= 0 or fy <= 0:
        raise ValueError("camera_matrix must contain finite positive focal lengths.")

    depth_m = depth_array.astype(np.float32, copy=False) * float(depth_scale)
    valid = np.isfinite(depth_m) & (depth_m >= min_depth) & (depth_m <= max_depth)
    rows, columns = np.nonzero(valid)
    z = depth_m[rows, columns]
    points = np.column_stack(((columns - cx) * z / fx, (rows - cy) * z / fy, z)).astype(
        np.float32,
        copy=False,
    )

    if extrinsics is not None:
        transform = np.asarray(extrinsics, dtype=np.float32)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"extrinsics must be a finite 4x4 matrix, got {transform.shape}.")
        points = points @ transform[:3, :3].T + transform[:3, 3]

    lower = _as_vec3(workspace_min, "workspace_min")
    upper = _as_vec3(workspace_max, "workspace_max")
    if lower is not None:
        keep = np.all(points >= lower, axis=1)
        points, rows, columns = points[keep], rows[keep], columns[keep]
    if upper is not None:
        keep = np.all(points <= upper, axis=1)
        points, rows, columns = points[keep], rows[keep], columns[keep]

    if points.shape[0] == 0:
        raise ValueError("No valid depth points remain after filtering.")

    features = points
    if rgb is not None:
        rgb_array = np.asarray(rgb)
        if rgb_array.shape != (*depth_array.shape, 3):
            raise ValueError(f"rgb must have shape {(*depth_array.shape, 3)}, got {rgb_array.shape}.")
        colors = rgb_array[rows, columns].astype(np.float32)
        if np.issubdtype(rgb_array.dtype, np.integer):
            colors /= np.iinfo(rgb_array.dtype).max
        features = np.concatenate((points, colors), axis=1)

    if num_points is not None and features.shape[0] > num_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(features.shape[0], size=num_points, replace=False)
        features = features[indices]
    elif num_points is not None and features.shape[0] < num_points:
        rng = np.random.default_rng(seed)
        padding = rng.choice(features.shape[0], size=num_points - features.shape[0], replace=True)
        indices = np.concatenate((np.arange(features.shape[0]), padding))
        rng.shuffle(indices)
        features = features[indices]

    return np.ascontiguousarray(features, dtype=np.float32)
