"""ROS-free tests for the ZED PnP extrinsic calibration: geometry, detection, solving, session, web."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from franka_duo_tele_data.mcap_to_lerobot import invert_transform, make_transform  # noqa: E402
from franka_duo_tele_data.zed_pnp_calib import (  # noqa: E402
    BeadDetection,
    CalibrationSession,
    CalibrationWebApp,
    Capture,
    Sample,
    bead_point_in_base,
    calibrate,
    detect_bead,
    hue_ranges_from_args,
    intersect_height,
    intrinsics_from_camera_info,
    load_calibration,
    load_samples,
    main,
    project_points,
    solve_extrinsics,
    stationary_pose,
    transform_difference,
    write_samples,
)

k_matrix = np.array([[520.0, 0.0, 320.0], [0.0, 520.0, 180.0], [0.0, 0.0, 1.0]])


def camera_pose() -> np.ndarray:
    """Camera 0.9 m above the base, pitched 55 degrees down, looking at +x; ROS optical axes."""
    pitch = math.radians(55)
    r_base_from_link = np.array(
        [
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)],
        ]
    )
    # optical (z forward, x right, y down) -> link (x forward, y left, z up)
    r_link_from_optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    return make_transform((-0.1, 0.0, 0.9), r_base_from_link @ r_link_from_optical).astype(np.float64)


def synthetic_points(rng, count=14) -> np.ndarray:
    xy = rng.uniform([0.3, -0.3], [0.6, 0.3], size=(count, 2))
    z = rng.choice([-0.2, -0.1, 0.0], size=count) + rng.normal(0, 0.005, size=count)
    return np.column_stack([xy, z])


def identity_manifest(tmp_path, t_base_from_cam):
    identity = np.eye(4).tolist()
    names = (
        "left_arm_world",
        "right_arm_world",
        "zed_mount_world",
        "new_base_world",
        "T_newbase_from_left_arm_base",
        "T_newbase_from_right_arm_base",
        "mount_to_optical",
    )
    transforms = dict.fromkeys(names, identity)
    transforms["T_newbase_from_zed_optical"] = np.asarray(t_base_from_cam).tolist()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"coordinate_transforms": transforms}))
    return path


def test_projection_and_ray_intersection_round_trip():
    t_base_from_cam = camera_pose()
    t_cam_from_base = invert_transform(t_base_from_cam)
    point = np.array([0.45, 0.1, -0.15])
    pixel = project_points(point, k_matrix, t_cam_from_base)[0]
    recovered = intersect_height(pixel, k_matrix, t_base_from_cam, point[2])
    assert np.allclose(recovered, point, atol=1e-5)
    with pytest.raises(ValueError):
        project_points(t_base_from_cam[:3, 3] + np.array([0, 0, 1.0]), k_matrix, t_cam_from_base)


def test_bead_point_uses_ee_z_axis_and_arm_base_transform():
    t_base_from_link0 = make_transform((0.0, 0.2, 0.0), [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    t_link0_ee = make_transform((0.5, 0.0, 0.3), [[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # z down
    point = bead_point_in_base(t_link0_ee, t_base_from_link0, 0.05)
    assert np.allclose(point, (0.0, 0.7, 0.25), atol=1e-6)
    with pytest.raises(ValueError):
        bead_point_in_base(t_link0_ee, t_base_from_link0, math.nan)


def test_solve_recovers_pose_and_flags_outlier():
    rng = np.random.default_rng(3)
    t_base_from_cam = camera_pose()
    t_cam_from_base = invert_transform(t_base_from_cam)
    points = synthetic_points(rng)
    pixels = project_points(points, k_matrix, t_cam_from_base) + rng.normal(0, 0.3, size=(len(points), 2))
    pixels[5] += (25.0, -18.0)  # one bad click
    result = solve_extrinsics(points, pixels, k_matrix)
    translation, rotation = transform_difference(t_base_from_cam, result.t_base_from_cam)
    assert translation < 0.01 and rotation < math.radians(0.5)
    assert 5 not in result.inliers and len(result.inliers) == len(points) - 1
    assert result.reprojection_px[5] > 10
    assert len(result.leave_one_out_xy_m) == len(result.inliers)
    assert max(result.leave_one_out_xy_m) < 0.01
    with pytest.raises(ValueError):
        solve_extrinsics(points[:5], pixels[:5], k_matrix)
    flat = points.copy()
    flat[:, 2] = 0.0
    with pytest.raises(ValueError, match="coplanar"):
        solve_extrinsics(flat, project_points(flat, k_matrix, t_cam_from_base), k_matrix)


def test_detect_orange_square_and_ignore_green_and_specks():
    image = np.full((180, 320, 3), 235, dtype=np.uint8)  # bright table
    cv2.circle(image, (200, 100), 12, (0, 0, 0), -1)  # dark tray
    cv2.circle(image, (100, 60), 20, (160, 210, 170), -1)  # pale green cup
    cv2.rectangle(image, (236, 126), (245, 135), (255, 120, 20), -1)  # orange 10x10 square
    image[10:12, 10:12] = (255, 120, 20)  # speck below min area
    detection = detect_bead(image)
    assert detection is not None
    assert abs(detection.pixel[0] - 240.5) < 0.5 and abs(detection.pixel[1] - 130.5) < 0.5
    assert detection.area_px > 20 and 0.5 < detection.circularity < 1.0
    # A red marker is not orange by default but is found with an explicit hue range.
    red = np.full((60, 60, 3), 235, dtype=np.uint8)
    cv2.circle(red, (30, 30), 6, (220, 30, 30), -1)
    assert detect_bead(red) is None
    assert detect_bead(red, hue_ranges=((0, 10), (170, 180))) is not None
    assert detect_bead(np.full((20, 20, 3), 235, dtype=np.uint8)) is None
    with pytest.raises(ValueError):
        detect_bead(np.zeros((20, 20), dtype=np.uint8))


def test_hue_range_argument_validation():
    assert hue_ranges_from_args(None) == ((5, 28),)
    assert hue_ranges_from_args([[0, 10], [170, 180]]) == ((0, 10), (170, 180))
    with pytest.raises(ValueError):
        hue_ranges_from_args([[20, 10]])


def test_intrinsics_from_camera_info_validates():
    info = SimpleNamespace(k=k_matrix.reshape(-1).tolist(), d=[], width=640, height=360)
    matrix, distortion, size = intrinsics_from_camera_info(info)
    assert np.allclose(matrix, k_matrix) and distortion.shape == (5,) and size == (640, 360)
    with pytest.raises(ValueError):
        intrinsics_from_camera_info(SimpleNamespace(k=[0] * 9, d=[], width=640, height=360))


def test_stationary_pose_rejects_motion_and_staleness():
    def pose(x, t):
        position = SimpleNamespace(x=x, y=0.0, z=0.0)
        return (t, SimpleNamespace(pose=SimpleNamespace(position=position)))

    now = 100.0
    still = [pose(0.5, now - 0.4), pose(0.5004, now - 0.2), pose(0.5002, now - 0.01)]
    assert stationary_pose(still, now=now, window_s=0.5, tolerance_m=0.001).pose.position.x == 0.5002
    moving = [pose(0.5, now - 0.4), pose(0.52, now - 0.01)]
    with pytest.raises(RuntimeError):
        stationary_pose(moving, now=now, window_s=0.5, tolerance_m=0.001)
    with pytest.raises(TimeoutError):
        stationary_pose([pose(0.5, now - 2.0)], now=now, window_s=0.5, tolerance_m=0.001)
    with pytest.raises(TimeoutError):
        stationary_pose([], now=now, window_s=0.5, tolerance_m=0.001)


def test_calibrate_document_and_file_round_trip(tmp_path):
    rng = np.random.default_rng(7)
    t_base_from_cam = camera_pose()
    t_cam_from_base = invert_transform(t_base_from_cam)
    points = synthetic_points(rng, 12)
    pixels = project_points(points, k_matrix, t_cam_from_base) + rng.normal(0, 0.2, size=(12, 2))
    flat_identity = tuple(np.eye(4).reshape(-1))
    samples = [
        Sample("left" if i % 2 else "right", tuple(pixels[i]), tuple(points[i]), flat_identity, 1000 + i)
        for i in range(12)
    ]
    nominal = t_base_from_cam.copy()
    nominal[:3, 3] += (0.02, 0.0, -0.01)
    camera = {"K": k_matrix.tolist(), "distortion": [0.0] * 5, "width": 640, "height": 360}
    document = calibrate(
        samples, k_matrix, np.zeros(5), (640, 360), tool_offset_m=0.05, nominal_t_base_from_cam=nominal
    )
    assert document["stats"]["reprojection_max_px"] < 1.5
    assert set(document["per_side"]) == {"left", "right"}
    expected = math.sqrt(0.02**2 + 0.01**2)
    assert abs(document["nominal_comparison"]["translation_m"] - expected) < 0.01
    out = tmp_path / "calib.json"
    out.write_text(json.dumps(document))
    loaded = load_calibration(out)
    assert np.allclose(loaded["T_newbase_from_zed_optical"], document["T_newbase_from_zed_optical"])

    samples_path = tmp_path / "samples.json"
    write_samples(samples_path, samples, camera, 0.05)
    reloaded, meta = load_samples(samples_path)
    assert len(reloaded) == 12 and reloaded[3].side == "left" and meta["tool_offset_m"] == 0.05

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(ValueError):
        load_calibration(bad)


def test_solve_mode_runs_without_ros(tmp_path, capsys):
    rng = np.random.default_rng(11)
    t_base_from_cam = camera_pose()
    t_cam_from_base = invert_transform(t_base_from_cam)
    points = synthetic_points(rng, 10)
    pixels = project_points(points, k_matrix, t_cam_from_base)
    flat_identity = tuple(np.eye(4).reshape(-1))
    samples = [Sample("left", tuple(pixels[i]), tuple(points[i]), flat_identity, i) for i in range(10)]
    samples_path = tmp_path / "samples.json"
    camera = {"K": k_matrix.tolist(), "distortion": [0.0] * 5, "width": 640, "height": 360}
    write_samples(samples_path, samples, camera, 0.04)
    manifest = identity_manifest(tmp_path, t_base_from_cam)
    output = tmp_path / "out.json"
    rc = main(["--manifest", str(manifest), "--solve", str(samples_path), "--output", str(output)])
    assert rc == 0
    document = load_calibration(output)
    assert document["stats"]["reprojection_max_px"] < 1e-2
    assert document["nominal_comparison"]["translation_m"] < 1e-3
    assert "wrote" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--manifest", str(manifest)])  # collection needs --tool-offset-m
    with pytest.raises(SystemExit):
        main(["--manifest", str(manifest), "--tool-offset-m", "0.04", "--hue-range", "30", "10"])
    with pytest.raises(SystemExit):
        main(["--manifest", str(manifest), "--tool-offset-m", "0.04"])  # needs --web or --auto-detect


def make_session(tmp_path, rng):
    """Session backed by synthetic captures whose pixels come from the true camera pose."""
    t_base_from_cam = camera_pose()
    t_cam_from_base = invert_transform(t_base_from_cam)
    points = iter(synthetic_points(rng, 20))
    frame = np.full((360, 640, 3), 200, dtype=np.uint8)
    truth: dict[str, np.ndarray] = {}

    def capture(side):
        point = next(points)
        truth["pixel"] = project_points(point, k_matrix, t_cam_from_base)[0]
        detection = BeadDetection((float(truth["pixel"][0]), float(truth["pixel"][1])), 30.0, 0.8)
        return Capture(side, frame.copy(), np.eye(4), point, detection, 5)

    def solve(samples):
        return calibrate(
            samples,
            k_matrix,
            np.zeros(5),
            (64, 36),
            tool_offset_m=0.03,
            nominal_t_base_from_cam=t_base_from_cam,
        )

    output_dir = tmp_path / "zed"
    output_dir.mkdir()
    session = CalibrationSession(
        capture_fn=capture,
        live_rgb_fn=lambda: frame,
        solve_fn=solve,
        samples=[],
        samples_path=output_dir / "samples.json",
        output_dir=output_dir,
        output=tmp_path / "calib.json",
        camera_doc={"K": k_matrix.tolist(), "distortion": [0.0] * 5, "width": 640, "height": 360},
        tool_offset_m=0.03,
    )
    return session, truth


def test_session_capture_accept_drop_solve(tmp_path):
    session, truth = make_session(tmp_path, np.random.default_rng(2))
    with pytest.raises(ValueError):
        session.accept(1.0, 1.0, manual=True)  # nothing captured yet
    with pytest.raises(ValueError):
        session.capture("middle")
    for i in range(8):
        result = session.capture("left" if i % 2 else "right")
        assert result["auto_pixel"] is not None and session.state()["pending"] == result["side"]
        u, v = result["auto_pixel"]
        state = session.accept(u, v, manual=False)
        assert len(state["samples"]) == i + 1 and state["pending"] is None
    assert (session.output_dir / "frame_00_right.png").exists()
    reloaded, _ = load_samples(session.samples_path)
    assert len(reloaded) == 8
    session.capture("left")
    with pytest.raises(ValueError):
        session.accept(5000.0, 5.0, manual=True)  # outside the 640x360 frame
    session.discard()
    assert session.state()["pending"] is None
    session.drop(0)
    assert len(session.samples) == 7
    with pytest.raises(ValueError):
        session.drop(99)
    summary = session.solve(write=False)
    assert summary["stats"]["reprojection_max_px"] < 1e-3 and summary["written"] is None
    assert not session.output.exists()
    summary = session.solve(write=True)
    assert summary["written"] == str(session.output) and load_calibration(session.output)


def test_web_app_routes(tmp_path):
    session, truth = make_session(tmp_path, np.random.default_rng(5))
    app = CalibrationWebApp(session)
    status, ctype, body = app.handle("GET", "/", b"")
    assert status == 200 and ctype.startswith("text/html") and b"Capture" in body
    status, ctype, body = app.handle("GET", "/live.jpg?123", b"")
    assert status == 200 and ctype == "image/jpeg" and body[:2] == b"\xff\xd8"
    status, _, body = app.handle("GET", "/pending.jpg", b"")
    assert status == 400 and "no frozen frame" in json.loads(body)["error"]
    for i in range(6):
        status, _, body = app.handle("POST", "/capture", json.dumps({"arm": "left"}).encode())
        assert status == 200
        u, v = json.loads(body)["auto_pixel"]
        status, _, _ = app.handle("GET", "/pending.jpg", b"")
        assert status == 200
        status, _, body = app.handle(
            "POST", "/accept", json.dumps({"u": u + 0.1, "v": v, "manual": True}).encode()
        )
        assert status == 200 and len(json.loads(body)["samples"]) == i + 1
    status, _, body = app.handle("GET", "/state", b"")
    assert status == 200 and json.loads(body)["samples"][0]["manual"] is True
    status, _, body = app.handle("POST", "/solve", json.dumps({"write": False}).encode())
    assert status == 200 and json.loads(body)["stats"]["samples"] == 6
    status, _, body = app.handle("POST", "/drop", b'{"index": 40}')
    assert status == 400
    status, _, body = app.handle("POST", "/accept", b"not json")
    assert status == 400
    assert app.handle("GET", "/nope", b"")[0] == 404
