"""Exercise driver bring-up and failure gates without ROS or robot connections."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ptp_home.sh"
MOCK_TOOL = r"""
import fcntl
import json
import os
import sys
import time
from pathlib import Path

root = Path(os.environ["PTP_MOCK_ROOT"])
cfg = json.loads((root / "mock.json").read_text())
tool = Path(sys.argv[0]).name
args = sys.argv[1:]
if tool == "timeout":
    os.execvp(args[1], args[1:])
if tool == "setsid":
    os.execvp(args[0], args)
if tool == "flock":
    try:
        fcntl.flock(int(args[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(1)
    sys.exit(0)
if tool == "ps":
    print("/opt/ros/lib/controller_manager/ros2_control_node --ros-args -r __ns:=/left/gripper")
    print("/opt/ros/lib/controller_manager/ros2_control_node --ros-args -r __ns:=/right/gripper")
    for process in cfg.get("processes", []):
        print(process)
    sys.exit(0)

fd = os.open(root / "calls.jsonl", os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
os.write(fd, (json.dumps(args) + "\n").encode())
os.close(fd)
side = next((part for arg in args for part in arg.split("/") if part in ("left", "right")), None)
if args[:2] == ["control", "list_controllers"]:
    count_file = root / f"{side}.queries"
    count = int(count_file.read_text()) + 1 if count_file.exists() else 1
    count_file.write_text(str(count))
    if count >= cfg.get("ready_after", {}).get(side, 10000):
        (root / f"{side}.ready").touch()
    if not (root / f"{side}.ready").exists():
        print(f"waiting for service /{side}/controller_manager/list_controllers", file=sys.stderr)
        sys.exit(124)
    for name in ("joint_state_broadcaster", "franka_robot_state_broadcaster"):
        print(f"{name} pkg/{name} \033[92mactive\033[0m")
    impedance = root / f"{side}.impedance"
    if impedance.exists():
        print(f"joint_impedance_controller pkg/JointImpedanceController {impedance.read_text()}")
    if cfg.get("other_active") == side:
        print("joint_trajectory_controller pkg/JointTrajectoryController active")
elif args[:2] == ["control", "switch_controllers"]:
    assert "--deactivate" in args and "--activate" not in args
    if cfg.get("deactivate_fail") == side:
        print("controller switch failed", file=sys.stderr)
        sys.exit(1)
    if cfg.get("impedance_stuck") != side:
        (root / f"{side}.impedance").write_text("inactive")
elif args[:2] == ["topic", "echo"]:
    topic = next(arg for arg in args if arg.startswith("/"))
    field = args[args.index("--field") + 1]
    if cfg.get("missing_topic") == topic.rsplit("/", 1)[-1]:
        sys.exit(124)
    if field == "current_errors":
        if cfg.get("empty_errors"):
            sys.exit(0)
        print("joint_reflex: " + ("true" if cfg.get("robot_error") == side else "false"))
    else:
        assert field == "header"
        print("stamp:\n  sec: 123\n  nanosec: 0\nframe_id: base")
elif args[:2] == ["action", "info"]:
    counter = root / f"{side}.action_queries"
    count = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(count))
    print("Action clients: 0")
    servers = cfg.get("action_servers", 1) if count >= cfg.get("action_ready_after", 1) else 0
    print(f"Action servers: {servers}")
elif args[:3] == ["launch", "franka_fr3_arm_controllers", "franka.launch.py"]:
    side = next(arg.split(":=")[1] for arg in args if arg.startswith("namespace:="))
    (root / f"{side}.pid").write_text(str(os.getpid()))
    if cfg.get("driver_fail") == side:
        print("FCI connection refused", file=sys.stderr)
        sys.exit(1)
    (root / f"{side}.ready").touch()
    time.sleep(15)
elif args[:3] == ["launch", "franka_duo_ptp_step", "duo_ptp_episode.launch.py"]:
    # A PTP request must never precede both arms' readiness or deactivation.
    for arm in ("left", "right"):
        assert (root / f"{arm}.ready").exists()
        impedance = root / f"{arm}.impedance"
        assert not impedance.exists() or impedance.read_text() == "inactive"
    assert "execute:=true" in args and "confirm:=true" in args
    if not cfg.get("missing_ptp_result"):
        print("PTP results: left_code=4 left_status=2 right_code=4 right_status=2")
    sys.exit(cfg.get("ptp_exit", 0))
else:
    raise AssertionError(args)
"""


@pytest.fixture
def robot_script(tmp_path):
    root = tmp_path / "robot"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
    (root / "site/install").mkdir(parents=True)
    (root / "site/install/setup.bash").write_text("true\n")
    (root / "tmr_env.sh").write_text("true\n")
    (root / "outputs").mkdir()
    (root / "configs").mkdir()
    (root / "configs/ptp_home_target.json").write_text("{}")
    bindir = root / "bin"
    bindir.mkdir()
    mock = bindir / "mock"
    mock.write_text(f"#!{sys.executable}\n{MOCK_TOOL}")
    mock.chmod(0o755)
    for name in ("ros2", "ps", "timeout", "setsid", "flock"):
        (bindir / name).symlink_to(mock)

    def run(*, ready=(), active=(), **config):
        (root / "mock.json").write_text(json.dumps(config))
        for side in ready:
            (root / f"{side}.ready").touch()
        for side in active:
            (root / f"{side}.impedance").write_text("active")
        result = subprocess.run(
            ["bash", str(root / "scripts" / SCRIPT.name)],
            env={
                **os.environ,
                "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
                "TMR_ENV_FILE": str(root / "tmr_env.sh"),
                "PTP_DRIVER_TIMEOUT_S": "2",
                "PTP_MOCK_ROOT": str(root),
            },
            text=True,
            capture_output=True,
            timeout=20,
        )
        assert (root / "calls.jsonl").exists(), result.stdout + result.stderr
        calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
        return result, calls

    yield run
    for pid_file in root.glob("*.pid"):
        with suppress(ProcessLookupError):
            os.kill(int(pid_file.read_text()), signal.SIGTERM)


def driver_launches(calls):
    return [c for c in calls if c[:2] == ["launch", "franka_fr3_arm_controllers"]]


def ptp_launches(calls):
    return [c for c in calls if c[:2] == ["launch", "franka_duo_ptp_step"]]


@pytest.mark.parametrize("ready", [(), ("left",), ("right",), ("left", "right")])
def test_only_missing_arm_drivers_start(robot_script, ready):
    result, calls = robot_script(ready=ready, active=ready)
    assert result.returncode == 0, result.stdout + result.stderr
    launches = driver_launches(calls)
    assert len(launches) == 2 - len(ready)
    for side, ip in (("left", "172.16.16.12"), ("right", "172.16.16.11")):
        matching = [c for c in launches if f"namespace:={side}" in c]
        assert len(matching) == (side not in ready)
        if matching:
            assert f"robot_ip:={ip}" in matching[0]
            assert "load_gripper:=false" in matching[0]
            assert "use_fake_hardware:=false" in matching[0]
    assert len(ptp_launches(calls)) == 1
    assert "PTP OK" in result.stdout


@pytest.mark.parametrize(
    "process",
    [
        "/opt/ros/lib/controller_manager/ros2_control_node --ros-args -r __ns:=/left",
        "/usr/bin/python3 /opt/ros/jazzy/bin/ros2 launch franka_fr3_arm_controllers franka.launch.py namespace:=left",
        "/opt/ros/jazzy/bin/ros2 launch franka_fr3_arm_controllers franka.launch.py robot_ip:=172.16.16.12",
    ],
)
def test_existing_starting_driver_is_reused(robot_script, process):
    result, calls = robot_script(ready=("right",), processes=[process], ready_after={"left": 3})
    assert result.returncode == 0, result.stdout + result.stderr
    assert not driver_launches(calls)
    assert len(ptp_launches(calls)) == 1


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"driver_fail": "left"}, "left driver exited"),
        (
            {"processes": ["/opt/ros/lib/controller_manager/ros2_control_node -r __ns:=/left"]},
            "left controller manager/state broadcasters not ready",
        ),
    ],
)
def test_unavailable_driver_aborts_before_motion(robot_script, config, message):
    result, calls = robot_script(ready=("right",), **config)
    assert result.returncode != 0
    assert message in result.stdout
    assert not ptp_launches(calls)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"missing_topic": "robot_state"}, "robot_state unavailable"),
        ({"missing_topic": "measured_joint_states"}, "measured_joint_states unavailable"),
        ({"missing_topic": "current_pose"}, "current_pose unavailable"),
        ({"empty_errors": True}, "current_errors could not be read"),
        ({"robot_error": "left"}, "current robot errors"),
        ({"action_servers": 0}, "exactly one PTP action server"),
        ({"action_servers": 2}, "exactly one PTP action server"),
        ({"deactivate_fail": "left"}, "could not deactivate left"),
        ({"impedance_stuck": "left"}, "still has active controllers"),
        ({"other_active": "right"}, "still has active controllers"),
    ],
)
def test_failed_preflight_never_sends_ptp(robot_script, config, message):
    result, calls = robot_script(ready=("left", "right"), active=("left", "right"), **config)
    assert result.returncode != 0
    assert message in result.stdout
    assert not driver_launches(calls)
    assert not ptp_launches(calls)


@pytest.mark.parametrize("config", [{"ptp_exit": 124}, {"missing_ptp_result": True}])
def test_ptp_failure_is_reported_as_failure(robot_script, config):
    result, calls = robot_script(ready=("left", "right"), **config)
    assert result.returncode != 0
    assert len(ptp_launches(calls)) == 1
    assert "PTP OK" not in result.stdout
    assert "ABORT:" in result.stdout


def test_waits_for_ptp_server_discovery(robot_script):
    result, calls = robot_script(ready=("left", "right"), action_ready_after=2)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(ptp_launches(calls)) == 1
    assert len([c for c in calls if c[:2] == ["action", "info"]]) == 4
