"""Coordinate the start-to-table cup/bowl mission across the base and arm hosts.

The base route is one persistent process on the navigation host; the arm work
runs locally.  The two ROS graphs are deliberately never mixed -- the base host
runs Humble on its own domain, this host runs Jazzy -- so the base is driven
over SSH and judged only by its structured report.

Phase order::

    CREATED -> INITIALIZING_ARMS -> INITIALIZING_SPINE (travel height)
      -> READY_TO_DEPART -> OUTBOUND_BASE_RUNNING -> AT_PICKUP_TABLE
      -> CUP_STAGE_RUNNING -> CUP_DONE -> BOWL_STAGE_RUNNING -> OBJECTS_HELD
      -> RAISING_SPINE_FOR_LETTER -> POST_GRASP_ROUTE_RUNNING
      -> LOWERING_SPINE_AT_LETTER -> AT_LETTER_TABLE
      -> TEST_PLACE_RUNNING -> TEST_PLACE_DONE
      -> RAISING_SPINE_FOR_RETURN -> RETURN_ROUTE_RUNNING
      -> AT_PICKUP_AFTER_RETURN -> PLACEMENT_ROUTE_RUNNING
      -> LOWERING_SPINE_AT_PLACEMENT -> AT_PLACEMENT_TABLE
      -> FINAL_PLACE_RUNNING -> COMPLETE

The spine rises to travel height before every drive and returns to the
calibrated grasp height on arrival.  Both arms hang off its carriage, so this
is what keeps them inside the navigation footprint while moving and at the
height the camera was calibrated for while working.

Each table stage lowers the spine, clears the camera view, detects and grasps
(see ``table_grasp_stage``).  The objects are then carried to the letter side,
set down there without being given up (``--mode test``: touch, open, close,
lift), carried back, and finally left on the step-20 placement table
(``--mode final``).  See ``table_place_stage``.

A checkpoint records the phase so an interrupted run resumes instead of
silently replaying the drive.  Without ``--execute`` this only prints the
strategy and starts nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .spine_client import GRASP_HEIGHT_M, TRAVEL_HEIGHT_M, report_is_stable

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class Phase(str, Enum):
    CREATED = "CREATED"
    INITIALIZING_ARMS = "INITIALIZING_ARMS"
    INITIALIZING_SPINE = "INITIALIZING_SPINE"
    READY_TO_DEPART = "READY_TO_DEPART"
    OUTBOUND_BASE_RUNNING = "OUTBOUND_BASE_RUNNING"
    AT_PICKUP_TABLE = "AT_PICKUP_TABLE"
    CUP_STAGE_RUNNING = "CUP_STAGE_RUNNING"
    CUP_DONE = "CUP_DONE"
    BOWL_STAGE_RUNNING = "BOWL_STAGE_RUNNING"
    OBJECTS_HELD = "OBJECTS_HELD"
    # Carry the objects to the letter side, set them down there without giving
    # them up, drive back, then leave them on the step-20 placement table.
    # The spine rises to travel height before every drive and returns to the
    # calibrated grasp height on arrival: the arms ride its carriage.
    RAISING_SPINE_FOR_LETTER = "RAISING_SPINE_FOR_LETTER"
    POST_GRASP_ROUTE_RUNNING = "POST_GRASP_ROUTE_RUNNING"
    LOWERING_SPINE_AT_LETTER = "LOWERING_SPINE_AT_LETTER"
    AT_LETTER_TABLE = "AT_LETTER_TABLE"
    TEST_PLACE_RUNNING = "TEST_PLACE_RUNNING"
    TEST_PLACE_DONE = "TEST_PLACE_DONE"
    RAISING_SPINE_FOR_RETURN = "RAISING_SPINE_FOR_RETURN"
    RETURN_ROUTE_RUNNING = "RETURN_ROUTE_RUNNING"
    AT_PICKUP_AFTER_RETURN = "AT_PICKUP_AFTER_RETURN"
    PLACEMENT_ROUTE_RUNNING = "PLACEMENT_ROUTE_RUNNING"
    LOWERING_SPINE_AT_PLACEMENT = "LOWERING_SPINE_AT_PLACEMENT"
    AT_PLACEMENT_TABLE = "AT_PLACEMENT_TABLE"
    FINAL_PLACE_RUNNING = "FINAL_PLACE_RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


# Phases after which the base has already driven; resuming past one of these
# must never re-run the route from the marked start.
DEPARTED_PHASES = frozenset(
    {
        Phase.AT_PICKUP_TABLE,
        Phase.CUP_STAGE_RUNNING,
        Phase.CUP_DONE,
        Phase.BOWL_STAGE_RUNNING,
        Phase.OBJECTS_HELD,
        Phase.RAISING_SPINE_FOR_LETTER,
        Phase.POST_GRASP_ROUTE_RUNNING,
        Phase.LOWERING_SPINE_AT_LETTER,
        Phase.AT_LETTER_TABLE,
        Phase.TEST_PLACE_RUNNING,
        Phase.TEST_PLACE_DONE,
        Phase.RAISING_SPINE_FOR_RETURN,
        Phase.RETURN_ROUTE_RUNNING,
        Phase.AT_PICKUP_AFTER_RETURN,
        Phase.PLACEMENT_ROUTE_RUNNING,
        Phase.LOWERING_SPINE_AT_PLACEMENT,
        Phase.AT_PLACEMENT_TABLE,
        Phase.FINAL_PLACE_RUNNING,
        Phase.COMPLETE,
    }
)


class MissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str
    elapsed_s: float


@dataclass(frozen=True)
class MissionConfig:
    base_host: str
    base_root: str
    arm_root: str
    arm_env: str
    dataset: str
    speed: float
    init_timeout_s: float
    outbound_timeout_s: float
    stage_timeout_s: float
    transition_settle_s: float


def emit(event: str, **values) -> None:
    print(
        "TABLE_MISSION=" + json.dumps({"event": event, **values}, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


# --------------------------------------------------------------------------- checkpoint


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "phase" not in value:
        raise MissionError(f"invalid mission checkpoint: {path}")
    return value


class MissionRunLock:
    """Cross-process non-blocking lock held for the whole mission run."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        import fcntl

        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise MissionError(f"another mission run holds {self.path}") from exc
        return self

    def __exit__(self, *exc_info):
        if self.handle is not None:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
        return False


# --------------------------------------------------------------------------- report parsing


def extract_last_json_object(text: str) -> dict | None:
    """Last JSON object in a stream, preferring a mission-shaped report."""
    clean = ANSI_ESCAPE.sub("", text)
    decoder = json.JSONDecoder()
    best = None
    best_span = -1
    mission_report = None
    for index, char in enumerate(clean):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(clean[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            if end > best_span:
                best, best_span = value, end
            if "status" in value and "final_state" in value:
                mission_report = value
    return mission_report if mission_report is not None else best


def base_report_is_stable(report: dict | None) -> bool:
    """True only when the base proved it stopped and latched zero velocity."""
    if not isinstance(report, dict):
        return False
    stationary = report.get("final_stationary")
    return bool(
        report.get("status") == "success"
        and report.get("final_state") == "FINAL_STOP"
        and isinstance(stationary, dict)
        and stationary.get("confirmed") is True
        and report.get("zero_command_latched") is True
    )


def route_report_is_stable(report: dict | None) -> bool:
    """True when an odometry-closed-loop base leg (script 13) finished and latched zero.

    Script 13 writes a plain ``status``/``zero_command_latched`` state; it has no
    doorway report, so the outbound checks do not apply to it.
    """
    if not isinstance(report, dict):
        return False
    return bool(
        report.get("status") == "complete" and report.get("zero_command_latched") is True
    )


def return_report_is_stable(report: dict | None) -> bool:
    """True when script 15 proved a stationary COMPLETE at the pickup side.

    Script 15 nests its doorway result under ``door_report``, so the flat
    outbound shape checked by :func:`base_report_is_stable` does not match it.
    This mirrors the acceptance test script 20 applies to the same file.
    """
    if not isinstance(report, dict):
        return False
    door = report.get("door_report")
    if not isinstance(door, dict):
        return False
    stationary = door.get("final_stationary")
    return bool(
        report.get("status") == "complete"
        and report.get("phase") == "COMPLETE"
        and report.get("zero_command_latched") is True
        and door.get("status") == "success"
        and door.get("final_state") == "FINAL_STOP"
        and door.get("zero_command_latched") is True
        and isinstance(stationary, dict)
        and stationary.get("confirmed") is True
    )


def placement_route_is_waiting(report: dict | None) -> bool:
    """True when step 20 stopped at the placement table and is awaiting the arm."""
    if not isinstance(report, dict):
        return False
    return bool(
        report.get("phase") == "WAITING_FOR_PLACEMENT"
        and report.get("zero_command_latched") is True
        and report.get("placement_completed") is False
    )


def place_report_is_stable(report: dict | None, mode: str) -> bool:
    """True when a placement finished, and ended holding what the mode requires.

    A ``test`` placement must come away still carrying the object; a ``final``
    placement must have released it.
    """
    if not isinstance(report, dict) or report.get("status") != "success":
        return False
    if not isinstance(report.get("place"), dict):
        return False
    return report.get("regrasp") is (mode == "test")


def stage_report_is_stable(report: dict | None) -> bool:
    """True when a table stage reached its grasp and left a success report.

    ``camera_clear`` and ``grasp_ready`` may be reported as skipped while they
    are still untaught, so only the grasp itself is required.
    """
    if not isinstance(report, dict) or report.get("status") != "success":
        return False
    names = {entry.get("stage") for entry in report.get("stages", []) if isinstance(entry, dict)}
    return {"camera_clear", "grasp_ready"} <= names and isinstance(report.get("grasp"), dict)


# --------------------------------------------------------------------------- command running


def _base_environment() -> str:
    return "\n".join(
        [
            "source /opt/ros/humble/setup.bash",
            'source "${HOME}/ros2_ws/install/setup.bash"',
            'source "${HOME}/tmr_navigation/install/setup.bash"',
            'source "${HOME}/tmr_navigation/install/tmr_local_navigation/share/'
            'tmr_local_navigation/local_setup.bash"',
            "export ROS_DOMAIN_ID=${TMR_CYCLE_ROS_DOMAIN_ID:-97}",
            "export ROS_LOCALHOST_ONLY=0",
            "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
            'if [[ -f "${HOME}/cyclonedds.xml" ]]; then',
            '  export CYCLONEDDS_URI="file://${HOME}/cyclonedds.xml"',
            "fi",
            "export PYTHONUNBUFFERED=1",
        ]
    )


def build_remote_base_shell(config: MissionConfig, run_id: str, mission: str | None = None) -> str:
    """Remote shell that runs one base leg as a supervised child under a lock.

    The lock is shared by every leg, so two routes can never own the velocity
    channel at once.
    """
    root = shlex.quote(config.base_root)
    pid_file = shlex.quote(f"/tmp/tmr_table_mission_base_{run_id}.pid")
    if mission is None:
        mission = (
            f"python3 {root}/scripts/07_start_to_pickup.py "
            f"--config {root}/config/start_to_pickup.yaml --execute --disable-collision-guard"
        )
    return "\n".join(
        [
            # ROS setup files probe unset variables; enable nounset after them.
            "set -eo pipefail",
            _base_environment(),
            "set -u",
            f"cd {root}",
            "command -v flock >/dev/null 2>&1 || { echo 'base lock utility unavailable' >&2; exit 72; }",
            "exec 9>/tmp/tmr_table_mission_base.lock",
            "flock -n 9 || { echo 'another base mission is already running' >&2; exit 73; }",
            "child=''",
            f"pid_file={pid_file}",
            "stop_child() {",
            '  if [[ -n "${child}" ]] && kill -0 "${child}" 2>/dev/null; then',
            '    kill -INT "${child}" 2>/dev/null || true',
            '    for _ in {1..30}; do kill -0 "${child}" 2>/dev/null || return 0; sleep 0.1; done',
            '    kill -TERM "${child}" 2>/dev/null || true',
            "  fi",
            "}",
            "trap 'stop_child' HUP INT TERM",
            f"{mission} &",
            "child=$!",
            "printf '%s\\n' \"${child}\" >\"${pid_file}\"",
            "set +e",
            'wait "${child}"',
            "rc=$?",
            "set -e",
            'rm -f "${pid_file}"',
            "trap - HUP INT TERM",
            'exit "${rc}"',
        ]
    )


def build_base_argv(config: MissionConfig, run_id: str, mission: str | None = None) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "ServerAliveInterval=2",
        "-o",
        "ServerAliveCountMax=3",
        config.base_host,
        "bash -lc " + shlex.quote(build_remote_base_shell(config, run_id, mission)),
    ]


# The right-shift distance script 13 travels to the letter side. Script 15 needs
# it to come back by the same amount.
LETTER_SIDE_RIGHT_M = 0.80 + 0.85


def build_post_grasp_argv(config: MissionConfig, run_id: str) -> list[str]:
    """Base leg carrying the held objects from the table to the letter side."""
    root = shlex.quote(config.base_root)
    mission = (
        f"python3 {root}/scripts/13_post_grasp_route.py --execute --fresh-start "
        f"--state-file /tmp/tmr_table_mission_post_grasp_{run_id}.json"
    )
    return build_base_argv(config, run_id, mission)


def build_return_from_letter_argv(config: MissionConfig, run_id: str) -> list[str]:
    """Base leg returning from the letter side to the pickup-table side."""
    root = shlex.quote(config.base_root)
    mission = (
        f"python3 {root}/scripts/15_return_from_letter.py --execute --fresh-start "
        f"--left-m {LETTER_SIDE_RIGHT_M:.2f} --disable-collision-guard "
        f"--state-file /tmp/tmr_table_mission_return_{run_id}.json"
    )
    return build_base_argv(config, run_id, mission)


def build_placement_route_argv(config: MissionConfig, run_id: str) -> list[str]:
    """Step 20: left 0.85 m, align with the far table leg, turn, await placement.

    The leg ROI is referenced to the pose the base is standing in when the
    detour starts, so no saved START capture is needed.
    """
    root = shlex.quote(config.base_root)
    mission = (
        f"python3 {root}/scripts/20_after_return_placement.py outbound --execute "
        f"--after15-state /tmp/tmr_table_mission_return_{run_id}.json "
        f"--state-file /tmp/tmr_table_mission_placement_{run_id}.json"
    )
    return build_base_argv(config, run_id, mission)


def build_place_argv(
    config: MissionConfig, mode: str, target: str, arm: str, table_z: float, run_id: str
) -> list[str]:
    """Arm-local placement: `test` touches and carries on, `final` leaves it."""
    arguments = " ".join(
        shlex.quote(value)
        for value in [
            "--dataset",
            config.dataset,
            "--mode",
            mode,
            "--arm",
            arm,
            "--target",
            target,
            "--table-z",
            f"{table_z:.4f}",
            "--speed",
            f"{config.speed:g}",
            "--publish",
            "--enable-robot",
            "--output",
            f"outputs/table_place_{mode}_{target}_{run_id}.json",
        ]
    )
    return _arm_python_command(config, f"franka_duo_tele_data.table_place_stage {arguments}")


def _arm_python_command(config: MissionConfig, module_arguments: str) -> list[str]:
    """Run an arm-host module through the repo venv with ROS overlays sourced.

    PYTHONPATH is appended, never replaced: overwriting it hides the host's ROS
    packages from the interpreter.
    """
    root = shlex.quote(config.arm_root)
    environment = shlex.quote(config.arm_env)
    command = (
        f"set -eo pipefail; source {environment}; source site/install/setup.bash; "
        f"export PYTHONUNBUFFERED=1; cd {root}; "
        f'export PYTHONPATH="$PWD/src${{PYTHONPATH:+:$PYTHONPATH}}"; '
        f"exec .venv/bin/python -m {module_arguments}"
    )
    return ["bash", "-lc", command]


def build_stow_argv(config: MissionConfig) -> list[str]:
    """Fold both arms into the navigation footprint over the impedance path."""
    arguments = " ".join(
        shlex.quote(value)
        for value in [
            "--dataset",
            config.dataset,
            "--pose-only",
            "travel_stow",
            "--speed",
            f"{config.speed:g}",
            "--publish",
            "--enable-robot",
            "--output",
            "outputs/table_stow.json",
        ]
    )
    return _arm_python_command(config, f"franka_duo_tele_data.table_grasp_stage {arguments}")


def build_stage_argv(config: MissionConfig, target: str, arm: str, run_id: str) -> list[str]:
    """Local table stage: spine down, clear view, detect, pose up, grasp."""
    arguments = " ".join(
        shlex.quote(value)
        for value in [
            "--dataset",
            config.dataset,
            "--target",
            target,
            "--arm",
            arm,
            "--speed",
            f"{config.speed:g}",
            "--spine-target-m",
            f"{GRASP_HEIGHT_M:.3f}",
            "--publish",
            "--enable-robot",
            "--output",
            f"outputs/table_stage_{target}_{run_id}.json",
        ]
    )
    return _arm_python_command(config, f"franka_duo_tele_data.table_grasp_stage {arguments}")


def build_spine_argv(config: MissionConfig, target_m: float) -> list[str]:
    return _arm_python_command(
        config, f"franka_duo_tele_data.spine_client --target-m {target_m:.3f} --execute"
    )


def _terminate(process: subprocess.Popen, grace_s: float = 5.0) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=grace_s)


def run_streamed_command(
    label: str, argv: list[str], timeout_s: float, log_path: Path
) -> CommandResult:
    """Run a child, stream and log its output, and stop it on timeout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    output: list[str] = []
    reader_done = False
    try:
        with log_path.open("w", encoding="utf-8") as log:
            while not reader_done or process.poll() is None:
                if time.monotonic() - started > timeout_s:
                    _terminate(process)
                    raise MissionError(f"{label} exceeded {timeout_s:.0f}s")
                try:
                    line = lines.get(timeout=0.2)
                except queue.Empty:
                    continue
                if line is None:
                    reader_done = True
                    continue
                output.append(line)
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
        process.wait(timeout=10)
    except BaseException:
        _terminate(process)
        raise
    return CommandResult(int(process.returncode), "".join(output), time.monotonic() - started)


# --------------------------------------------------------------------------- strategy


def strategy(config: MissionConfig, checkpoint_path: Path, args=None) -> dict:
    """The exact plan, printed instead of executed without --execute."""
    previous = load_checkpoint(checkpoint_path)
    letter_z = getattr(args, "letter_table_z", None)
    placement_z = getattr(args, "placement_table_z", None)
    return {
        "status": "dry_run",
        "motion_enabled": False,
        "base_host": config.base_host,
        "travel_spine_m": TRAVEL_HEIGHT_M,
        "grasp_spine_m": GRASP_HEIGHT_M,
        "letter_table_z": letter_z,
        "placement_table_z": placement_z,
        "existing_checkpoint": previous,
        "steps": [
            "initialize both arms into the travel posture inside the navigation footprint",
            f"raise spine to {TRAVEL_HEIGHT_M:.3f} m and verify it before departing",
            "run the base route to the table over SSH and require FINAL_STOP",
            f"cup stage: lower spine to {GRASP_HEIGHT_M:.3f} m, clear view, detect, pose up, grasp",
            f"bowl stage: lower spine to {GRASP_HEIGHT_M:.3f} m, clear view, detect, pose up, grasp",
            f"raise spine to {TRAVEL_HEIGHT_M:.3f} m, then carry both objects to the letter side",
            f"lower spine to {GRASP_HEIGHT_M:.3f} m at the letter table",
            "test placement: touch the letter table, open, close again and lift each object",
            f"raise spine to {TRAVEL_HEIGHT_M:.3f} m, then drive back to the pickup side",
            "step-20 route: left 0.85 m, align with the far leg, turn, await placement",
            f"lower spine to {GRASP_HEIGHT_M:.3f} m at the placement table",
            "final placement: set each object down and leave it",
        ],
    }


def run_mission(config: MissionConfig, args) -> int:
    run_id = uuid.uuid4().hex[:8]
    checkpoint_path = args.checkpoint
    logs = args.log_dir / f"table_mission_{run_id}"
    state: dict = {}

    def set_phase(phase: Phase, **values) -> None:
        state.update(values)
        payload = {
            "version": 1,
            "run_id": run_id,
            "phase": phase.value,
            "updated_unix_s": time.time(),
            **state,
        }
        atomic_write_json(checkpoint_path, payload)
        emit("phase", run_id=run_id, phase=phase.value)

    def run_phase(phase: Phase, label: str, argv: list[str], timeout_s: float) -> CommandResult:
        set_phase(phase)
        return run_streamed_command(label, argv, timeout_s, logs / f"{label}.log")

    def move_spine(phase: Phase, label: str, target_m: float) -> dict:
        """Change the spine height and prove it before anything else moves.

        The arms ride the spine carriage, so this is what keeps them inside the
        navigation footprint while driving and at the calibrated height while
        working at a table.
        """
        result = run_phase(phase, label, build_spine_argv(config, target_m), config.init_timeout_s)
        report = extract_last_json_object(result.output)
        if result.returncode != 0 or not report_is_stable(report, target_m):
            raise MissionError(f"spine did not prove the {target_m:.3f} m height ({label})")
        return report or {}

    previous = load_checkpoint(checkpoint_path)
    if previous is not None and not args.fresh_start_confirmed:
        phase = previous.get("phase")
        if phase in {p.value for p in DEPARTED_PHASES}:
            raise MissionError(
                f"checkpoint {checkpoint_path} is at {phase}: the base has already driven. "
                f"Return it to the marked start and pass --fresh-start-confirmed."
            )

    try:
        set_phase(Phase.CREATED)

        # The arms must be inside the navigation footprint before the base moves;
        # they ride the spine carriage, so posture and height are one safety case.
        if args.skip_arm_init:
            emit("skipped", step="arm_stow")
        else:
            result = run_phase(
                Phase.INITIALIZING_ARMS,
                "arm_stow",
                build_stow_argv(config),
                config.init_timeout_s,
            )
            stow_report = extract_last_json_object(result.output)
            if result.returncode != 0 or not isinstance(stow_report, dict) or (
                stow_report.get("status") != "success"
            ):
                raise MissionError("arms did not reach the travel stow posture")
            state["stow_report"] = stow_report

        spine_report = move_spine(Phase.INITIALIZING_SPINE, "spine_travel", TRAVEL_HEIGHT_M)
        set_phase(Phase.READY_TO_DEPART, spine_travel_report=spine_report)

        result = run_phase(
            Phase.OUTBOUND_BASE_RUNNING,
            "outbound",
            build_base_argv(config, run_id),
            config.outbound_timeout_s,
        )
        base_report = extract_last_json_object(result.output)
        if result.returncode != 0 or not base_report_is_stable(base_report):
            raise MissionError("outbound route did not prove FINAL_STOP and latched zero speed")
        set_phase(Phase.AT_PICKUP_TABLE, base_report=base_report)
        time.sleep(max(0.0, min(0.5, config.transition_settle_s)))

        result = run_phase(
            Phase.CUP_STAGE_RUNNING,
            "cup_stage",
            build_stage_argv(config, "cup", args.cup_arm, run_id),
            config.stage_timeout_s,
        )
        cup_report = extract_last_json_object(result.output)
        if result.returncode != 0 or not stage_report_is_stable(cup_report):
            raise MissionError("cup stage did not complete detection and grasp")
        set_phase(Phase.CUP_DONE, cup_report=cup_report)

        held: list[tuple[str, str]] = [("cup", args.cup_arm)]
        if not args.cup_only:
            result = run_phase(
                Phase.BOWL_STAGE_RUNNING,
                "bowl_stage",
                build_stage_argv(config, "bowl", args.bowl_arm, run_id),
                config.stage_timeout_s,
            )
            bowl_report = extract_last_json_object(result.output)
            if result.returncode != 0 or not stage_report_is_stable(bowl_report):
                raise MissionError("bowl stage did not complete detection and grasp")
            held.append(("bowl", args.bowl_arm))
            state["bowl_report"] = bowl_report
        set_phase(Phase.OBJECTS_HELD, held=[name for name, _arm in held])

        if args.stop_after_grasp:
            emit("stopped_after_grasp", run_id=run_id, held=[name for name, _arm in held])
            set_phase(Phase.COMPLETE)
            return 0

        def place_all(phase: Phase, label: str, mode: str, table_z: float) -> list[dict]:
            """Set every held object down, one arm at a time."""
            reports = []
            for name, arm in held:
                result = run_phase(
                    phase,
                    f"{label}_{name}",
                    build_place_argv(config, mode, name, arm, table_z, run_id),
                    config.stage_timeout_s,
                )
                report = extract_last_json_object(result.output)
                if result.returncode != 0 or not place_report_is_stable(report, mode):
                    raise MissionError(f"{mode} placement of the {name} did not complete")
                reports.append({"object": name, "arm": arm, **(report or {})})
            return reports

        # Raise before driving: the arms ride the spine and must stay inside the
        # navigation footprint, and the grasp posture reaches outside it.
        move_spine(Phase.RAISING_SPINE_FOR_LETTER, "spine_travel_to_letter", TRAVEL_HEIGHT_M)

        # Carry the objects to the letter side.
        result = run_phase(
            Phase.POST_GRASP_ROUTE_RUNNING,
            "post_grasp_route",
            build_post_grasp_argv(config, run_id),
            config.outbound_timeout_s,
        )
        post_grasp_report = extract_last_json_object(result.output)
        if result.returncode != 0 or not route_report_is_stable(post_grasp_report):
            raise MissionError("post-grasp route did not finish with latched zero speed")

        # Back down to the calibrated working height before touching a table.
        move_spine(Phase.LOWERING_SPINE_AT_LETTER, "spine_grasp_at_letter", GRASP_HEIGHT_M)
        set_phase(Phase.AT_LETTER_TABLE, post_grasp_report=post_grasp_report)
        time.sleep(max(0.0, min(0.5, config.transition_settle_s)))

        # Touch the letter table, then pick the objects straight back up.
        test_reports = place_all(
            Phase.TEST_PLACE_RUNNING, "test_place", "test", args.letter_table_z
        )
        set_phase(Phase.TEST_PLACE_DONE, test_place_reports=test_reports)

        # Raise again for the drive back.
        move_spine(Phase.RAISING_SPINE_FOR_RETURN, "spine_travel_to_pickup", TRAVEL_HEIGHT_M)

        # Drive back to the pickup side.
        result = run_phase(
            Phase.RETURN_ROUTE_RUNNING,
            "return_from_letter",
            build_return_from_letter_argv(config, run_id),
            config.outbound_timeout_s,
        )
        return_report = extract_last_json_object(result.output)
        if result.returncode != 0 or not return_report_is_stable(return_report):
            raise MissionError("return route did not prove a stationary COMPLETE at the pickup side")
        set_phase(Phase.AT_PICKUP_AFTER_RETURN, return_report=return_report)

        # Step 20 is another base leg, so the spine stays raised through it.
        result = run_phase(
            Phase.PLACEMENT_ROUTE_RUNNING,
            "placement_route",
            build_placement_route_argv(config, run_id),
            config.outbound_timeout_s,
        )
        placement_report = extract_last_json_object(result.output)
        if result.returncode != 0 or not placement_route_is_waiting(placement_report):
            raise MissionError("step-20 route did not stop and wait for placement")

        # Down to working height for the last placement.
        move_spine(Phase.LOWERING_SPINE_AT_PLACEMENT, "spine_grasp_at_placement", GRASP_HEIGHT_M)
        set_phase(Phase.AT_PLACEMENT_TABLE, placement_route_report=placement_report)
        time.sleep(max(0.0, min(0.5, config.transition_settle_s)))

        # Leave the objects on the placement table.
        final_reports = place_all(
            Phase.FINAL_PLACE_RUNNING, "final_place", "final", args.placement_table_z
        )
        set_phase(Phase.COMPLETE, final_place_reports=final_reports)
        emit("complete", run_id=run_id)
        return 0
    except KeyboardInterrupt:
        set_phase(Phase.INTERRUPTED)
        emit("interrupted", run_id=run_id)
        return 130
    except MissionError as error:
        set_phase(Phase.FAILED, error=str(error))
        emit("failed", run_id=run_id, error=str(error))
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-host", default="tmr-user@172.16.0.50")
    parser.add_argument("--base-root", default="/home/tmr-user/tmr_cycle")
    parser.add_argument("--arm-root", type=Path, default=Path.cwd())
    parser.add_argument("--arm-env", default=str(Path.home() / "tmr_env.sh"))
    parser.add_argument("--dataset", default="datasets/franka_duo_lerobot_rgb20d_v1")
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--cup-arm", choices=("left", "right"), default="right")
    parser.add_argument("--bowl-arm", choices=("left", "right"), default="left")
    parser.add_argument("--cup-only", action="store_true")
    parser.add_argument("--skip-arm-init", action="store_true")
    parser.add_argument(
        "--stop-after-grasp",
        action="store_true",
        help="hold the objects at the pickup table and skip every placement leg",
    )
    parser.add_argument(
        "--letter-table-z",
        type=float,
        default=None,
        help="letter-side table top in the midpoint base frame, for the test placement; "
        "measure it at the grasp spine height (defaults to --placement-table-z)",
    )
    parser.add_argument(
        "--placement-table-z",
        type=float,
        default=-0.220,
        help="step-20 placement table top in the midpoint base frame",
    )
    parser.add_argument("--init-timeout-s", type=float, default=180.0)
    parser.add_argument("--outbound-timeout-s", type=float, default=420.0)
    parser.add_argument("--stage-timeout-s", type=float, default=300.0)
    parser.add_argument("--transition-settle-s", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/table_mission.json"))
    parser.add_argument("--log-dir", type=Path, default=Path("log/table_mission"))
    parser.add_argument("--lock", type=Path, default=Path("log/table_mission/.lock"))
    parser.add_argument("--fresh-start-confirmed", action="store_true")
    parser.add_argument("--execute", action="store_true", help="required to move anything")
    args = parser.parse_args(argv)
    if not 0 < args.speed <= 1:
        parser.error("speed must be in (0,1]")
    if args.cup_arm == args.bowl_arm:
        parser.error("cup and bowl must use different arms")
    # The letter-side table is usually the same height as the placement table;
    # override it only when it has been measured separately.
    if args.letter_table_z is None:
        args.letter_table_z = args.placement_table_z

    config = MissionConfig(
        base_host=args.base_host,
        base_root=args.base_root,
        arm_root=str(args.arm_root.resolve()),
        arm_env=args.arm_env,
        dataset=args.dataset,
        speed=args.speed,
        init_timeout_s=args.init_timeout_s,
        outbound_timeout_s=args.outbound_timeout_s,
        stage_timeout_s=args.stage_timeout_s,
        transition_settle_s=args.transition_settle_s,
    )
    if not args.execute:
        print(json.dumps(strategy(config, args.checkpoint, args), ensure_ascii=False, indent=2), flush=True)
        return 0
    try:
        with MissionRunLock(args.lock):
            return run_mission(config, args)
    except MissionError as error:
        print(json.dumps({"status": "failed", "error": str(error)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
