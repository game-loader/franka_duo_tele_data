"""Preflight a whole Labs chunk with continuous, jerk-limited joint tracking.

Reference rows keep their 30 Hz timestamps. As in the existing site's
JointRuckigTracker, retargeting carries position, velocity and acceleration
forward instead of resetting at each row. No robot interfaces live here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackedPlan:
    joints: np.ndarray  # reference anchors, including the initial commanded hold
    grippers: np.ndarray
    durations: np.ndarray  # reference row intervals, NOT tracker settling time
    segments: list
    period: float
    diagnostics: dict
    preserve_grippers: bool = False

    @property
    def duration(self):
        return len(self.segments) * self.period

    def kinematics(self, elapsed):
        if elapsed >= self.duration:
            return self.joints[-1].copy(), np.zeros(14), np.zeros(14)
        elapsed = max(0.0, elapsed)
        index = min(int(elapsed / self.period), len(self.segments) - 1)
        return tuple(np.asarray(v) for v in self.segments[index].at_time(elapsed - index * self.period))

    def sample(self, elapsed):
        index = min(max(0, int(elapsed / self.durations[0])), len(self.grippers) - 1)
        return self.kinematics(elapsed)[0], self.grippers[index], elapsed >= self.duration


def track_chunk(
    joints,
    grippers,
    row_period,
    bounds,
    *,
    period=0.01,
    max_duration=90.0,
    max_reference_lag=0.15,
    max_velocity=0.5,
    max_acceleration=1.0,
    max_jerk=10.0,
    target_velocity_weight=0.0,
):
    """Calculate and validate all command segments before publishing any target."""
    from ruckig import InputParameter, Result, Ruckig, Synchronization, Trajectory
    from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

    joints = np.asarray(joints, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    grippers = np.asarray(grippers, dtype=float)
    if (
        joints.ndim != 2
        or joints.shape[1] != 14
        or len(joints) < 2
        or bounds.shape != (14, 2)
        or grippers.shape != (len(joints) - 1, 2)
        or not np.isfinite(joints).all()
        or not np.isfinite(bounds).all()
        or not np.isfinite(grippers).all()
        or not np.isfinite(row_period)
        or row_period <= 0
        or not np.isfinite(period)
        or period <= 0
        or not np.isfinite(max_duration)
        or max_duration <= 0
        or not np.isfinite(max_reference_lag)
        or max_reference_lag <= 0
        or not np.isfinite(target_velocity_weight)
        or not 0 <= target_velocity_weight <= 1
        or any(not np.isfinite(v) or v <= 0 for v in (max_velocity, max_acceleration, max_jerk))
        or np.any(joints < bounds[:, 0])
        or np.any(joints > bounds[:, 1])
    ):
        raise ValueError("Invalid continuous joint tracking input")
    times = np.arange(len(joints)) * row_period
    # Shape-preserving tangents retain every reference row without spline
    # overshoot. Only the complete chunk's endpoints have forced zero velocity.
    slopes = PchipInterpolator(times, joints).derivative()(times)
    slopes[0] = slopes[-1] = 0
    reference = CubicHermiteSpline(times, joints, slopes)
    inp = InputParameter(14)
    inp.synchronization = Synchronization.No
    inp.current_position = joints[0].tolist()
    inp.current_velocity = [0.0] * 14
    inp.current_acceleration = [0.0] * 14
    inp.max_velocity = [max_velocity] * 14
    inp.max_acceleration = [max_acceleration] * 14
    inp.max_jerk = [max_jerk] * 14
    inp.target_acceleration = [0.0] * 14
    tracker = Ruckig(14)
    segments = []
    peak_v = peak_a = peak_lag = 0.0
    # Checking each 1 ms plus a Lipschitz margin covers inter-sample positions
    # at the maximum commanded velocity, including a turning point.
    check_times = np.linspace(0, period, max(2, int(np.ceil(period / 0.001)) + 1))
    margin = max_velocity * (check_times[1] - check_times[0]) / 2
    for step in range(int(max_duration / period)):
        elapsed = (step + 1) * period
        target_time = min(elapsed, times[-1])
        inp.target_position = reference(target_time).tolist()
        # Zero endpoint velocity avoids chasing a moving position AND its
        # tangent. Retarget every period while carrying current q/dq/ddq below;
        # this is not a stop-and-wait at each reference row.
        inp.target_velocity = np.clip(
            target_velocity_weight * reference(target_time, 1), -0.98 * max_velocity, 0.98 * max_velocity
        ).tolist()
        segment = Trajectory(14)
        result = tracker.calculate(inp, segment)
        if result not in (Result.Working, Result.Finished):
            raise ValueError(f"Continuous joint tracker rejected row reference: {result}")
        samples = np.asarray([segment.at_time(t) for t in check_times])
        if (
            not np.isfinite(samples).all()
            or np.any(samples[:, 0, :] < bounds[:, 0] + margin)
            or np.any(samples[:, 0, :] > bounds[:, 1] - margin)
        ):
            raise ValueError("Continuous tracked trajectory exceeds joint bounds")
        peak_v = max(peak_v, float(np.abs(samples[:, 1]).max()))
        peak_a = max(peak_a, float(np.abs(samples[:, 2]).max()))
        if peak_v > max_velocity + 1e-6 or peak_a > max_acceleration + 1e-6:
            raise ValueError("Continuous tracked trajectory exceeds dynamic limits")
        position, velocity, acceleration = segment.at_time(period)
        peak_lag = max(peak_lag, float(np.max(np.abs(np.asarray(position) - reference(target_time)))))
        if peak_lag > max_reference_lag:
            raise ValueError(
                f"Continuous tracker reference lag {peak_lag:.4f} exceeds {max_reference_lag} rad"
            )
        segments.append(segment)
        inp.current_position = position
        inp.current_velocity = velocity
        inp.current_acceleration = acceleration
        if (
            elapsed >= times[-1]
            and np.max(np.abs(np.asarray(position) - joints[-1])) < 1e-9
            and np.max(np.abs(velocity)) < 1e-8
            and np.max(np.abs(acceleration)) < 1e-7
        ):
            return TrackedPlan(
                joints,
                grippers,
                np.full(len(grippers), row_period),
                segments,
                period,
                {
                    "tracking": "continuous_ruckig_v2",
                    "target_velocity_weight": float(target_velocity_weight),
                    "reference_duration_s": float(times[-1]),
                    "command_duration_s": len(segments) * period,
                    "max_velocity_rad_s": peak_v,
                    "max_acceleration_rad_s2": peak_a,
                    "max_jerk_rad_s3": max_jerk,
                    "velocity_limit_rad_s": max_velocity,
                    "acceleration_limit_rad_s2": max_acceleration,
                    "max_reference_lag_rad": peak_lag,
                    "reference_lag_limit_rad": max_reference_lag,
                },
            )
    raise ValueError("Continuous tracker cannot settle within 90 s")
