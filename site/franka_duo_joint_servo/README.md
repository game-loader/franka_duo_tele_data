# Franka Duo Joint Servo

Joint-space execution of 20D Cartesian action chunks on the Franka Duo Mobile
through the site's existing `joint_impedance_controller` (effort interface).

```text
20D chunk, absolute step index in layout.data_offset, 30 Hz rows
  -> policy_chunk_joint_servo
       MoveIt KDL IK per row (link8 tip, 0.174 m tool offset, seeded from the
       previously planned step, consistency-limited)
       absolute-step JointTimeline: new chunks replace only steps at or after
       ceil(now) + commit_lead_steps and blend into the old plan
       1 kHz per-arm Ruckig position tracker; state never reset
  -> /franka_duo/joint_servo/{left,right}/target   sensor_msgs/JointState
  -> gello_target_relay (enable_robot gate, joint names, freshness)
  -> /{left,right}/gello/joint_states
  -> franka_fr3_arm_controllers/JointImpedanceController, 1 kHz effort
```

Why this path: Franka's Cartesian and joint-position motion generators check
joint velocity/acceleration continuity every cycle and trigger a reflex on a
single bad sample. The site impedance controller only sees a joint target and
produces torque; the arm follows a jerk-limited command, and a late sample on
the non-real-time kernel does not abort the motion. The dataset itself was
recorded under this controller.

Site controller contract (read from the overlay source on 2026-09-05):

- It copies the first seven `position` values of `gello/joint_states` by
  index every cycle; the servo publishes joint1..joint7 in order.
- It calls `rclcpp::shutdown()` when no target arrives, or the stamp is older
  than 0.5 s. The servo therefore streams targets from startup, following the
  measured joints until the first chunk, and keeps streaming the held command
  after the timeline ends or on a fault.
- On activation it moves to the first received target with its internal
  motion generator (speed factor 0.2), so activating it while the servo is
  idle holds the current pose.

Safety gates: the servo itself never publishes to a controller topic. The
relay drops everything unless `enable_robot:=true`; grippers additionally
need `enable_gripper:=true` on both the servo and the relay. The servo enters
a permanent hold on stale joint feedback, on a tracking error above
`max_tracking_error_rad`, or when Ruckig rejects a target; it keeps
publishing the held command so the driver does not shut down. Restart the
node to clear a fault.

## Build on the robot host

```bash
cd /home/aup/franka_duo_tele_data_infer_20260905
source ~/tmr_env.sh
colcon build --base-paths site --build-base site/build --install-base site/install \
  --merge-install --packages-select franka_duo_joint_servo --cmake-args -DBUILD_TESTING=ON
colcon test --base-paths site --build-base site/build --install-base site/install \
  --merge-install --packages-select franka_duo_joint_servo
source site/install/setup.bash
```

## Start order for the recorded-trajectory test

1. Both arm drivers up with only the broadcasters active
   (`franka.launch.py` per arm, see the status document).
2. PTP to the first action of the selected episode with the existing tool;
   wait for both `TARGET_REACHED`.
3. `ros2 launch franka_duo_joint_servo joint_servo.launch.py playback_speed:=0.1`
   (dry: publishes to site topics only). Confirm it logs `Idle`.
4. `ros2 launch franka_duo_joint_servo gello_target_relay.launch.py enable_robot:=true`
   and confirm `/left/gello/joint_states` has exactly one publisher.
5. Spawn and activate `joint_impedance_controller` on both managers. It moves
   to the servo's held target, which equals the measured pose.
6. `bash scripts/run_rgb20d_replay.sh --dataset ... --live --recorded-actions --joint-servo \
   --episode 0 --max-steps 60 --speed 0.1 --publish --enable-robot`

The Python side publishes each chunk with `layout.data_offset` set to the
absolute dataset step of row zero and paces chunk requests from the servo's
status topic. `--speed` must equal `playback_speed`.

## Continuous SmolVLA inference

```bash
bash scripts/run_smolvla_loop.sh 20 0.3 400
```

The defaults are 1000 chunks, playback speed 0.3 and a 400 ms request lead.
Both client and servo retain the dataset's 30 Hz timeline and use speed 0.3,
so actions advance at 9 Hz. The wrapper reuses a matching servo, or waits for
the accepted plan to hold and safely reconfigures it before starting the
client. A second policy publisher prevents reconfiguration.

`smolvla_stream` requests the next chunk when `(last_step - step) / 9` is at
most 0.4 seconds. It keeps one WebSocket connection and sends full-resolution
JPEG images at quality 95 without chroma subsampling; `--image-format png`
is available on the Python entry point. It waits for the servo to acknowledge
each chunk before scheduling another request.

Returned rows keep an absolute start time anchored to their input observation.
The servo drops any prefix that expired during inference or IK, preserves the
old plan's position and velocity at the commit boundary, and blends toward the
new rows with a quintic weight over four action intervals (about 444 ms at
9 Hz). If the previous plan has ended, its held endpoint provides the start of
the same blend. The 1 kHz Ruckig trackers retain their state and joint limits.
The streaming launcher uses `commit_lead_steps=0`: after IK finishes, commit
at the next integer step. Freezing the splice position and velocity preserves
the entire preceding spline segment, without adding another full-step delay.
An inference that exceeds the available lead can still cause a hold; the
400 ms trigger is not a guarantee of uninterrupted motion.

Per-run `chunks.jsonl` records request timing, remaining milliseconds, raw
model results, accepted steps and discarded prefixes. Ctrl-C stops further
requests; the already accepted plan finishes and the servo keeps holding.
