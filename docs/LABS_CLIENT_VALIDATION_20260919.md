# Labs FastWAM client verification — 2026-09-19

Station: `agile@10.3.8.31`. New independent installation:
`/home/agile/work/labs/data/tools/labs31_client`.

Service: `wss://release-organisation-sara-chapters.trycloudflare.com/infer`,
`fastwam.msgpack.v1`, FastWAM checkpoint step 10000 on RTX 4090.
Health advertised state34/action14, horizon32, 30 Hz and the Labs layouts/task.
The original ROS drivers/controllers were not restarted or switched. No robot
command was published during this verification.

The client acquired real measured joints/grippers and three synchronized
images, computed the saved-URDF link0/link8 state34 and resized the RGB views to
640x480. Two consecutive requests reused one WebSocket connection and each
returned finite [32,14] values.

Raw gripper regression overshoot was observed: one earlier reply ranged from
0.983148 to 1.016355 on the left and 1.019484 to 1.054240 on the right, with
server inference time 383.49 ms. The Labs client now uses the dataset's binary
0.5 decision boundary, rejects predictions outside [-0.1,1.1], and retains the
unmodified raw reply. Cartesian values are not clipped or normalized.

Per the user's explicit decision, every returned row references the request's
initial state. The server's differing temporal-description text remains in the
saved health/replies, alongside the chosen client convention.

Successful run artifacts:

```text
outputs/fastwam_live_04/trace.jsonl
outputs/fastwam_live_04/observation_000000/
outputs/fastwam_live_04/observation_000001/
outputs/fastwam_live_04/ik_validation.json
```

The host-built MoveIt/KDL solver validated the complete reconstructed chunks:

| Request | Rows | IK work | Planned duration | Largest joint step |
| --- | --- | --- | --- | --- |
| 1789798122640088875 | 32 | 118.18 ms | 4.6894 s | 0.016899 rad |
| 1789798128505304772 | 32 | 57.16 ms | 4.6172 s | 0.017149 rad |

All 128 arm IK solves passed joint bounds, continuity and FK residual checks.
Durations include the relay's per-segment velocity/acceleration limits at
default speed 0.3; they are planning estimates, not measured robot execution.
The host build resolved kinematics but lacked the `franka_description` mesh
package. No collision checking was performed or claimed.

Robot publication still requires both flags on client and relay, exclusive
ownership of the follower/gripper topics, and the expected active controllers.
Those execution prerequisites and physical motion remain unvalidated.

## Startup joint return rewrite

The initial implementation incorrectly routed return-to-start through Cartesian
IK, which cannot guarantee the dataset's recorded redundant joint posture.
It has been replaced with a separate measured-joint return command and planner.
The client requests it once outside the policy loop; the relay independently
loads episode 0/frame 0, verifies the start content identity, and targets all
14 recorded measured joints directly. Cameras and IK are not used for return.
Gripper command publication is suppressed throughout return/hold. Completion
requires 0.01 rad position tolerance and 0.02 rad/s measured velocity tolerance
continuously for 0.5 seconds after trajectory completion.

Validation on 10.3.8.31 (no robot command publishers created):

- Actual dataset: `labs_fr3_link8_delta14_20260916`, episode 0/frame 0.
- Fresh measured positions and velocities received for both arms.
- Maximum measured joint speed: 0.0015245 rad/s.
- Maximum current-to-start joint displacement: 0.6721401 rad.
- Synchronized quintic return duration: 5.04105 s, bounded to 0.25 rad/s
  and 0.5 rad/s². This is a computed plan, not an executed movement.
- Start identity:
  `32bb08eae254a6cd6bab1bc61af833ce0eee3817ad7db9ceb188935fc9886e17`.
- Updated client dry-run against the real FastWAM endpoint completed with
  32 action rows, `published=false`, `chunk_reference=request_observation`.
  Trace: `outputs/return_rewrite_dryrun_01/trace.jsonl` in the remote client
  installation. DDS emitted deserialization diagnostics during startup;
  valid observations and the inference request nevertheless completed.
- 65 focused tests passed locally, including direct joint return, target
  identity mismatch, missing/duplicate frame 0, velocity gating, settle dwell,
  failure blocking inference, gripper preservation and fixed-rate rejection.
- Changed-file Ruff, TOML parsing and all shell syntax checks passed.
  Full-repository Ruff still reports the two existing unrelated findings in
  `local_navigation/launch/dual_lidar_slam.launch.py` and
  `scripts/wall_servo_control.py`.

The new return path has not physically moved the robot. It has no collision
planner. Policy chunks use a separate fixed 30 Hz timing check; this dry-run
validates transport/conversion, not successful 30 Hz physical playback of the
model's targets. Targets exceeding configured dynamic limits remain rejected.

## Unified control script

Added `scripts/labs_control.sh` with independent `--restore` / `--infer` flags.
Combined mode returns once before inference. Default remains dry-run; actual
publication requires both existing gates. Executing modes reuse or launch a
persistent site relay without switching controllers.

Verified on 10.3.8.31 without robot output:

- `outputs/script_restore_check_01`: selected episode 0's recorded 14-joint
  target; did not connect to a model or subscribe to robot/camera streams.
- `outputs/script_infer_check_01`: real observations and server response,
  32 action rows, no restore, `published=false`.
- `outputs/script_combined_check_01`: one restore preview followed by a real
  32-row response, `published=false`.
- Both follower controllers were inactive in the live controller lists.
  The script's actual execution path requires them active and will reject
  readiness until site controller ownership is prepared.
- 69 focused tests passed; changed-file Ruff, TOML parsing and shell syntax
  checks passed. Added mode routing tests verify restore-only never opens the
  model connection, infer-only never loads a return target, combined mode
  returns once, and dry-run never starts a publishing relay.

## Explicit follower activation

At the user's request, both follower controllers were activated after adding
and testing a stationary startup hold in the site relay. The relay exclusively
publishes fixed measured joint targets and preserves grippers while arming;
readiness additionally waits for both controller `FOLLOWING` states and a
0.5 s measured settling interval. Startup DDS discovery no longer permanently
latches a fault before any target has been published. 70 focused tests passed,
plus changed-file Ruff, TOML and shell syntax checks.

A previously launched relay was found in ROS domain 66 and occupied the shared
process lock. Its process was gracefully stopped; the replacement runs in the
station's domain 100. `labs_control.sh` now defaults explicitly to domain 100;
`LABS_ROS_DOMAIN_ID` is the explicit override, preventing an unrelated shell's
inherited `ROS_DOMAIN_ID` from silently selecting another domain.

Live verification:

- Left and right `joint_follower_controller`: active.
- Left and right follower state: `FOLLOWING`.
- Relay: `ready=true`, `phase=holding`, empty fault.
- Maximum activation position change: 0.00185037 rad.
- Maximum measured error from the fixed hold target: 0.00185180 rad.
- The relay continues to publish the hold target after the activation helper
  exits. A later controller/status check confirmed the same active/holding state.
- No episode return or policy chunk was requested during this activation.
- Machine-readable result: remote `outputs/follower_activation_result.json`.

## Return tolerance adjusted to 0.05 rad

The first physical return reached the commanded endpoint but failed the old
0.01 rad position gate. A subsequent five-second read-only observation found
steady joint-7 errors of approximately 0.0301 rad (left) and 0.0272 rad (right);
all measured velocities remained below 0.0034 rad/s. The user explicitly
approved a 0.05 rad return tolerance.

Updated the return position gate to 0.05 rad. Kept velocity <=0.02 rad/s,
continuous 0.5-second settling, and the startup activation hold's separate
0.01 rad gate unchanged. 71 focused tests, changed-file Ruff, TOML parsing and
shell syntax checks passed. The faulted relay was restarted with both followers
briefly deactivated and then reactivated against a fresh stationary hold.

Physical return to episode 0 then succeeded using the unified script:
`outputs/restore_tolerance005_validation/trace.jsonl`, `restore=completed`,
`published=true`. No model inference or policy chunk was executed in this test.

## Continuous whole-chunk tracking

The user explicitly requested continuous whole-chunk tracking, allowing bounded
lag instead of forcing a start/stop at every 30 Hz row. Replaced per-row
quintic rest-to-rest motion with a shape-preserving reference and a persistent
Ruckig state carried through the whole chunk. The planner validates all IK
solutions and every tracked segment before publishing. Command limits remain
0.5 rad/s and 1 rad/s²; jerk is bounded to 10 rad/s³. Reference lag above
0.15 rad is rejected. The reference clock stays 30 Hz, followed by a settling
tail. Gripper commands follow the corresponding reference-row timeline.

A chunk begins at the previous held command, avoiding a reset to measured
joints while the impedance controller has a steady-state tracking error.
Only the complete chunk ends at zero commanded velocity/acceleration. Policy
completion now uses the authorized 0.05 rad tolerance, measured velocity
<=0.02 rad/s and a 0.5 s dwell, matching return completion. Startup arming
still uses 0.01 rad. No collision planning was added.

Validation:

- 74 focused tests passed, including C2 continuity across tracker retargets,
  nonzero speed at internal policy steps, bounded velocity/acceleration/jerk,
  lag and joint-bound rejection, exact final command, and preserving the
  prior held command across chunk starts. Ruff, TOML and shell syntax passed.
- Offline replay of the previously rejected real chunk passed: nominal
  reference 1.0667 s, tracked command 1.86 s, peak commanded velocity
  0.21791 rad/s, peak acceleration 1 rad/s², max reference lag 0.07751 rad.
  Remote artifact: `outputs/continuous_chunk_preflight.json`.
- One new live model chunk was actually executed successfully with
  `--infer --max-chunks 1 --publish --enable-robot`. Trace:
  `outputs/continuous_tracking_live_01/trace.jsonl`.
- Live accepted command `dbdd9c4955134f4b8925494191943eb8`: all 32 reference
  rows retained, nominal duration 1.0667 s, command duration 1.99 s, peak
  commanded velocity 0.25337 rad/s, acceleration <=1 rad/s², configured jerk
  <=10 rad/s³, maximum commanded-reference lag 0.07262 rad.
- A five-second read-only post-run measurement found maximum actual endpoint
  error 0.01653 rad and measured speed <=0.00340 rad/s. Both followers remained
  `FOLLOWING`; relay `ready=true`, `phase=holding`, empty fault. Result:
  `outputs/continuous_tracking_live_01/final_status_measurements.json`.
  Its legacy `fraction_within_position` diagnostic still compares against
  0.01 rad; completion correctly uses the newly authorized 0.05 rad threshold.
- The client now also includes the final relay tracking statistics in each
  completion trace event. No second live chunk was necessary for that logging
  change; its focused client tests passed.

## FR3-C23, 9 Hz playback

Added a separate `scripts/labs_c23_control.sh` using the C23 endpoint and
required health model/variant identity. Kept state34, 640x480 source images,
32x14 delta14 and the existing initial-observation convention. The server's
source action rate remains 30 Hz; the dedicated entry point fixes speed=.3,
which the existing continuous relay plays at 9 Hz. No controller/realtime
publication rate was lowered; relay interpolation/publication remains 100 Hz.

Live C23 health and one inference completed on the station with no motion:
`outputs/c23_9hz_dryrun_01/trace.jsonl` and its captured images/raw response.
Model: FastWAM-FR3-C23, variant c23, checkpoint step 10000. Checkpoint SHA256:
`55d88ec06605f413ce20b25f5453c531aa958246a2baf0361235050903311d2e`.
The response contained 32 rows and the command recorded execution_rate_hz=9.

Complete read-only IK/tracker preflight passed against the current held
command (`outputs/c23_9hz_dryrun_01/preflight.json`):

- 32 reference rows at 9 Hz: 3.55556 s.
- Continuous command trajectory including tail: 3.91 s.
- Peak commanded speed: 0.19204 rad/s; acceleration <=1 rad/s².
- Jerk bounded to 10 rad/s³; maximum reference lag 0.05645 rad.
- No C23 robot commands were published in these checks.

77 focused tests passed. Added real WebSocket tests for C23 identity matching
and rejecting another model before inference, plus a 9 Hz playback test that
verifies source metadata and reconstructed targets remain unchanged. Changed
files passed Ruff; TOML parsing and all shell syntax checks passed.

## Recorded episode 0 replay, relaxed dynamics

The user requested a real LeRobot episode replay and explicitly authorized
increased tracking speed/acceleration and relaxed reference-lag limits.
The original 0.15 rad reference-lag gate rejected episodes 0 and 1 offline;
no motion was published for those attempts. Episode 0 was retained as requested.

Added a separately identified recorded-replay command, independent dataset
loading/hash verification in the relay, per-row delta14 reconstruction, full
IK/tracker preflight, and `scripts/labs_replay_episode.sh`. The entire 320-row
episode runs as one continuous plan after a separate start-position return.
Grippers follow the actual recorded action labels. The recorded measured
joint trajectories are never substituted for applied action labels.

Replay limits: velocity 0.8 rad/s, acceleration 2 rad/s², jerk 20 rad/s³,
reference lag 0.3 rad. URDF travel bounds, actual command tracking error and
other joint/IK guards remain enforced. Inference limits are unchanged.
80 focused tests passed, including per-row reconstruction, duplicate/missing
frame rejection, separate replay admission/start-pose gates, relaxed dynamic
limits, hard joint-bound preservation and inference-default regression.
Changed-file Ruff, TOML parsing and all shell syntax checks passed.

Live result (`outputs/episode0_replay_live_01/trace.jsonl` on the station):

- Episode 0, 320 rows; action/state content identity
  `66f167c67d9a4b5958190578a594930433f826fe8b6d7f66576a417209225d8a`.
- Start return completed, followed by complete recorded action replay.
- Reference rate 9 Hz, reference duration 35.55556 s.
- Continuous command duration 35.91 s, peak commanded speed 0.636912 rad/s,
  acceleration <=2 rad/s², jerk <=20 rad/s³, maximum reference lag 0.244720 rad.
- Command `1d474fbd73d44b6a937cdb4d59f7280f` completed successfully. Relay
  `ready=true`, `phase=holding`, empty fault; both arms `FOLLOWING`.
- The robot remains holding the final replay pose. This was actual robot
  execution, not only a dry-run or model-service test.

## Removed extra planning displacement check

A user's first repeated episode-0 replay completed its return, then latched
`Robot moved during chunk IK planning` on command
`c861d1a98b1f4a169a11fed9ce60cb2b`. The next two launches only read that
existing fault before sending a new command. The check compared measured
joint displacement during planning against an additional 0.002 rad threshold.
At the user's explicit instruction this check was deleted, without adding a
replacement planning-displacement threshold. Existing command admission,
stationarity, freshness and trajectory/tracking guards remain in place.

80 focused tests and changed-file Ruff, TOML parsing and shell syntax passed.
Deployed the updated relay, cleared the old fault by a controlled follower
reinitialization and replayed episode 0 successfully again:

- Trace: `outputs/episode0_replay_no_planning_drift_01/trace.jsonl`.
- Command `707cc9a442ca477fb1e1de61354d5d74` completed all 320 reference rows.
- 9 Hz reference, continuous command duration 35.91 s.
- Final relay `ready=true`, `phase=holding`, empty fault; both followers
  `FOLLOWING`. Robot holds the final replay pose.

## C23 direct WebSocket endpoint

Updated `scripts/labs_c23_control.sh` to default to
`ws://workspace.featurize.cn:50706/infer`; the existing client derives
`http://workspace.featurize.cn:50706/health` automatically. The station script
was synchronized and tested with one live-observation inference, without
robot publication:

- Health returned HTTP 200 and the expected `FastWAM-FR3-C23` / `c23` identity.
- Checkpoint SHA256 remained
  `55d88ec06605f413ce20b25f5453c531aa958246a2baf0361235050903311d2e`.
- Trace: `outputs/c23_direct_ws_dryrun_01/trace.jsonl` on the station.
- Returned 32x14 actions; execution rate 9 Hz; `published=false`.
- Single-request client round trip: 932.74 ms. Server-reported inference:
  394.25 ms. These are one-request observations, not a controlled comparison
  of transports.
- Four focused WebSocket/C23 tests passed; Ruff, TOML parsing and all shell
  syntax checks passed.

## Shared inference and replay dynamics

At the user's request both original-model and C23 policy planning now use the
same `solve_and_track` defaults as recorded replay: velocity 0.8 rad/s,
acceleration 2 rad/s², jerk 20 rad/s³ and command-to-reference lag 0.3 rad.
Removed the separate replay override so all three entry points use one set
of site defaults. Reference rates remain 30 Hz for the original model and
9 Hz for C23. Actual-to-command tracking remains limited to 0.15 rad;
joint bounds and the separate return planner are unchanged.

82 focused tests passed, including original/C23 policy-versus-replay trajectory
parity at both rates. Ruff, TOML parsing and shell syntax checks passed.
The deployed module was validated offline with saved model outputs:

- C23 chunk 4, command `f0fa46a0cd9447dcadf56f5e10bbaf93`, previously rejected
  at a 0.1500399 rad reference lag. New planning passed with 0.1488293 rad
  peak lag, 0.531914 rad/s peak velocity and 3.89 s duration.
- Original-model saved chunk `dbdd9c4955134f4b8925494191943eb8` passed at
  30 Hz, 0.0688656 rad peak lag and 1.57 s duration. Its offline initial
  command was the saved request state because no prior completion hold
  was recorded in that trace.
- Station report: `outputs/shared_dynamics_validation_20260919.json`.

The relay was restarted after follower deactivation, then both followers were
reactivated against a fixed current-position hold. Final status was
`ready=true`, `phase=holding`, empty fault, both followers `FOLLOWING`.
D7 remained 5 on both arms. No policy chunk was physically executed as part
of this parameter deployment; the chunk checks above were offline planning.

## Default 9 Hz inference and slower episode return

Both model clients now default to `--speed 0.3`, giving a 9 Hz policy reference
clock. C23 continues to fix that value. The original script can still take an
explicit `--speed` override. The low-level client command builder uses the
same default; model metadata remains 30 Hz source data.

Return now uses the same time multiplier. Its synchronized quintic has no
dataset frame clock, so its base duration is divided by speed; publication
remains continuous at 100 Hz. At speed 0.3 the duration is 3.333 times the base
duration, with velocity <=0.075 rad/s and acceleration <=0.045 rad/s².
Replay's initial return also uses `rate/30`. The return command is versioned
`labs_fr3_episode_joint_return_v2` and requires explicit, consistent `speed`
and `execution_rate_hz` fields so an old relay cannot silently apply the old
return timing. Updated client, relay and replay modules were deployed together.

83 focused tests passed, including return path/time-scaling equivalence,
scaled velocity/acceleration, invalid/legacy return rejection, default 9 Hz
and original/C23 policy/replay timing. Ruff, TOML and shell syntax passed.
Station validation (`outputs/nine_hz_return_validation_20260919.json`):

- Both station script restore previews selected the v2 command with speed 0.3.
- The original-model chunk `b94962b6d6314a34ae17831c80ec6dca`, previously
  rejected near right joint 7's upper limit at 30 Hz, passed offline planning
  at 9 Hz: duration 3.62 s, maximum lag 0.080557 rad.
- Read-only planning from fresh measured joints to episode 0's start gave a
  base return duration of 12.1419 s and scaled duration of 40.4731 s.
- Relay restarted and old bounds fault cleared. Both followers `FOLLOWING`,
  `ready=true`, `phase=holding`, empty fault. Maximum activation hold error
  0.0018173 rad.

No episode return or policy motion was executed during this deployment;
the arms were reactivated against a fixed current-position hold.

## Original model direct WebSocket endpoint

Updated and deployed `scripts/labs_control.sh` to default to
`ws://workspace.featurize.cn:38415/infer`, with automatically derived health
URL `http://workspace.featurize.cn:38415/health`. C23 keeps its separate
port 50706. One live-observation, no-motion request succeeded on the station:

- Model `FastWAM`, checkpoint SHA256
  `cead1fe4519a136ed432157263f247447d4adf2ba6b744829c07331e6e927dbf`.
- Trace: `outputs/original_direct_ws_9hz_dryrun_01/trace.jsonl`.
- Response 32x14; execution rate 9 Hz; `published=false`.
- Single-request round trip 867.32 ms; server inference 383.63 ms.
- Four focused WebSocket/C23 tests, Ruff, TOML parsing and shell syntax passed.

The return time multiplier remains 0.3. No relay restart or robot motion was
needed for this endpoint change.

## Planner target velocity weight disabled

Added `target_velocity_weight` in [0,1] to `track_chunk` and the shared
`solve_and_track` site planner, default 0.0. Diagnostics now identify
`continuous_ruckig_v2` and the weight. Weight 1.0 reproduces the old tangent
term. Current position/velocity/acceleration carry over every 10 ms; internal
reference rows do not stop. Both policy clients and recorded replay use this
default. Return planning, 9 Hz reference timing, 100 Hz publication, K/D,
dynamic limits and physical bounds were unchanged.

Validation:

- 88 focused tests passed, including derivative continuity, no internal-row
  stops, monotonic-reference reversal regression, intermediate weight and
  invalid-weight checks. Changed-file Ruff, TOML and all shell syntax passed.
  Full-repository Ruff reports two unrelated existing violations:
  `local_navigation/launch/dual_lidar_slam.launch.py` I001 and
  `scripts/wall_servo_control.py` SIM102.
- Production-code offline comparison passed episode 0 and saved C23 chunks
  0, 10, 20, 26. Episode peak reference lag fell from 0.244720 to 0.118584 rad;
  peak command velocity from 0.636912 to 0.515287 rad/s. Right-tip Z total
  variation fell from 549.45 to 436.48 mm (reference 432.21 mm); left from
  524.03 to 441.58 mm (reference 440.18 mm). These are planned cumulative
  travel measurements, not measured robot oscillation amplitudes.
- Station comparison artifact: `outputs/target_velocity_weight_validation.json`.
  Previous source backup: `outputs/target_velocity_backup_1789816851199326694`.
- Relay reloaded with followers deactivated first and then synchronized to
  current measured hold. An attempted diagnostic subscriber to the command
  topic triggered the exclusive-subscriber guard before replay motion. The
  observer exited; followers/relay were recovered without changing the guard.
  No moving-feedback comparison was obtained from that observer.
- Subsequent physical episode 0 test completed the initial return and all
  320 rows. Command `eb7fed5432eb41c2821e40d1ed5d5281`, trajectory 35.72 s,
  `target_velocity_weight=0.0`, final `ready=true`, `phase=holding`, no fault,
  both followers `FOLLOWING`. Completion confirms execution, not visual
  elimination of oscillation.
- Live gains confirmed both arms retain K `[240,240,240,240,100,60,20]`,
  D `[20,20,20,10,10,10,5]`, `k_alpha=0.99`.
