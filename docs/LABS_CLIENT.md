# Labs model inference clients

`labs_client` is separate from the TMR midpoint client. It reuses
`SmolVLAClient` and `labs_action_delta.delta14_to_absolute20`. ROS 2 and the
MoveIt/KDL solver stay host-managed. Python 3.10 supports the transport now,
so ROS Humble's Python extension does not need to be rebuilt for Python 3.11.

## Unified station script

### Labs SmolVLA

`scripts/labs_smolvla_control.sh` uses `ws://100.73.14.65:8081/infer`,
`--server-profile absolute20`, and binary `smolvla.msgpack.v1` transport.
It sends raw state20, three 640x480 RGB images and task, and accepts finite
`actions[H,20]` (1 <= H <= 256). The current service returns 32 rows.
The SmolVLA wire request contains only `state`, `images`, `task` and `request_id`;
it does not include FastWAM's `state_format` or `action_format` extensions.

Input/output order is `[left xyz+rotation6D, right xyz+rotation6D,
left gripper, right gripper]`. Each pose is link8 in that arm's own link0;
rotation6D concatenates the first two matrix columns. No client normalization,
inverse normalization or delta integration is performed. XYZ stays unchanged;
rotation columns are orthogonalized and grippers thresholded at 0.5.

The absolute entry points no longer request `/info` or `/health`, compare model
names/checkpoints/policies, or send a named policy. They use the server default.
`prediction_horizon` and `n_action_steps` are informational; execution length
comes from `actions`. Physical format declarations, when present, must agree
with the selected mode: normalized/delta actions or incompatible dimensions,
layouts, units and rotation formats are rejected. Response `request_id` must
match. Raw server metadata and predictions are preserved before adaptation;
missing metadata does not establish how a model's training labels were made.

By default all returned rows execute. `--execute-steps N` selects a prefix
(1 <= N <= returned rows), while the entire response is still saved. Unused
rows are discarded. At 9 Hz, 32 rows span 3.556 seconds plus settling. The site
relay performs continuous tracking at 100 Hz and retains its physical limits.

```bash
# Dry-run, no robot publication:
bash scripts/labs_smolvla_control.sh --infer
# Return once, then execute complete chunks:
bash scripts/labs_smolvla_control.sh --restore --infer --max-chunks 100 --publish --enable-robot
```

Use `--url` or `LABS_SMOLVLA_SERVER_URL` to override the endpoint. `--task`
overrides the station dataset task. Internal control/recording retains state34;
the model receives its first 20 entries. The default dataset supplies station
configuration and measured episode-start states, not evidence of model training.

### FastWAM and joint16 entry points

`scripts/labs_control.sh` uses `absolute_joint16` with binary
`fastwam.msgpack.v1` at
`wss://u730748-b58d-17b41c61.bjb1.seetacloud.com:8443/infer`.
It sends raw state16 and accepts `actions[H,16]` ordered
`[left joint1..7, left gripper, right joint1..7, right gripper]`.
Joint angles are absolute radians and track directly without IK or delta
integration; grippers are thresholded at 0.5. All returned rows execute unless
`--execute-steps N` is supplied. Raw replies remain complete in the archive.

`scripts/labs_fastwam_eef_control.sh` selects `absolute20` with
`fastwam.msgpack.v1` at the same URL, adding `state_format=rot6d_cols20` and
`action_format=absolute20` to requests. Select this script when that server's
default policy is EEF20. Model identity can change freely, but state dimension
and physical action meaning must still match the chosen entry point.

To change the FastWAM EEF20 server, edit `INFERENCE_URL` near the top of
`scripts/labs_fastwam_eef_control.sh`, then run:

```bash
bash scripts/labs_fastwam_eef_control.sh --infer --task 1
```

Only the inference WebSocket address is needed; no health URL or model name
is configured or sent. Alternatively use `LABS_FASTWAM_EEF_SERVER_URL` or
`--url ws://HOST:PORT/infer` (the command-line option takes precedence).
The command above is a dry-run; robot motion requires both `--publish --enable-robot`.

For FastWAM, select `--task 1`, `--task 2`, `--task 3`, or `--task 4` on every
inference run (`--task-id` is also accepted). The MessagePack request sends an
integer `task_id` instead of `task` text. The server selects the training
instruction and text embedding. No model name is sent. Request keys are
`request_id`, `state`, `images`, `state_format`, `action_format`, and `task_id`.
The run configuration and observation records preserve the selected `task_id`.

| Task | Instruction |
| --- | --- |
| 1 | Use the left arm to place the square head into the yellow box on the left, and the right arm to place the screw into the green box on the right. |
| 2 | Open the drawer, pick up the white charger and place it inside the drawer, then close the drawer. |
| 3 | Stack the three bowls together. |
| 4 | Fold the towel. |

```bash
# On the robot host; change 3 to 1, 2 or 4 for a different task:
cd /home/agile/work/labs/data/tools/labs31_client
bash scripts/labs_fastwam_eef_control.sh --infer --task 3 --max-chunks 100 --publish --enable-robot
```

Inference starts from the current pose. With `--restore`, the task number selects
episode 0/frame 0 from the dataset in `configs/labs_fr3_31/task_starts.json`:

| Task | Restore dataset under `/home/agile/work/labs/data/` |
| --- | --- |
| 1 | `lerobot/labs_fr3_link8_delta14_20260916` |
| 2 | `lerobot_next_state20/labs_fr3_link8_next_state20_20260923` |
| 3 | `lerobot_next_state20/labs_fr3_link8_next_state20_20260924_bowls105` |
| 4 | `lerobot_next_state20/labs_fr3_link8_next_state20_20260924_towel98` |

```bash
# Return only to the bowls start (change 3 to 2 for drawer, 4 for towel):
bash scripts/labs_fastwam_eef_control.sh --restore --task 3 --publish --enable-robot
# Return once, then infer the same task:
bash scripts/labs_fastwam_eef_control.sh --restore --infer --task 3 --max-chunks 100 --publish --enable-robot
```

Both arms return to the recorded measured joints; grippers retain their current
opening. Pose20 datasets store these joints in `meta/measured_provenance`, whose
checksum, first-frame pose/timestamp and URDF are checked. No action label or IK
solution is used for return. The client records the actual restore dataset.

The relay loads all configured task starts independently and advertises their
identities, so changing tasks does not require restarting controllers. After
updating relay code or the catalog, stop inference clients and reload once with
`bash scripts/labs_robot_control.sh --start --restart-relay --publish --enable-robot`.
This re-establishes a hold at the current pose; it does not perform a task return.
For task-specific returns, `--start-episode` must remain 0. `--dataset` continues
to select the base station contract; edit the task catalog to change task targets.

`scripts/labs_joint16_control.sh` selects `absolute_joint16` for an explicit
URL and defaults to `smolvla.msgpack.v1`. `--wire-protocol NAME` overrides the
transport subprotocol; `--joint16-protocol NAME` remains a wrapper alias.
Legacy named profiles remain available explicitly; the absolute wrappers now
select the physical profiles above. C23's legacy delta14 contract is unchanged.

### Connection recovery

The client disables aiohttp's automatic heartbeat because no receiver runs
while a robot chunk or restore is executing. Active requests still have a
timeout. If a connection closes or an inference request times out before an
action is delivered, it reconnects once and captures a fresh state and fresh
images for a new request ID. A second failure stops the client. Server errors
and invalid action payloads are not retried, and robot commands are never
republished by this retry path. Timeout defaults to 60 seconds in the wrappers;
the relay observation-age limit remains independent. Retry attempts and their
observations are recorded along with the original action chunks and feedback.

On `agile@100.90.202.124`:

```bash
cd /home/agile/work/labs/data/tools/labs31_client

# Preview episode 0's joint target without ROS subscriptions or model access:
bash scripts/labs_control.sh --restore

# Real camera observations + one model response, without motion:
bash scripts/labs_control.sh --infer

# Actual arm return only (preserves grippers):
bash scripts/labs_control.sh --restore --publish --enable-robot

# Infer and execute from the current pose, without another return:
bash scripts/labs_control.sh --infer --max-chunks 10 --publish --enable-robot

# Return once, then infer/execute 10 complete chunks:
bash scripts/labs_control.sh --restore --infer --max-chunks 10 --publish --enable-robot
```

`--restore` and `--infer` are independent. Combined mode connects/checks the
server, returns once, waits for measured arrival, then obtains new images and
starts inference. Restore-only does not connect to the model server, subscribe
to cameras, or invoke IK. Without both publication gates, restore is a target
preview, not a claim of actual arrival. Both model clients default to one chunk
at 9 Hz target timing (`--speed 0.3`); physical execution remains subject to all
relay checks. Return uses the same time multiplier, extending the base joint
return duration by 1/0.3 while retaining continuous 100 Hz interpolation.

The script supplies this station's dataset, endpoint, ROS environment and
host-built IK location. Override with `--dataset`, `--url`, `--start-episode`,
`--ik`, `--output`, `--max-chunks` (positive count), or `--speed`.
`LABS_DATASET` / `LABS_SERVER_URL` can change defaults. The station script
uses ROS domain 100 even if the calling shell inherited another domain; set
`LABS_ROS_DOMAIN_ID` explicitly to override it. Each run creates a new
trace directory. `--help` works without ROS.

For execution, the script discovers an existing site relay or launches a
persistent one under `flock`. Its log is `outputs/labs_relay/relay.log`.
A newly launched relay uses `--hold-current-on-start`: it latches a stationary
measured pose, streams that fixed target without gripper commands, and waits
for both followers to be active and report `FOLLOWING`, with measured position
and velocity settled for 0.5 s. This supports explicit site activation without
jumping to an old target. No inference commands are accepted while arming.
The relay remains alive after the client exits to hold the final joint target;
interrupting the client does not cancel an accepted chunk. Reused relays keep
the dataset/episode/configuration and task-start catalog loaded at startup;
a mismatched return identity fails rather than silently selecting another target.
The script does not activate controllers or take over teleoperation topics:
both arms must already have the follower and state broadcasters active, with
exclusive destination ownership. Both followers were explicitly activated on 2026-09-19 after establishing
a fixed hold at their current measured joints. The subsequent check reported
both `FOLLOWING` and the relay `ready=true`, `phase=holding`.

The older `run_labs_client.sh` retains its default return-then-infer behavior
when neither explicit operation flag is passed.

## FR3-C23 at 9 Hz

Use the separate `scripts/labs_c23_control.sh` entry point for:

- WebSocket: `ws://workspace.featurize.cn:50706/infer` (direct connection).
- Health: `http://workspace.featurize.cn:50706/health`.
- Required identity: `model=FastWAM-FR3-C23`, `variant=c23`.

```bash
cd /home/agile/work/labs/data/tools/labs31_client

# Real C23 inference without moving the robot:
bash scripts/labs_c23_control.sh --infer

# Execute ten chunks from the current pose:
bash scripts/labs_c23_control.sh --infer --max-chunks 10 --publish --enable-robot

# Return once to the episode start, then execute ten chunks:
bash scripts/labs_c23_control.sh --restore --infer --max-chunks 10 --publish --enable-robot
```

The source contract remains 30 Hz with state34, three 640x480 images and
32x14 delta actions. C23's image resizing/compositing is performed by the
server; the client sends the same three source images as the original model.
Both model clients default to `--speed 0.3`, and the C23 entry point fixes it:
policy reference rows
advance at **9 Hz**, giving 32/9 = 3.556 seconds of reference time per chunk.
Ruckig interpolation and relay publication remain at 100 Hz, with a settling
tail when needed. Grippers follow the same slowed reference timeline. The
trace records `action_rate_hz=30`, `speed=0.3`, `execution_rate_hz=9` and the
unmodified server health. It accumulates every delta onto the previous target,
starting from the request observation, just like the legacy FastWAM delta14 profile.

This script shares the existing relay, return tolerance and robot gates. It
checks the C23 identity before inference/return in combined mode. It does not
restart the relay or alter the original model's entry point. Use `--url` or
`LABS_C23_SERVER_URL` when the C23 endpoint changes. `--speed` and
`--server-profile` overrides are rejected by this dedicated entry point.

## Model inputs and chunk convention

The inputs match the Labs dataset:

- `state`: float32[34], measured link8 poses in each arm's own link0, rotation
  columns, binary left/right gripper openness, then measured joint1..7 for
  each arm. FK, name-based joint ordering and Robotiq mapping reuse the export
  implementation. There is no midpoint transform or extra TCP offset.
- `images`: `head`, `wrist_left`, `wrist_right`, PNG-encoded RGB, 640x480.
  BGRA/BGR conversion and PyAV direct resizing match the exporter. No crop,
  letterbox, client-side normalization or action clipping is performed.
- `task`: the Labs square-head/yellow-box and screw/green-box task, read from
  dataset metadata when `--dataset` is supplied.

Live observations use the head image's header timestamp. Wrist tolerance is
45 ms, measured joint/gripper tolerance 50 ms, lookahead 60 ms, maximum age
200 ms by both header and receipt. Joint histories retain 1200 messages.
The client subscribes to the same `measured_joint_states` topics used by the
Labs export. It never changes the raw recorder, topics or stored MCAP messages.

**The legacy FastWAM delta14 profile (`labs`) and C23 accumulate each delta14 chunk.**
The current FastWAM entry point instead uses absolute joint16 as described above.
SmolVLA uses absolute pose20 targets as described above; joint16 uses absolute
joints as described in `LABS_JOINT16_CLIENT.md`.
For each arm, with the initial pose as `p[-1], R[-1]`:

```text
p[k] = p[k-1] + delta_xyz[k]
R[k] = Exp(delta_rotvec[k]) @ R[k-1]
```

Translation remains in the arm's link0 axes; it is not rotated into tool axes.
Rotation vectors are composed as rotations, not added as vectors. Each new
chunk starts from its newly measured request state, with no integration state
carried between requests. Every returned row is retained and converted to an
absolute20 target before the complete chunk is planned and continuously tracked.
The next observation is captured after the chunk finishes and the relay holds.

Grippers remain absolute binary open/closed values, using `prediction >= 0.5`.
Legacy delta14 FastWAM/C23 reject predictions outside [-0.1,1.1]; SmolVLA thresholds finite
gripper predictions directly. Cartesian deltas are not normalized or clipped.
Existing Cartesian step/extent, IK and joint checks still apply to the full
accumulated trajectory.
The whole-chunk seed-relative rotation limit is now pi radians (180 degrees,
the full SO(3) principal-angle range), with 1e-6 numerical tolerance. This is
not a cap on the sum of rotation travel: a smooth sequence may turn farther.
Per-step rotation remains limited to 0.35 rad; translation limits remain
0.04 m per step and 0.60 m from the chunk seed. Joint bounds, IK checks and
continuous tracker speed/acceleration limits remain in force.

`chunk_reference=request_observation` identifies the integration seed.
`chunk_integration=cumulative_link0_delta14_v1` identifies the execution rule in
commands, configuration records, trace events and terminal chunk summaries.
Older recordings without this integration field used independent offsets from
the fixed request pose; inspect their saved `command.targets` for the exact
trajectory rather than reinterpreting their raw deltas with today's decoder.
The relay continues to consume absolute20 targets, so this client-only change
does not require restarting an existing relay or follower controllers.

This corrects the earlier misunderstanding of the user's requested cumulative
execution. It does not modify offline labels or dataset replay: those deltas
remain relative to each row's own measured state. They are not strictly
successive-target increments. Accumulation implements the requested execution
rule, not a claim that existing checkpoints now reproduce demonstration paths.
The server's `no_cumulative_delta=true` describes its unaccumulated wire output;
client execution integration is recorded separately and raw metadata is retained.

## Existing WebSocket transport

Use subprotocol `fastwam.msgpack.v1`, binary MessagePack frames, and the existing
request keys `request_id`, `state`, `images`, `task`. The server now needs to
accept 34 state values. Images are byte strings, not base64.

The response must echo `request_id` and contain:

```json
{
  "request_id": "same-as-request",
  "action_representation": "franka_fr3_duo_link8_delta14_v1",
  "normalized": false,
  "chunk_reference": "request_observation",
  "actions": [[0,0,0,0,0,0,0,0,0,0,0,0,1,1]]
}
```

The example shows one row; this deployed service returns exactly [32,14]. Its
`/health` must confirm dimensions, full state/action layouts, 30 Hz source
timing, task and unnormalized physical values. Both the original Labs dataset
identifier and the server's `franka_fr3_duo_link8_delta14_v1` are supported;
the reply must match health. Midpoint identifiers and nonfinite actions are
rejected. Temporal metadata is retained, with the explicit client convention
described above. The old TMR client still uses `smolvla.msgpack.v1`.

## Install and verify without motion

On `agile@100.90.202.124`, use a separate working directory and the system Python
matching ROS Humble:

```bash
uv venv --python /usr/bin/python3 --system-site-packages .venv
uv pip install --python .venv/bin/python -e '.[labs-client]'

# Read real images/state only; no model server or robot publishers:
bash scripts/run_labs_client.sh --capture-only \
  --dataset /home/agile/work/labs/data/lerobot/labs_fr3_link8_delta14_20260916

# One complete model chunk, decode to absolute20 and log it, no execution:
bash scripts/run_labs_client.sh --url wss://YOUR_SERVER/infer \
  --dataset /home/agile/work/labs/data/lerobot/labs_fr3_link8_delta14_20260916
```

The wrapper defaults to ROS domain 100 and the station's CycloneDDS file.
Override `LABS_ROS_SETUP`, `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`,
`LABS_DDS_CONFIG`/`CYCLONEDDS_URI` or `LABS_PYTHON` when using another host.
The client saves every transmitted image, measured state, source timestamps,
raw model reply, reconstructed full chunk and publication/completion events in
a new output directory. Existing output directories are never overwritten.
Each observation directory also includes `response.msgpack`, retained even
when subsequent numeric validation rejects the model response.

The installed station copy is
`/home/agile/work/labs/data/tools/labs31_client`. Its verified service URL is
`ws://workspace.featurize.cn:37388/infer`.
See [the validation record](LABS_CLIENT_VALIDATION_20260919.md).

## Site relay and explicitly enabled execution

The client only publishes versioned JSON commands to
`/franka_duo/labs/action`; it never publishes controller commands. The new
`labs_relay` owns the adapter to the site's existing follower joint topics
and Robotiq opening-fraction topics. It does not start or switch controllers.

Build the existing `site/labs_fr3_kinematics` package with host ROS/MoveIt, then
source its install. The relay uses `labs_ik.MoveItKDL` to solve ALL chunk rows
before publishing any target. Each IK seed is the preceding solution. The
solver runs in isolated ROS domain 213 and never publishes robot commands.

Both the client and relay require **both** `--publish --enable-robot` for
publication. Without these flags the relay is a read-only status monitor.
Only start publication after site ownership is assigned to this relay: the
joint follower and two state broadcasters must be active per arm, other arm
controllers inactive, and teleoperation publishers removed from the follower
and gripper destinations. Checks reject competing publishers; do not bypass
them by simply changing topic names.

With the host ROS environment/domain configured as above, the explicit relay
command is:

```bash
PYTHONPATH=src .venv/bin/python -m franka_duo_tele_data.labs_relay \
  --ik site/install/labs_fr3_kinematics/lib/labs_fr3_kinematics/labs_fr3_ik \
  --dataset /home/agile/work/labs/data/lerobot/labs_fr3_link8_delta14_20260916 \
  --start-episode 0 --publish --enable-robot
```

The corresponding explicitly enabled client command is:

```bash
bash scripts/run_labs_client.sh --url wss://YOUR_SERVER/infer \
  --dataset /home/agile/work/labs/data/lerobot/labs_fr3_link8_delta14_20260916 \
  --start-episode 0 --max-chunks 10 --publish --enable-robot
```

The relay checks model hashes, command age (1 s on receipt), observation age
(10 s including inference/IK), measured joint drift (0.02 rad), current
feedback freshness (150 ms), controller status, exclusive output ownership,
Cartesian jumps, URDF joint bounds, IK residuals and joint continuity. Commands
have unique IDs; the client waits for the matching ID and final hold, with
fresh status throughout. Motion faults latch until relay restart.

Policy reference timing defaults to 9 Hz (`--speed 0.3`), with every IK row
retained as a reference knot. A shape-preserving cubic Hermite reference joins
the knots with continuous velocity; only the complete chunk's endpoints have
forced zero velocity. The relay precomputes a continuous Ruckig tracker at
100 Hz, carrying its commanded position, velocity and acceleration into every
retarget. This follows the existing site's `JointRuckigTracker` design instead
of restarting a motion at every policy row. The optional `labs-client` group
includes `ruckig==0.12.2`.

Original-model inference, C23 inference and recorded replay share the site
dynamics: commanded velocity, acceleration and jerk are bounded to 0.8 rad/s,
2 rad/s² and 20 rad/s³. Reference velocities are bounded for Ruckig input;
reference joint positions are retained. Actual commands may lag the reference;
intermediate reference rows are not guaranteed to be reached at their nominal
timestamps. The user explicitly authorized this continuous tracking behavior.
Reference lag above 0.3 rad or tracked paths outside joint bounds reject the
whole chunk before publication. `--speed` explicitly slows the reference clock.
The next chunk starts from the previous held command, avoiding a discontinuity
caused by the impedance controller's measured steady-state error.

After the last reference row the tracker decelerates and converges to its exact
final commanded joint target. The client waits for this tail and measured
settling before requesting another chunk. Completion uses the authorized
0.05 rad position tolerance, measured velocity <=0.02 rad/s and a continuous
0.5 s dwell for both return and policy motion. Gripper opening does not gate
completion: closing on an object need not reach zero opening. Gripper commands
and fresh-feedback checks remain active. Startup arming retains its
separate 0.01 rad hold tolerance. The status/trace reports reference duration,
command duration, peak velocity/acceleration and maximum reference lag. The
precomputed command duration must be <=90 s; controller settling adds time.

### Once-per-startup joint return

With `--restore --infer` (or the legacy default), before the first inference
request the client returns the arms once to the
selected `--start-episode` (default 0), frame 0. FastWAM task-specific returns
always select episode 0/frame 0 from that task's catalog dataset. It reads
**measured joints** from `observation.state[20:34]` or, for pose20 exports, the
matching hashed `meta/measured_provenance` sidecar, not action labels or an IK
solution for the saved end-effector poses. The state layout, FK and joint limits are validated. Both
client and relay load the dataset independently and compare a content hash of
the episode, frame, state and URDF hashes. Missing/duplicate first frames fail.

The separate `labs_fr3_episode_joint_return_v2` command uses the relay's fresh
joint feedback and does not depend on cameras or invoke IK. Both arms follow
one synchronized quintic trajectory. The base duration is bounded to 0.25 rad/s
and 0.5 rad/s², then divided by the requested `speed`. Return has no dataset
frame clock: its `execution_rate_hz=30*speed` records the shared time-scale
choice, while the actual joint target stream remains 100 Hz. At the default
speed 0.3, the return takes 3.333 times the base duration, with velocity bounded
to 0.075 rad/s and acceleration to 0.045 rad/s². The command explicitly carries
both timing fields; the version prevents an older relay silently ignoring the
slower return. Deploy client and relay together. The status records the base
and scaled durations, selected speed and scaled dynamic limits.
The return preserves the physical gripper opening and publishes no gripper
commands. Joint feedback must include fresh velocities, with every joint at
or below 0.02 rad/s before starting. Completion requires every joint within
0.05 rad of its recorded target and velocity at or below 0.02 rad/s continuously
for 0.5 s, after the trajectory ends. A failure/timeout prevents inference.

Only after the matching completion acknowledgment does the client acquire a
new camera observation and start the policy loop. Later chunks do not return
again. Dry-run records the selected target and command without moving the
robot or claiming it reached the target; inference then uses the actual live
pose. `--capture-only` skips return and dataset-start loading entirely.

The joint return checks bounds and tracking, but does not plan a collision-free
path through the scene. The selected episode start and the path from the current
pose must be suitable for the site's workspace.

The relay publishes at 100 Hz, rejects scheduling gaps over 50 ms while moving,
checks tracking error, and holds after the final target settles. Stopping the
client stops new chunks; an already accepted chunk finishes. A relay fault or
relay shutdown stops publication; physical behavior then depends on the site's
existing controller watchdog. These bounds/IK checks do not provide collision
planning or compensate for missing physical dual-arm extrinsic calibration.

## Replay recorded LeRobot episode actions

`scripts/labs_replay_episode.sh` reconstructs each stored action against its
own recorded same-row state, then performs whole-episode IK and continuous
tracking through the same site relay. Measured joints are used only for the
explicit starting-pose return; they are never substituted for action labels.
No model server is used. The relay independently loads the same dataset and
checks a content identity covering the episode, states, actions and URDF hashes.

```bash
# Full offline preflight (default); creates no robot command publishers:
bash scripts/labs_replay_episode.sh --episode 0

# Return once, then replay the complete episode and recorded gripper actions:
bash scripts/labs_replay_episode.sh --episode 0 --publish --enable-robot
```

Default reference rate is 9 Hz (`--rate`), with 100 Hz continuous tracking.
The initial joint return uses the same `speed=rate/30` time multiplier.
The running relay must use the same dataset and `--start-episode` as the replay.
Data frame indices must be unique/consecutive and timestamps match the source
30 Hz grid. The entire episode, including its continuous command trajectory,
is checked before any replay motion. A whole-episode motion cannot be longer
than 90 s; the first return is a separate motion. The accepted replay continues
if the client exits, then the relay holds the final target.

At the user's explicit request, recorded replay and both model clients share:
0.8 rad/s velocity, 2 rad/s² acceleration, 20 rad/s³ jerk and 0.3 rad maximum
command-to-reference lag. Actual-to-command tracking error still trips at
0.15 rad, and URDF joint travel bounds remain enforced. Completion uses the
approved 0.05 rad joint-position tolerance, velocity <=0.02 rad/s and a 0.5 s
dwell. Recorded trajectory extent is checked against each own row instead of
applying one short inference chunk's global extent cap to the entire episode.
Adjacent-step and IK jump checks remain in place.

The shared planner now uses `continuous_ruckig_v2` with
`target_velocity_weight=0.0` (in `labs_relay.solve_and_track` and
`labs_tracking.track_chunk`). This scales the interpolated reference velocity
used as Ruckig's endpoint velocity; 1.0 reproduces the previous tangent term.
It is a Python planner parameter in [0,1], not a controller gain or client CLI
flag. Both model clients and replay use the same default, recorded in the
relay's tracking diagnostics. Each 10 ms retarget still inherits the previous
position, velocity and acceleration: zero endpoint velocity does not introduce
per-row stops. The separate initial-position return planner is unaffected.

The trace saves episode identity, source actions/states, reconstructed targets,
preflight dynamics, return/replay commands, progress, final measured joints and
matching relay completion status. Replay does not provide collision planning.

The extra 0.002 rad displacement comparison between the beginning and end of
IK planning was removed at the user's request. No replacement planning-drift
threshold was added. The existing command admission, feedback freshness,
stationary-start, trajectory-limit and physical tracking checks still apply.

## Save every server action chunk and inspect tracking

Every model entry point automatically writes `actions.msgpack` inside the
printed run directory, on normal exit, Ctrl+C or an exception. `--output PATH`
selects that directory (it must not already exist). No extra recording flag
is needed. The portable `labs_inference_recording_v1` bundle includes:

- Every original server reply, before validation or gripper thresholding,
  including replies rejected by the client; its matching 34D request state,
  image references/source timestamps, task and receipt time.
- Validated absolute20 targets, command/request IDs, source/playback rate,
  model metadata and URDF text/hashes. All chunk rows use their own saved
  request observation; never reconstruct later chunks from the first state
  of the entire run or cumulatively sum delta actions.
- Whether the command was published, completed, faulted or remains unconfirmed.
  Published does not mean completed. Dry-run chunks remain `not_published`.
- Approximately 20 Hz relay status and fresh actual joint positions while
  publishing, keyed to command ID; sampled maximum/RMS joint error. Completed
  chunks with a fresh holding sample also include final measured-vs-model
  endpoint position and rotation error for each arm.

The relay status provides its last published joint target. It is paired with
latest measured feedback when received, with source/receipt timestamps saved.
These are asynchronous samples, not exact 100 Hz pairs; use their trends and
endpoint error for diagnosis, not as a precise reconstruction of every servo
tick. No extra command-topic subscriber is created. Missing/stale feedback is
recorded as missing, never as zero error or a successful endpoint.

The first Ctrl+C stops new inference requests and immediately exports a
snapshot. If a chunk has already been published, the client continues receiving
feedback until its matching completion acknowledgment or an error/120 s timeout,
then exports the final bundle. A second Ctrl+C exits that wait; the relay may
still finish the accepted chunk. Unconfirmed execution stays explicitly marked.
Raw replies and publication/completion events are flushed and fsynced to
`actions.journal.msgpack` as they arrive, so a process crash does not require
waiting for the final export to retain previously received chunks.

```bash
# Actual inference plus automatic recording in this chosen new directory:
bash scripts/labs_control.sh --restore --infer --max-chunks 100 \
  --publish --enable-robot --output outputs/fastwam_recording_01

# Print per-chunk results and create an offline interactive playback page:
bash scripts/labs_inspect_actions.sh outputs/fastwam_recording_01/actions.msgpack \
  --html outputs/fastwam_recording_01/replay.html

# After abnormal termination, rebuild the portable file from the journal:
# Run only after that inference process has exited.
bash scripts/labs_inspect_actions.sh outputs/fastwam_recording_01 --recover
```

Open `replay.html` in a browser. Select the chunk, arm and XYZ axis, then play
or scrub through the model targets, recorded relay targets and measured FK
positions; a separate plot shows all seven joint errors. Model and execution
curves have separate time origins (reference start vs first execution sample).
The viewer runs offline and does not move the robot. The archive retains all
raw 14D values and reconstructed targets for subsequent robot replay tooling.
It includes URDF geometry for FK; camera bytes remain in the observation folders.

Python access:

```python
import msgpack
with open("outputs/fastwam_recording_01/actions.msgpack", "rb") as f:
    run = msgpack.unpack(f, raw=False)
for chunk in run["chunks"]:
    actions = chunk["raw_response"]["actions"]
    reference_state = chunk["observation"]["state"]
    print(chunk["index"], chunk["execution"], chunk["summary"])
```
