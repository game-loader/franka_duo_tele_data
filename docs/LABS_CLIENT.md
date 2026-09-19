# Labs state34 / delta14 inference client

`labs_client` is separate from the TMR midpoint client. It reuses
`SmolVLAClient` and `labs_action_delta.delta14_to_absolute20`. ROS 2 and the
MoveIt/KDL solver stay host-managed. Python 3.10 supports the transport now,
so ROS Humble's Python extension does not need to be rebuilt for Python 3.11.

## Unified station script

The original-model entry point defaults to the direct WebSocket
`ws://workspace.featurize.cn:38415/infer`, with health at
`http://workspace.featurize.cn:38415/health`.

On `agile@10.3.8.31`:

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
the dataset/episode/configuration with which they were started; a mismatched
return identity fails rather than silently selecting another target.
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
unmodified server health. It retains the user-confirmed initial-observation
reference for every delta row; it does not cumulatively integrate the chunk.

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

**The agreed inference contract is that ALL H rows reference the observation
sent with that request.** For every row independently:

```python
references = np.broadcast_to(request_state34, (H, 34))
targets20 = delta14_to_absolute20(actions14, references)
```

Positions use `p_request + delta_xyz`, rotations use
`Exp(delta_rotvec) @ R_request`. Grippers are absolute binary open/closed values:
raw model predictions are mapped with `open = prediction >= 0.5`, matching the
Labs labels. Predictions outside [-0.1,1.1] are rejected as a scale error. This
allows modest endpoint overshoot from the regression model without clipping
Cartesian deltas. Raw predictions and `binary_open_ge_0.5_v1` are retained in
the trace; the existing bidirectional conversion functions remain unchanged.
There is no cumulative sum. Every row is retained, in order, for execution.
The next request captures an observation only after the complete chunk has
finished and the relay reports holding.

This inference convention does not relabel the dataset: its rows reference
their respective same-row measured states. The deployed server's health text
currently says "each row is one incremental dataset step". The user explicitly
confirmed interpreting its entire returned chunk against the initial state
anyway. The trace preserves that original server metadata and records the
user-confirmed `request_observation` interpretation separately. The client
never cumulatively sums deltas or fabricates future measured states.

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

On `agile@10.3.8.31`, use a separate working directory and the system Python
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
`ws://workspace.featurize.cn:38415/infer`.
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
0.5 s dwell for both return and policy motion. Startup arming retains its
separate 0.01 rad hold tolerance. The status/trace reports reference duration,
command duration, peak velocity/acceleration and maximum reference lag. The
precomputed command duration must be <=90 s; controller settling adds time.

### Once-per-startup joint return

With `--restore --infer` (or the legacy default), before the first inference
request the client returns the arms once to the
selected `--start-episode` (default 0), frame 0. It reads **measured joints
`observation.state[20:34]`**, not action labels or an IK solution for the saved
end-effector poses. The state layout, FK and joint limits are validated. Both
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
