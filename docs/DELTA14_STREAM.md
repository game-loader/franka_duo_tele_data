# Delta14 inference and execution

The FastWAM server returns `franka_duo_midpoint_delta14_v1` actions. The client
sends unnormalized RGB20D state and three original-resolution JPEG/PNG images
over the existing `smolvla.msgpack.v1` WebSocket transport. The server owns image
resizing, concatenation, normalization and historical rotation correction.
Only `pick cup and bowl` is currently supported.

## Action contract

Responses must declare `normalized: false`, the exact representation identifier,
and finite `[32,14]` actions. Missing or incompatible metadata is rejected.

| Columns | Meaning |
| --- | --- |
| 0:3 | Left delta xyz in meters, midpoint-base axes |
| 3:6 | Left rotation vector in radians |
| 6:9 | Right delta xyz in meters, midpoint-base axes |
| 9:12 | Right rotation vector in radians |
| 12:14 | Absolute left/right openness, 0 closed and 1 open |

For each arm, starting from the **observation sent with that request**:

```text
p[i+1] = p[i] + delta_xyz[i]
R[i+1] = Exp(rotvec[i]) @ R[i]
```

All 32 rows are accumulated and scheduled for execution after inference. A new
request starts from its own observation, not the previous prediction's endpoint
or feedback received after inference. No second denormalization or sign flip is
performed. Absolute openness is preserved, not accumulated or thresholded.

The adapter produces `[32,20]` absolute midpoint targets (xyz plus first two
rotation-matrix columns per arm, followed by grippers). It checks per-row supplied
delta norms, reconstructed target jumps, workspace, finite values and gripper
bounds, then uses the existing midpoint-to-link0 transforms. Publication goes
only to `/franka_duo/joint_servo/action_chunk`; the controller receives its
existing absolute20 contract. Original SmolVLA absolute20 entrypoints remain
available.

## Installation and file-based inference

Python 3.11+ is required by the existing WebSocket transport's asyncio timeout.
ROS 2 and camera drivers remain host-managed. Install optional client packages:

```bash
uv pip install --python .venv/bin/python -e '.[inference-client]'
PYTHONPATH=src .venv/bin/python -m franka_duo_tele_data.delta14_client \
  --url ws://127.0.0.1:8081/infer \
  --state-json observation_state.json \
  --head head.png --wrist-left wrist_left.png --wrist-right wrist_right.png \
  --repeat 3
```

`observation_state.json` contains a flat unnormalized 20D array. This command
prints the raw delta14 responses and does not import ROS or publish commands.

## Live stream

### Return to the dataset initial pose

On the robot host, stop the inference loop and wait for the existing servo to
hold, then run:

```bash
cd /home/aup/franka_duo_tele_data_20260913_fastwam_check
# Read current feedback and save the plan; no motion:
bash scripts/return_to_start.sh
# Execute the return through the current servo:
bash scripts/return_to_start.sh --publish --enable-robot
```

The saved target in `configs/dataset_initial_state.json` is **episode 0, frame 0,
`observation.state`** from `franka_duo_lerobot_rgb20d_v1.tar.zst`, with its archive
SHA-256 and arm transforms. Midpoint xyz targets in meters are left
`[0.28671184, 0.19295156, -0.07738026]` and right
`[0.34742624, -0.21213263, -0.09078587]`; orientations also come from that state.
The script verifies the dataset metadata and preserves the **current physical
gripper opening**, including intermediate values. It uses smooth Cartesian
interpolation capped at 0.025 m/s and 0.10 rad/s, followed by a two-second hold.
This is a direct Cartesian return, not an obstacle-planning routine.

It reuses the active servo and the inference lock. It does not invoke PTP,
start controllers, request inference, subscribe to cameras, or record MCAP.
Only a local target plan and JSONL diagnostics are saved under
`outputs/return_to_start_<timestamp>/`. `DELTA14_DATASET`, `DELTA14_SPEED` and
`TMR_ENV_FILE` work as for the inference wrapper; speed defaults to 0.3 and
must match the servo. Stale feedback, another command publisher, incompatible
servo settings or a skipped chunk prefix stop further commands. Ctrl-C also
stops new chunks; targets already accepted by the servo can finish.

### Inference execution

For the deployed FastWAM service, use `scripts/run_delta14_loop.sh`. It checks
HTTPS `/health`, the versioned 14D/20D/32-step/30-Hz contract and the dataset
metadata before starting the stream. It reuses an already running servo; it
does not activate controllers or change their speed. Each 32-step chunk must
finish and the servo must report holding before the next observation is captured
and sent for inference. No inference is requested ahead of completion.
Each invocation creates a unique output directory.

On the robot host, after both drivers and gripper state publishers are ready:

```bash
cd /home/aup/franka_duo_tele_data_20260913_fastwam_check
bash scripts/start_servo_and_activate.sh 0.3

# One live observation and inference, no MCAP or action publication:
bash scripts/run_delta14_loop.sh

# Execute at 30 * 0.3 = 9 Hz, up to 1000 chunks:
bash scripts/run_delta14_loop.sh --publish --enable-robot
```

The default URL is
`wss://u730748-7892a859b4e0.bjb2.seetacloud.com:8443/infer`; the client uses the
supported `smolvla.msgpack.v1` subprotocol. Override `FASTWAM_URL`,
`DELTA14_DATASET`, `DELTA14_SPEED`, `DELTA14_CHUNKS` or
`DELTA14_TIMEOUT` as needed. `DELTA14_SPEED` must match the running servo.
For example, `DELTA14_CHUNKS=10 bash scripts/run_delta14_loop.sh --publish
--enable-robot` limits the run to 10 chunks. Ctrl-C stops new requests; an
already accepted plan can finish before the servo holds. Do not stop the
servo/relay while the impedance controllers are active.

Source the robot host's ROS environment and site installation first. `--dataset`
points to the original RGB20D dataset metadata (`meta/info.json` and
`franka_duo_extras/derived_manifest.json`), which supplies 30 Hz timing, camera
dimensions, arm transforms and gripper calibration. It is not the delta14 model
training dataset. Image pixels retain their native dimensions; the metadata
must match the live camera profiles.

Direct arm feedback runs at about 1 kHz. The RGB20D input cache therefore keeps
1000 pose/gripper messages independently of the image history, so camera
delivery latency and synchronization lookahead do not evict matching states.
Lookahead is measured from the image capture stamp, so network delay does not
add a second wait after receipt. The head frame must satisfy the existing
freshness limit by both capture and receipt time. Message stamps and matching
tolerances remain unchanged; no recording relay is started.

```bash
PYTHONPATH=src .venv/bin/python -m franka_duo_tele_data.delta14_stream \
  --dataset datasets/franka_duo_lerobot_rgb20d_v1 \
  --url ws://127.0.0.1:8081/infer \
  --output outputs/delta14_stream.jsonl
```

Default operation captures one observation, writes inference and reconstructed
targets to local JSONL diagnostics, and exits without robot command publication.
Neither dry-run nor action execution starts rosbag2, an arm recording relay, or
an MCAP trace publisher. This inference path has no MCAP recording option and
does not require the MCAP storage plugin or a recording configuration.
The Python entrypoint overwrites its JSONL path, so use a new path per run;
the shell wrapper creates a unique directory automatically.

For an explicitly authorized robot run, append **both** `--publish --enable-robot`.
The site servo must already be running with matching playback speed, 30 Hz
dataset timing, `commit_lead_steps=0`, `blend_steps=4` and `quintic_hold_v1`.
Default `--speed 0.3` executes at 9 Hz (about 3.56 seconds per chunk), then holds
while capturing a fresh observation and waiting for inference. Delayed images
captured before completion are discarded. Once the response is validated, its
full 32 rows start one second ahead of the running servo clock to allow transfer
and IK; an unstarted servo can begin at step zero. The client verifies the
acknowledged start and end steps, and stops further requests if any prefix was
skipped. The inherited `--request-lead-ms` option is unused by delta14.
Servo status freshness, fault checks, unique subscriber checks, acknowledgement
and final hold checks remain active. Stopping requests does not cancel an
already accepted plan: it finishes and the site servo holds.

`127.0.0.1` refers to the client machine. When the server runs on another host,
bind it to that host's Tailscale address and pass the matching `--url`; do not
expose the unauthenticated service publicly.

Automated tests cover protocol and ROS-free execution. Live model and robot
validation must be performed on the deployment host before relying on this path.
