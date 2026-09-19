# labs FR3 station 10.3.8.31

The repository now uses only **rot6d_columns**. For a rotation R, encode
`[R00,R10,R20,R01,R11,R21]`. Existing row-encoded datasets/checkpoints must
not be relabelled as columns; regenerate or explicitly migrate their numeric
contents. The runtime rejects an explicitly incompatible rotation identifier.
Static pose presets have been numerically migrated, preserving their rotations.

## Data contract

Source robot descriptions are the actual `/left/robot_description` and
`/right/robot_description` read on 2026-09-16. FR3 link7 -> link8 is a fixed
107 mm offset. No additional tool offset is applied. Each arm uses its own
link0 reference. The two identity base mounts in the source descriptions are
not a physical dual-arm extrinsic calibration.

- action float32[20]: left xyz [0:3], left first rotation column [3:6], left
  second column [6:9], right xyz [9:12], right first column [12:15], right
  second column [15:18], binary left/right grippers [18:20].
- observation.state float32[34]: the same measured pose/gripper layout [0:20],
  then unmodified measured left joint1..7 [20:27], right joint1..7 [27:34].
  Joint angles are radians, positions metres; storage casts values to float32.
- All three RGB videos: 640x480, direct resize, nominal 30 fps.
- State uses nearest measured joints, then FK. Action uses recorded follower
  joint targets, then FK. Targets are the latest message received (MCAP
  log_time) at or before the head image stamp, with a 100 ms age bound.
- Measured state tolerance 50 ms; wrist image tolerance 45 ms. Head timestamps
  anchor a nearest-slot rate gate that tolerates camera jitter. Real source times and ten alignment offsets
  are saved. Skipped frames and reasons are recorded per episode.
- Robotiq measured knuckle: 0 rad open, 0.8 rad closed. Recorded target is an
  opening fraction. Both are thresholded at 0.5 into closed=0, open=1.
  The 0.8 rad value is the nominal normalization endpoint. Input validity uses
  the deployed driver's actual byte mapping, `0.7929 * (byte - 3) / 227` for
  bytes 0..255. Feedback can exceed nominal closure; e.g. 0.855773 rad is valid
  and binarizes to closed. It must not cause an otherwise valid episode to fail.

## Export

Install repository postprocess dependencies plus `mcap-ros2-support`.

```bash
python -m franka_duo_tele_data.labs_mcap_to_lerobot \
  --input /home/agile/work/labs/data/raw_episodes/2026/09/16 \
  --output /home/agile/work/labs/data/lerobot/labs_fr3_link8_columns_20260916 \
  --config configs/labs_fr3_31 --workers 3
```

The default task is exactly:

> Use the left arm to place the square head into the yellow box on the left, and the right arm to place the screw into the green box on the right.

One UUID is one episode; MCAP splits are merged by log time. Output contains
v3 data/video chunks, tasks, episode metadata, statistics and a conversion
manifest. A sibling `.work` directory checkpoints complete episodes, including
float64 measured/target joint provenance. Retry the same command after an
interruption; completed episodes are reused only if input size/mtime, code,
URDF and conversion settings match. Final output is assembled atomically and
never overwritten. Video hard links keep checkpoints from doubling disk use.

Each run saves its full producer contract in `.work/producer_contract_HASH.json`.
For reviewed code-only fixes, `--reuse-compatible-contract PATH` explicitly
permits reuse of that older contract. All conversion settings, URDF hashes and
input identities must still match. The final manifest retains each episode's
original producer hash and both complete contracts; it never relabels old
checkpoints as produced by new code. This option requires reviewing the code
change for output compatibility, not merely checking that schema names match.
The 2026-09-16 export reused 121 complete episodes and retried one episode
after fixing the measured-gripper range check. Final image statistics were
recomputed from every decoded output frame.

## Offline IK

Build `site/labs_fr3_kinematics` with colcon in a ROS 2/MoveIt environment.
The executable loads each saved URDF/SRDF and explicitly selects
`kdl_kinematics_plugin/KDLKinematicsPlugin`. It follows the existing executor's
seed, 0.5 rad consistency, joint-bound and continuity checks. It additionally
checks FK residuals (position <= 0.1 mm; rotation <= 0.001 rad).

```python
from pathlib import Path
from franka_duo_tele_data.labs_ik import MoveItKDL
with MoveItKDL(Path('/path/to/labs_fr3_ik'), Path('configs/labs_fr3_31'), 'left') as solver:
    result = solver.solve(pose9, current_joint_angles)
    # On success, use result['joint_positions'] as the next seed.
```

Input is xyz + two rotation columns, in that arm's link0. Tool offset is zero.
IK does not overwrite dataset labels: they remain FK of recorded targets.
The offline executable contains no robot command publishers. IK success is not
collision checking or approval to execute a trajectory; dual-arm collision
planning additionally requires the real mounting extrinsics.

## Validation

Run `python -m franka_duo_tele_data.validate_labs_dataset --dataset OUTPUT --config configs/labs_fr3_31`.
It verifies every state/action against the saved joint provenance, all frame
indices, causal target offsets and video frame counts. It also recomputes
per-channel image statistics from decoded video pixels on an 8-pixel spatial
grid in every frame, including within-image variance. Keep the sibling `.work`
directory until this validation and provenance review are complete.
