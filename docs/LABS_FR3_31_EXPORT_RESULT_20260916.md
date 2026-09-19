# labs FR3 export — 2026-09-16

Host: `agile@10.3.8.31`.

- Input: `/home/agile/work/labs/data/raw_episodes/2026/09/16`
- Output: `/home/agile/work/labs/data/lerobot/labs_fr3_link8_columns_20260916`
- Provenance/checkpoints: output path plus `.work`.
- 122 MCAP files → 122 episodes, 46,502 frames, 366 video files.
- LeRobot v3.0; nominal 30 fps; all three RGB views 640×480.
- Source file paths, sizes and modification times were verified unchanged.
  The 25 directories without MCAP files are listed in the manifest.

Task text for every frame:

> Use the left arm to place the square head into the yellow box on the left, and the right arm to place the screw into the green box on the right.

## Labels

Each arm's position/orientation describes link8 in that arm's own link0 frame.
Position is in metres. Rotation is **only columns**, in the order
`[R00,R10,R20,R01,R11,R21]`; no TCP midpoint offset is applied.

| Field | Indices | Contents |
| --- | --- | --- |
| state/action | 0:9 | left xyz, rotation column 0, rotation column 1 |
| state/action | 9:18 | right xyz, rotation column 0, rotation column 1 |
| state/action | 18:20 | left/right binary gripper, closed 0, open 1 |
| state only | 20:27 | original measured left joint1..7, radians, cast to float32 |
| state only | 27:34 | original measured right joint1..7, radians, cast to float32 |

State poses use measured joints; action poses use recorded follower targets.
All 46,502 rows have at least one measured/target joint difference >1e-6 rad.
Across all 14 joints and all rows, mean absolute difference is 0.02142215 rad;
maximum is 0.42804538 rad. No measured-state replacement was used for action.

## Synchronization accounting

There were 46,857 source head frames and 46,502 exported frames (355 skipped):

- 192 lacked a required numeric match at recording start. No numeric-match
  failures were classified as interior/end missing data.
- 163 lacked a wrist image within the 45 ms matching tolerance.
- 0 extra frames dropped by the output-rate gate; 0 invalid-payload drops.

Targets use the last receipt timestamp at or before the head image, max age
100 ms. Measured state uses nearest header timestamp within 50 ms. The task
uses nominal 30 fps video timing and retains actual head timestamps and all
matching offsets in the Parquet columns.

One episode needed a retry because the original validity bound rejected a
valid Robotiq feedback value of 0.855773 rad. The deployed driver maps bytes as
`0.7929 * (byte - 3) / 227`; byte values may exceed nominal closure. The fixed
check accepts its complete range, and this sample binarizes to closed.
The other 121 completed episodes retain their original producer hashes.

## Verification reports

Reports are in the final dataset's `meta` directory:

- `conversion_manifest.json`: full contract, source files, matching/drop
  details and per-episode original producer contracts.
- `source_audit.json`: all input files covered and unchanged; target/state
  difference statistics.
- `validation.json`: every-row measured/target FK consistency, column rotation,
  indices, binary grippers, causal timestamps and full video decode. Image
  channel statistics are recomputed from decoded RGB pixels on an 8-pixel grid
  in every frame, including within-image variance.
- `official_reader_validation.json`: LeRobot 0.4.3 with PyAV loaded the first
  and last frame of all 122 episodes (244 rows), including all three cameras,
  correct tasks, episode indices and state/action/image shapes.

MoveIt/KDL is provided separately for offline IK using the saved machine
URDF/SRDF. The 60 recorded-pose perturbation tests passed; see
`configs/labs_fr3_31/ik_validation.json`. Dataset pose labels are computed by
FK; IK does not rewrite their recorded targets. No robot motion was commanded.

Implementation and usage: [LABS_FR3_31_EXPORT.md](LABS_FR3_31_EXPORT.md).
