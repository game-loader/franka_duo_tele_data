# labs absolute20 ↔ delta14

This offline tool converts the labs FR3 link8 dataset in both directions.
The reference is the **same row's measured observation.state**, independently
for every frame. Each arm uses its own link0 axes. State remains float32[34].

For each arm:

```text
delta_xyz = p_target - p_state
delta_rotvec = Log(R_target @ R_state.T)
p_target = p_state + delta_xyz
R_target = Exp(delta_rotvec) @ R_state
```

`Log`/`Exp` are SO(3) rotation-vector conversions implemented with SciPy
`Rotation`. The vector direction is the rotation axis and its norm is the
angle in radians (principal angle 0..pi). These are not Euler-angle differences.
At exactly pi the axis sign is non-unique, but the physical rotation is the
same. Reconstruct using the same measured state, not the current state at
some later time.

| delta14 indices | Meaning |
| --- | --- |
| 0:3 | Left delta xyz, metres, left link0 axes |
| 3:6 | Left delta rotation vector, radians, left link0 axes |
| 6:9 | Right delta xyz, metres, right link0 axes |
| 9:12 | Right delta rotation vector, radians, right link0 axes |
| 12:14 | Absolute left/right gripper, unchanged (dataset uses 0/1) |

The absolute20 representation is left xyz + first two rotation **columns**,
right xyz + first two rotation columns, then left/right grippers. Column order
is `[R00,R10,R20,R01,R11,R21]`.

The version identifier is `labs_fr3_link8_state_relative_link0_delta14_v1`.
It is intentionally distinct from the existing midpoint-based sequential
`franka_duo_midpoint_delta14_v1` inference protocol: **do not cumulatively sum
this dataset's deltas across rows**. A future training/inference adapter must
honor the reference state convention, including when building action chunks.

## Dataset CLI

Install the repository's optional `postprocess` dependencies. SciPy is included.
Output paths must be new and separate; source files are never overwritten.

```bash
python -m franka_duo_tele_data.labs_action_delta to-delta \
  --input /path/to/absolute20_dataset \
  --output /path/to/delta14_dataset

python -m franka_duo_tele_data.labs_action_delta to-absolute \
  --input /path/to/delta14_dataset \
  --output /path/to/restored_absolute20_dataset
```

The equivalent installed command is `franka-duo-labs-action-delta`.
Both outputs remain LeRobot v3 datasets. The converter rewrites only action,
updates its feature shape/names and statistics, and verifies every serialized
row. All other data columns are preserved exactly. Videos are reused by hard
link when possible, copied otherwise, with no reencoding. Source metadata and
validation reports are archived in `meta/source_metadata` to avoid presenting
them as validation of the new action representation. The new contract and
all-row round-trip validation are saved in `meta/conversion_manifest.json` and
`meta/action_conversion_validation.json`.

Float32 quantization means restoration is numerically equivalent rather than
bit-for-bit identical for pose values. Every conversion requires round-trip
position and orientation errors below 1e-6 m / 1e-6 rad. Gripper values remain
exactly unchanged. There is no action normalization or clamping.

## Array API

```python
from franka_duo_tele_data.labs_action_delta import (
    absolute20_to_delta14,
    delta14_to_absolute20,
)

delta14 = absolute20_to_delta14(action20, measured_state34)
restored20 = delta14_to_absolute20(delta14, measured_state34)
```

Arrays may be single rows or batches with matching leading dimensions. A
20D pose/gripper reference is also accepted by these array functions; the
dataset CLI requires the labs state34 contract. Grippers are passed through
without differencing or rebinarization.

## Image size and normalization

Width is 640 and height is 480. NumPy RGB shape is `(480, 640, 3)`; the official
LeRobot reader returns `(3, 480, 640)`. These are axis conventions, not swapped
image dimensions.

The dataset writer stores physical state/action values in float32, without
z-score or min-max normalization. Positions/deltas are metres, joint angles
and rotation vectors are radians. Videos store encoded RGB frames; the
LeRobot reader returns floating RGB values in [0,1]. `meta/stats.json` contains
mean/std/min/max for later training normalization, but calculating statistics
does not normalize the stored action/state. Training policy preprocessing
chooses whether/how to normalize using these statistics.

## Exported on 100.90.202.124

Under `/home/agile/work/labs/data/lerobot/`:

- `labs_fr3_link8_columns_20260916`: original absolute20.
- `labs_fr3_link8_delta14_20260916`: same-frame measured-state delta14.
- `labs_fr3_link8_delta14_restored_absolute20_20260916`: restored absolute20.

Each contains 122 episodes and 46,502 rows, with the same task and three
640×480 RGB streams at nominal 30 fps.

All 46,502 rows passed bidirectional conversion checks. Direct comparison of
restored20 against the original dataset found maximum translation error
`1.862645149230957e-09 m` and rotation error `8.588537754633716e-08 rad`.
All non-action columns and both gripper values were exactly preserved.
Official LeRobot 0.4.3 loaded each new dataset's 244 episode boundary samples.

The 14D dataset was packed as:

`/home/agile/work/labs/data/lerobot/labs_fr3_link8_delta14_20260916.tar.zst`

Archive size: 1,025,470,042 bytes. It contains 505 files, including all 366
videos. Zstandard integrity and archive paths/sizes were verified against the
source directory. The adjacent `.tar.zst.sha256` file records:

```text
eba1aeb387dd5c5e0540b25db9fa3e2cc9059f468f6c54f25ffd3dc031867240
```

```bash
sha256sum -c labs_fr3_link8_delta14_20260916.tar.zst.sha256
tar --zstd -xf labs_fr3_link8_delta14_20260916.tar.zst
```
