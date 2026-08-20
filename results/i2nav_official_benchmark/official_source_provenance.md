# Official Source Provenance

Date: 2026-08-20

No frozen V2 benchmark trajectories were exported or scored in this task.

## Authoritative Sources Cloned

| Source | URL | Commit | Evidence path(s) | License |
|---|---|---:|---|---|
| i2Nav-Robot | `https://github.com/i2Nav-WHU/i2Nav-Robot.git` | `2ffdca6d56e6432d4daf27c070e570171548f32d` | `README.md`, `LICENSE` | GPLv3; README states academic use |
| evaluate_odometry | `https://github.com/i2Nav-WHU/evaluate_odometry.git` | `6553b1a1d1de79ea50686664ceaacfa02d4f2fe1` | `README.md`, `evaluate.py` | No LICENSE file present in clone |
| LE-VINS | `https://github.com/i2Nav-WHU/LE-VINS.git` | `ad0b395f5675d17a35ed1a6386f411fd1d61ee19` | `README.md`, `LICENSE` | GPLv3 |

## Source Evidence

`i2Nav-Robot/README.md` verifies:

- `*_groundtruth.nav`: local NED position/velocity/attitude.
- `*_trajectory.csv`: ground-truth trajectory in TUM format with local NED position and quaternion `xyzw`.
- Raw text timestamps are GNSS seconds-of-week.
- The dataset is GPLv3 and for academic use.

`LE-VINS/README.md` verifies i2Nav group evaluation practice:

- TUM trajectory files are evaluated using `evo`.
- The group provides `evaluate_odometry` scripts for evaluation.
- LE-VINS expects IMU data in front-right-down format.

`evaluate_odometry/evaluate.py` verifies evaluator mechanics:

- `MAX_TIME_SYNC_DIFF = 0.005`.
- `RPE_DELTA = [50, 100, 150, 200, 250, 300]`.
- `IS_RPE_ALL_PAIRS = True`.
- APE translation uses `metrics.PoseRelation.translation_part`.
- APE rotation uses `metrics.PoseRelation.rotation_angle_deg`.
- RPE uses `metrics.Unit.meters`.
- RPE relative delta tolerance is `0.002`.
- Trajectory association uses `sync.associate_trajectories`.
- Alignment uses `traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)`.

## Dependency Setup

The active repo interpreter is:

```text
C:\Python313\python.exe
```

Installed/verified:

```text
evo v1.37.0
numpy-quaternion 2024.0.13
```

Synthetic sanity test:

```text
associated poses: 401
APE translation RMSE: 0.0 m
APE rotation RMSE: 0.0 deg
RPE 50 m translation RMSE: 0.0 m
RPE 50 m rotation RMSE: 0.0 deg
```

The sanity test used a non-collinear synthetic TUM trajectory and an offset copy. The zero result after SE(3) alignment confirms that the official evaluator mechanics import and run correctly.

## Verification Status

The metric and alignment protocol is sufficiently verified for a later frozen V2 official-style evaluation:

```text
READY_FOR_FROZEN_V2_OFFICIAL_EVALUATION
```

The next task should export the already frozen V2 trajectories and run the verified evaluator. It should not retrain, retune, change checkpoints, or use benchmark results to modify V2.
