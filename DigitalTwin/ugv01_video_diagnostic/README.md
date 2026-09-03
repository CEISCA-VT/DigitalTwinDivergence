# UGV01 smooth-floor video diagnostic

This is a fast, directory-wide screening tool for the prospective UGV01
AprilTag validation footage. It answers whether each recording is technically
usable for trajectory reconstruction; it does not claim that the digital twin
passed its fidelity contracts.

The default protocol expects five **separate** recordings. Thirteen minutes in
one file is still one physical run. Development/calibration recordings should
not be mixed with the final held-out run directory.

## Install

From the repository root:

```powershell
py -m venv .venv-video-audit
.venv-video-audit\Scripts\python -m pip install -r ugv01_video_diagnostic\requirements.txt
```

## Run (all videos in a directory)

The easiest Windows command creates the isolated environment automatically:

```powershell
.\ugv01_video_diagnostic\run_ugv01_video_audit.ps1 -VideoDirectory "C:\path\to\smooth_floor_videos"
```

Or invoke Python directly:

```powershell
.venv-video-audit\Scripts\python ugv01_video_diagnostic\audit_videos.py "C:\path\to\smooth_floor_videos" --output results\ugv01_smooth_video_audit --moving-tag-ids 0 --expected-runs 5
```

For a faster first pass, use `--sample-hz 2`. The default 4 Hz is preferable
for checking whether occlusion gaps are short enough for the 1-second service
horizon. Use `--workers 1` if parallel video decoding overloads a slow disk.

If ID 0 is attached to both the front and rear of the rover, the program flags
frames in which both copies are visible simultaneously. Two physical tags with
one ID cannot be uniquely distinguished and should not both be used in a single
pose estimate.

## Optional configuration

Copy `config.example.json`, edit it, and pass `--config my_config.json`.
Command-line values override the JSON. List known fixed floor/reference tag IDs
under `reference_tag_ids`; do not put the rover's moving ID there.

The output directory contains:

- `video_audit_report.md`: human-readable decision and reasons
- `video_audit_videos.csv`: one row per video
- `video_audit_summary.json`: complete machine-readable results

Interpretation:

- `READY_FOR_ANALYSIS`: at least five separate videos pass all footage gates.
- `USABLE_WITH_WARNINGS`: the files may be recoverable, but warnings must be
  reviewed before publication analysis.
- `NOT_READY`: too few independent usable runs or a critical footage failure.

The audit checks decoding, duration, resolution, brightness, blur, moving-tag
coverage, tag pixel size, longest tag-loss gap, and duplicate moving-tag IDs.
Metric ground truth additionally requires frozen camera calibration, measured
tag size/mounting geometry, timestamps/synchronization, and the corresponding
UGV telemetry logs; those cannot be proven from video pixels alone.
