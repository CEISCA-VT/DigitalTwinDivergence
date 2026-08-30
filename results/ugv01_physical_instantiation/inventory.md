# UGV01 Asset-Specific Instantiation Inventory

This inventory was generated from existing repository artifacts only. No training, retuning, firmware changes, or new data collection were performed.

## Accepted Quantitative Evidence

### strict_headline_fidelity_run
- Source path: `DigitalTwin/datasets/analysis/ugv01_apriltag_finetuned_full_142023_continuity_repaired`
- Available sensors: T:147 encoder counts, yaw/gyro/IMU fields in telemetry; AprilTag rover pose from video; GPS unavailable/disconnected for this run
- Independent reference source: `DigitalTwin/datasets/analysis/validation_carpet_142023_four_tag_continuity_repaired/apriltag_still_summary.json`
- Duration/run support: 2153 samples, selected video interval 0.0-239.3 s
- Surface/operating condition: low-speed indoor carpet, 2.0 m x 1.0 m reference rectangle
- Synchronization method: full_motion_activity_correlation, offset 20.000 s, correlation 0.933
- Calibration parameters: fitted carpet candidate: distance_scale 0.975, clockwise_width_m 0.200, counterclockwise_width_m 0.190, gyro_weight 0.20
- Known reference limitations: no hardware sync pulse; directly decoded fraction 0.281; repaired short gaps; camera jitter p95 0.0039 m
- Validity: yes, current strict development headline; not final publication-standard prospective run
- Notes: Used for final UGV01 asset-instantiation metrics in this audit.

### old_current_same_run_reference
- Source path: `DigitalTwin/datasets/analysis/ugv01_apriltag_old_current_142023`
- Available sensors: same telemetry/reference as strict headline
- Independent reference source: `DigitalTwin/datasets/analysis/ugv01_apriltag_old_current_142023/fidelity_summary.json`
- Duration/run support: 1893 samples across selected usable windows 0.0-162.66 s and 188.66-239.3 s
- Surface/operating condition: low-speed indoor carpet
- Synchronization method: same motion-correlation offset as strict run
- Calibration parameters: current old model: effective_track_width_m 0.192, gyro_weight 0.20 in saved summary
- Known reference limitations: same camera/sync limits; different interval than full repaired headline
- Validity: yes, for paired old-vs-fitted comparison on same recording
- Notes: Best available existing current-model comparator.

### temporal_finetune_calibration
- Source path: `DigitalTwin/datasets/analysis/ugv01_apriltag_finetune_142023/temporal_calibration_summary.json`
- Available sensors: AprilTag trajectory plus T:147 telemetry
- Independent reference source: `DigitalTwin/datasets/analysis/validation_carpet_142023_four_tag_continuity_repaired/apriltag_still_summary.json`
- Duration/run support: 75/25 temporal split within the same recording
- Surface/operating condition: low-speed indoor carpet
- Synchronization method: training-fitted offset 20.000 s
- Calibration parameters: {"clockwise_width_m": 0.20000000000000004, "counterclockwise_width_m": 0.19000000000000003, "distance_scale": 0.975, "gyro_scale": 1.0, "gyro_weight": 0.2}
- Known reference limitations: development calibration, not independent run-level validation
- Validity: yes for development/instantiation effect; no for final generalization claim
- Notes: Records exact fitted stage parameters and temporal holdout diagnostics.

### strict_tracking_source
- Source path: `DigitalTwin/datasets/analysis/validation_carpet_142023_four_tag_continuity_repaired/apriltag_still_summary.json`
- Available sensors: video-derived AprilTag pose: rover tag ID 0 and fixed reference IDs 1,2,3,4
- Independent reference source: `AprilTag/ChArUco camera geometry`
- Duration/run support: 7181 frames at 29.993 fps
- Surface/operating condition: carpet reference rectangle
- Synchronization method: video-only tracking artifact; telemetry sync supplied by fidelity run
- Calibration parameters: world_tags_m={'1': [0.0, 0.0], '2': [2.0, 0.0], '3': [2.0, 1.0], '4': [0.0, 1.0]}; calibration=DigitalTwin\datasets\analysis\camera_calibration_landscape\camera_calibration_charuco.json
- Known reference limitations: uses repaired/continuous tracking and fixed reference geometry
- Validity: yes as the accepted independent reference for the strict run
- Notes: This is the physical trajectory source.

## Supplemental, Excluded, Or Questionable Evidence

### supplemental_turn_calibration
- Source path: `docs/apriltag_turn_calibration.md`
- Reference/condition: docs/footage/footage_trapezoid.mp4 / earlier indoor turn-calibration surface
- Why limited: supplemental; sparse event count and not the strict headline run
- Validity: supplemental only
- Notes: Supports why nominal tracked-turn geometry was insufficient.

### excluded_low_speed_2m
- Source path: `DigitalTwin/datasets/analysis/carpet_low_speed_2m_continuity_repaired`
- Reference/condition: rover tag ID 0 / carpet
- Why limited: rover tag visibility too low
- Validity: no
- Notes: Explicitly excluded by docs/apriltag_validation_split.md.

### excluded_still_footage
- Source path: `docs/footage/still footage.mp4`
- Reference/condition: reference tags / stationary setup
- Why limited: no rover tag ID 0
- Validity: no
- Notes: Useful setup evidence, not a rover trajectory.
