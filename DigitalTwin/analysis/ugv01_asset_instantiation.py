"""UGV01 asset-specific digital-twin instantiation audit.

This post-hoc analysis reads existing UGV01 AprilTag and telemetry-derived
artifacts only. It does not train models, retune parameters, or modify the
frozen i2Nav Twin V2 results.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "ugv01_physical_instantiation"

STRICT_TRACKING = (
    ROOT
    / "DigitalTwin"
    / "datasets"
    / "analysis"
    / "validation_carpet_142023_four_tag_continuity_repaired"
    / "apriltag_still_summary.json"
)
STRICT_TELEMETRY = ROOT / "raw_logs" / "telemetry" / "ugv_t147_interactive_20260812_142023.csv"
SPLIT_DOC = ROOT / "docs" / "apriltag_validation_split.md"
FINETUNE_DOC = ROOT / "docs" / "ugv01_apriltag_finetune_results.md"
TURN_DOC = ROOT / "docs" / "apriltag_turn_calibration.md"
TEMPORAL_CAL = (
    ROOT
    / "DigitalTwin"
    / "datasets"
    / "analysis"
    / "ugv01_apriltag_finetune_142023"
    / "temporal_calibration_summary.json"
)
OLD_CURRENT = ROOT / "DigitalTwin" / "datasets" / "analysis" / "ugv01_apriltag_old_current_142023"
FITTED_STRICT = ROOT / "DigitalTwin" / "datasets" / "analysis" / "ugv01_apriltag_finetuned_full_142023"
FITTED_FULL = (
    ROOT
    / "DigitalTwin"
    / "datasets"
    / "analysis"
    / "ugv01_apriltag_finetuned_full_142023_continuity_repaired"
)
TRAPEZOID_FIDELITY = (
    ROOT
    / "DigitalTwin"
    / "datasets"
    / "analysis"
    / "apriltag_trapezoid_fidelity_calibrated"
)
TRIAL1_SQUARE_FIDELITY = (
    ROOT
    / "DigitalTwin"
    / "datasets"
    / "analysis"
    / "apriltag_trial1_square_1p5_elevation_fidelity"
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, q))


def wrap_deg(angle: np.ndarray) -> np.ndarray:
    return (angle + 180.0) % 360.0 - 180.0


def rpe_rmse(t: np.ndarray, truth_xy: np.ndarray, twin_xy: np.ndarray, horizon_s: float) -> float:
    if t.size < 2 or (t[-1] - t[0]) < horizon_s:
        return math.nan
    errs = []
    j = 0
    for i, ti in enumerate(t):
        target = ti + horizon_s
        while j < t.size and t[j] < target:
            j += 1
        if j >= t.size:
            break
        truth_delta = truth_xy[j] - truth_xy[i]
        twin_delta = twin_xy[j] - twin_xy[i]
        errs.append(float(np.linalg.norm(twin_delta - truth_delta)))
    if not errs:
        return math.nan
    return float(math.sqrt(mean([e * e for e in errs])))


def trajectory_metrics(samples_path: Path) -> dict[str, float]:
    rows = read_csv_rows(samples_path)
    t = np.array([float(r["telemetry_elapsed_s"]) for r in rows], dtype=float)
    truth_xy = np.array([[float(r["truth_x_m"]), float(r["truth_y_m"])] for r in rows], dtype=float)
    twin_xy = np.array([[float(r["twin_x_m"]), float(r["twin_y_m"])] for r in rows], dtype=float)
    truth_heading = np.array([float(r["truth_heading_deg"]) for r in rows], dtype=float)
    twin_heading = np.array([float(r["twin_heading_deg"]) for r in rows], dtype=float)
    dp = np.linalg.norm(twin_xy - truth_xy, axis=1)
    signed_heading = wrap_deg(twin_heading - truth_heading)
    abs_heading = np.abs(signed_heading)

    out = {
        "samples": int(len(rows)),
        "duration_s": float(t[-1] - t[0]) if t.size else math.nan,
        "ate_rmse_m": float(math.sqrt(mean((dp * dp).tolist()))) if dp.size else math.nan,
        "dp_median_m": pct(dp, 50),
        "dp_p95_m": pct(dp, 95),
        "dp_max_m": float(np.max(dp)) if dp.size else math.nan,
        "heading_mae_deg": float(np.mean(abs_heading)) if abs_heading.size else math.nan,
        "dtheta_p95_deg": pct(abs_heading, 95),
        "dtheta_max_deg": float(np.max(abs_heading)) if abs_heading.size else math.nan,
        "persistent_signed_yaw_residual_deg": float(np.mean(signed_heading))
        if signed_heading.size
        else math.nan,
        "rpe1_rmse_m": rpe_rmse(t, truth_xy, twin_xy, 1.0),
        "rpe5_rmse_m": rpe_rmse(t, truth_xy, twin_xy, 5.0),
        "rpe10_rmse_m": rpe_rmse(t, truth_xy, twin_xy, 10.0),
    }

    if t.size > 2:
        dt = np.diff(t)
        good = dt > 0
        truth_step = np.linalg.norm(np.diff(truth_xy, axis=0), axis=1)
        twin_step = np.linalg.norm(np.diff(twin_xy, axis=0), axis=1)
        truth_speed = truth_step[good] / dt[good]
        twin_speed = twin_step[good] / dt[good]
        truth_yaw = np.unwrap(np.deg2rad(truth_heading))
        twin_yaw = np.unwrap(np.deg2rad(twin_heading))
        yaw_residual_rate = (np.diff(twin_yaw) - np.diff(truth_yaw))[good] / dt[good]
        out.update(
            {
                "velocity_disagreement_mean_mps": float(np.mean(np.abs(twin_speed - truth_speed))),
                "velocity_disagreement_p95_mps": pct(np.abs(twin_speed - truth_speed), 95),
                "yaw_rate_disagreement_mean_degps": float(
                    np.mean(np.abs(np.rad2deg(yaw_residual_rate)))
                ),
                "yaw_rate_disagreement_p95_degps": pct(np.abs(np.rad2deg(yaw_residual_rate)), 95),
                "iomega_abs_accumulated_deg": float(
                    np.sum(np.abs(np.rad2deg(yaw_residual_rate)) * dt[good])
                ),
            }
        )
    else:
        out.update(
            {
                "velocity_disagreement_mean_mps": math.nan,
                "velocity_disagreement_p95_mps": math.nan,
                "yaw_rate_disagreement_mean_degps": math.nan,
                "yaw_rate_disagreement_p95_degps": math.nan,
                "iomega_abs_accumulated_deg": math.nan,
            }
        )
    return out


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def build_inventory() -> list[dict[str, str]]:
    strict_summary = read_json(FITTED_FULL / "fidelity_summary.json")
    tracking_summary = read_json(STRICT_TRACKING)
    temporal = read_json(TEMPORAL_CAL)
    return [
        {
            "artifact": "strict_headline_fidelity_run",
            "source_path": rel(FITTED_FULL),
            "available_sensors": "T:147 encoder counts, yaw/gyro/IMU fields in telemetry; AprilTag rover pose from video; GPS unavailable/disconnected for this run",
            "independent_reference_source": rel(STRICT_TRACKING),
            "duration_or_samples": f"{strict_summary['evaluation_samples_approximately_10hz']} samples, selected video interval 0.0-239.3 s",
            "surface_condition": "low-speed indoor carpet, 2.0 m x 1.0 m reference rectangle",
            "synchronization_method": f"{strict_summary['synchronization_method']}, offset {strict_summary['estimated_video_minus_telemetry_offset_s']:.3f} s, correlation {strict_summary['synchronization_correlation']:.3f}",
            "calibration_parameters": "fitted carpet candidate: distance_scale 0.975, clockwise_width_m 0.200, counterclockwise_width_m 0.190, gyro_weight 0.20",
            "reference_limitations": f"no hardware sync pulse; directly decoded fraction {strict_summary['directly_decoded_evaluation_fraction']:.3f}; repaired short gaps; camera jitter p95 {strict_summary['camera_stationary_jitter_p95_m']:.4f} m",
            "valid_for_quantitative_fidelity": "yes, current strict development headline; not final publication-standard prospective run",
            "notes": "Used for final UGV01 asset-instantiation metrics in this audit.",
        },
        {
            "artifact": "old_current_same_run_reference",
            "source_path": rel(OLD_CURRENT),
            "available_sensors": "same telemetry/reference as strict headline",
            "independent_reference_source": rel(OLD_CURRENT / "fidelity_summary.json"),
            "duration_or_samples": "1893 samples across selected usable windows 0.0-162.66 s and 188.66-239.3 s",
            "surface_condition": "low-speed indoor carpet",
            "synchronization_method": "same motion-correlation offset as strict run",
            "calibration_parameters": "current old model: effective_track_width_m 0.192, gyro_weight 0.20 in saved summary",
            "reference_limitations": "same camera/sync limits; different interval than full repaired headline",
            "valid_for_quantitative_fidelity": "yes, for paired old-vs-fitted comparison on same recording",
            "notes": "Best available existing current-model comparator.",
        },
        {
            "artifact": "temporal_finetune_calibration",
            "source_path": rel(TEMPORAL_CAL),
            "available_sensors": "AprilTag trajectory plus T:147 telemetry",
            "independent_reference_source": rel(STRICT_TRACKING),
            "duration_or_samples": "75/25 temporal split within the same recording",
            "surface_condition": "low-speed indoor carpet",
            "synchronization_method": f"training-fitted offset {temporal['split_policy']['ugv01_carpet_142023']['synchronization_offset_s_fitted_on_training']:.3f} s",
            "calibration_parameters": json.dumps(temporal["fitted_parameters"], sort_keys=True),
            "reference_limitations": "development calibration, not independent run-level validation",
            "valid_for_quantitative_fidelity": "yes for development/instantiation effect; no for final generalization claim",
            "notes": "Records exact fitted stage parameters and temporal holdout diagnostics.",
        },
        {
            "artifact": "strict_tracking_source",
            "source_path": rel(STRICT_TRACKING),
            "available_sensors": "video-derived AprilTag pose: rover tag ID 0 and fixed reference IDs 1,2,3,4",
            "independent_reference_source": "AprilTag/ChArUco camera geometry",
            "duration_or_samples": f"{tracking_summary['video']['frame_count']} frames at {tracking_summary['video']['fps']:.3f} fps",
            "surface_condition": "carpet reference rectangle",
            "synchronization_method": "video-only tracking artifact; telemetry sync supplied by fidelity run",
            "calibration_parameters": f"world_tags_m={tracking_summary['world_tags_m']}; calibration={tracking_summary['calibration_path']}",
            "reference_limitations": "uses repaired/continuous tracking and fixed reference geometry",
            "valid_for_quantitative_fidelity": "yes as the accepted independent reference for the strict run",
            "notes": "This is the physical trajectory source.",
        },
        {
            "artifact": "supplemental_turn_calibration",
            "source_path": rel(TURN_DOC),
            "available_sensors": "AprilTag video plus T:147 telemetry",
            "independent_reference_source": "docs/footage/footage_trapezoid.mp4",
            "duration_or_samples": "event-level short turns",
            "surface_condition": "earlier indoor turn-calibration surface",
            "synchronization_method": "activity-derived offset -10.10 +/- 0.28 s",
            "calibration_parameters": "nominal width 0.141 m; effective tracked-turn width 0.192 m; signs -1/-1",
            "reference_limitations": "supplemental; sparse event count and not the strict headline run",
            "valid_for_quantitative_fidelity": "supplemental only",
            "notes": "Supports why nominal tracked-turn geometry was insufficient.",
        },
        {
            "artifact": "excluded_low_speed_2m",
            "source_path": "DigitalTwin/datasets/analysis/carpet_low_speed_2m_continuity_repaired",
            "available_sensors": "AprilTag video",
            "independent_reference_source": "rover tag ID 0",
            "duration_or_samples": "13696 frames; only 15 valid before repair and 505 after repair",
            "surface_condition": "carpet",
            "synchronization_method": "not used",
            "calibration_parameters": "not applicable",
            "reference_limitations": "rover tag visibility too low",
            "valid_for_quantitative_fidelity": "no",
            "notes": "Explicitly excluded by docs/apriltag_validation_split.md.",
        },
        {
            "artifact": "excluded_still_footage",
            "source_path": "docs/footage/still footage.mp4",
            "available_sensors": "fixed reference AprilTags only",
            "independent_reference_source": "reference tags",
            "duration_or_samples": "setup/reference video",
            "surface_condition": "stationary setup",
            "synchronization_method": "not applicable",
            "calibration_parameters": "not applicable",
            "reference_limitations": "no rover tag ID 0",
            "valid_for_quantitative_fidelity": "no",
            "notes": "Useful setup evidence, not a rover trajectory.",
        },
    ]


def build_stages() -> list[dict]:
    temporal = read_json(TEMPORAL_CAL)
    return [
        {
            "stage_id": "S0",
            "stage_name": "generic_nominal_vendor_template",
            "status": "supplemental_only",
            "source_artifact": rel(TURN_DOC),
            "changes_from_previous": "None; uses vendor/nominal tracked geometry before asset binding.",
            "parameters": {
                "nominal_physical_width_m": 0.141,
                "distance_scale": 1.0,
                "direct_gyro_fusion_weight": 0.0,
            },
            "evaluation_note": "Only available on supplemental turn-calibration evidence, not on the strict 142023 headline run.",
        },
        {
            "stage_id": "S1",
            "stage_name": "ugv01_frame_sensor_adapter_current",
            "status": "accepted_comparator",
            "source_artifact": rel(OLD_CURRENT),
            "changes_from_previous": "Binds telemetry conventions, encoder signs, AprilTag frame, video/telemetry timing, and current UGV01 effective turn width.",
            "parameters": read_json(OLD_CURRENT / "fidelity_summary.json")
            | {"distance_scale": 1.0, "clockwise_width_m": 0.192, "counterclockwise_width_m": 0.192},
            "evaluation_note": "Valid paired comparator on the same physical run and strict usable windows.",
        },
        {
            "stage_id": "S2",
            "stage_name": "ugv01_carpet_asset_specific_fitted",
            "status": "accepted_development_instantiation",
            "source_artifact": rel(FITTED_STRICT),
            "changes_from_previous": "Adds UGV01 carpet-specific distance scale, asymmetric effective track widths, and bounded gyro contribution selected on training portion.",
            "parameters": temporal["fitted_parameters"],
            "evaluation_note": "Evaluated on same strict usable windows and temporal holdout; not an independent final run.",
        },
        {
            "stage_id": "S3",
            "stage_name": "ugv01_carpet_asset_specific_full_repaired",
            "status": "current_headline_low_speed_carpet",
            "source_artifact": rel(FITTED_FULL),
            "changes_from_previous": "Applies the asset-specific candidate to the repaired full 142023 window.",
            "parameters": temporal["fitted_parameters"],
            "evaluation_note": "Current headline UGV01 fidelity result for low-speed carpet development validation.",
        },
    ]


def build_stage_metrics() -> list[dict]:
    rows = []
    specs = [
        ("S1", "ugv01_frame_sensor_adapter_current", OLD_CURRENT),
        ("S2", "ugv01_carpet_asset_specific_fitted", FITTED_STRICT),
        ("S3", "ugv01_carpet_asset_specific_full_repaired", FITTED_FULL),
    ]
    for stage_id, stage_name, folder in specs:
        summary = read_json(folder / "fidelity_summary.json")
        metrics = trajectory_metrics(folder / "aligned_fidelity_samples.csv")
        # Use the previously generated UGV01 fidelity summary as the source of
        # truth for headline metrics; compute only diagnostics that were not
        # already saved there, such as RPE5/RPE10 and residual-rate summaries.
        metrics.update(
            {
                "ate_rmse_m": summary["position_ate_rmse_m"],
                "dp_median_m": summary["position_error_median_m"],
                "dp_p95_m": summary["position_error_p95_m"],
                "dp_max_m": summary["position_error_max_m"],
                "heading_mae_deg": summary["heading_mae_deg"],
                "dtheta_p95_deg": summary["heading_p95_deg"],
                "rpe1_rmse_m": summary["rpe_1s_rmse_m"],
            }
        )
        rows.append(
            {
                "stage_id": stage_id,
                "stage_name": stage_name,
                "source_path": rel(folder),
                "reference": rel(Path(summary["tracking_source"])),
                "telemetry": summary["telemetry_source"],
                "surface_condition": "low-speed carpet",
                "sync_method": summary["synchronization_method"],
                "sync_correlation": summary["synchronization_correlation"],
                "sync_uncertainty_s": summary["synchronization_uncertainty_s"],
                "directly_decoded_fraction": summary["directly_decoded_evaluation_fraction"],
                **metrics,
                "truth_path_length_m": summary["per_interval"][0]["truth_path_length_m"]
                if len(summary.get("per_interval", [])) == 1
                else sum(p["truth_path_length_m"] for p in summary.get("per_interval", [])),
                "estimated_path_length_m": summary["per_interval"][0]["estimated_path_length_m"]
                if len(summary.get("per_interval", [])) == 1
                else sum(p["estimated_path_length_m"] for p in summary.get("per_interval", [])),
            }
        )
    return rows


def write_inventory_markdown(rows: list[dict[str, str]]) -> None:
    lines = [
        "# UGV01 Asset-Specific Instantiation Inventory",
        "",
        "This inventory was generated from existing repository artifacts only. No training, retuning, firmware changes, or new data collection were performed.",
        "",
        "## Accepted Quantitative Evidence",
        "",
    ]
    for row in rows:
        if str(row["valid_for_quantitative_fidelity"]).startswith("yes") or "accepted" in str(row["valid_for_quantitative_fidelity"]):
            lines.extend(
                [
                    f"### {row['artifact']}",
                    f"- Source path: `{row['source_path']}`",
                    f"- Available sensors: {row['available_sensors']}",
                    f"- Independent reference source: `{row['independent_reference_source']}`",
                    f"- Duration/run support: {row['duration_or_samples']}",
                    f"- Surface/operating condition: {row['surface_condition']}",
                    f"- Synchronization method: {row['synchronization_method']}",
                    f"- Calibration parameters: {row['calibration_parameters']}",
                    f"- Known reference limitations: {row['reference_limitations']}",
                    f"- Validity: {row['valid_for_quantitative_fidelity']}",
                    f"- Notes: {row['notes']}",
                    "",
                ]
            )
    lines.extend(["## Supplemental, Excluded, Or Questionable Evidence", ""])
    for row in rows:
        if not (str(row["valid_for_quantitative_fidelity"]).startswith("yes") or "accepted" in str(row["valid_for_quantitative_fidelity"])):
            lines.extend(
                [
                    f"### {row['artifact']}",
                    f"- Source path: `{row['source_path']}`",
                    f"- Reference/condition: {row['independent_reference_source']} / {row['surface_condition']}",
                    f"- Why limited: {row['reference_limitations']}",
                    f"- Validity: {row['valid_for_quantitative_fidelity']}",
                    f"- Notes: {row['notes']}",
                    "",
                ]
            )
    (OUT_DIR / "inventory.md").write_text("\n".join(lines), encoding="utf-8")


def write_reference_uncertainty(stage_rows: list[dict]) -> None:
    headline = read_json(FITTED_FULL / "fidelity_summary.json")
    text = f"""# UGV01 AprilTag Reference Uncertainty

The accepted UGV01 reference is the repaired AprilTag trajectory in
`{rel(STRICT_TRACKING)}` paired with telemetry `{rel(STRICT_TELEMETRY)}`.

Known uncertainty sources:

- Camera/telemetry synchronization is motion-correlation based, not a hardware-visible sync pulse.
- Estimated video-minus-telemetry offset: `{headline['estimated_video_minus_telemetry_offset_s']:.3f} s`.
- Synchronization correlation: `{headline['synchronization_correlation']:.3f}`.
- Synchronization uncertainty: `{headline['synchronization_uncertainty_s']:.3f} s`.
- Directly decoded evaluation fraction: `{headline['directly_decoded_evaluation_fraction']:.3f}`.
- Recovered evaluation samples: `{headline['recovered_evaluation_samples']}` of `{headline['evaluation_samples_approximately_10hz']}`.
- Camera stationary jitter RMSE / p95: `{headline['camera_stationary_jitter_rmse_m']:.4f} m / {headline['camera_stationary_jitter_p95_m']:.4f} m`.
- The accepted docs also record a measured reference-geometry/camera-model reprojection discrepancy on the order of several pixels for the development setup.

Interpretation:

The reference is strong enough for an asset-instantiation development audit on
low-speed carpet, because the observed improvement in heading and full-window
path consistency is much larger than camera stationary jitter. Smaller position
changes of only a few millimeters to centimeters should not be overstated,
because they can be comparable to synchronization, reference-geometry, and
recovered-track uncertainty. The current reference supports a low-speed carpet
UGV01 claim, not a general all-surface/all-speed claim.
"""
    (OUT_DIR / "ugv01_reference_uncertainty.md").write_text(text, encoding="utf-8")


def write_claim_matrix() -> None:
    rows = [
        {
            "claim": "A UGV01-specific instantiation improves the carpet pilot trajectory relative to the current UGV01 model.",
            "current_evidence": "Supported on the 142023 AprilTag/T:147 pilot; strict-window ATE 0.131->0.099 m, RPE1 0.035->0.028 m, heading MAE 13.3->5.6 deg.",
            "support_level": "development-supported",
            "minimum_new_experiment_if_stronger_claim_needed": "One untouched synchronized carpet run with deliberate sync event.",
        },
        {
            "claim": "The resulting model is a defensible twin of this specific UGV01 under low-speed carpet operation.",
            "current_evidence": "Supported with qualifications: full repaired headline ATE 0.114 m, RPE1 0.030 m, heading MAE 6.7 deg, path ratio 1.021.",
            "support_level": "qualified",
            "minimum_new_experiment_if_stronger_claim_needed": "Repeat low-speed carpet run to separate asset behavior from session-specific calibration.",
        },
        {
            "claim": "The calibrated asset twin generalizes across surfaces.",
            "current_evidence": "Not supported by accepted independent reference; rough/smooth telemetry exists but lacks accepted AprilTag ground truth.",
            "support_level": "unsupported",
            "minimum_new_experiment_if_stronger_claim_needed": "One second-surface AprilTag run, e.g. smooth floor or rougher indoor surface, with same sync protocol.",
        },
        {
            "claim": "The calibrated asset twin remains accurate at higher speed.",
            "current_evidence": "Not supported by strict low-speed carpet reference.",
            "support_level": "unsupported",
            "minimum_new_experiment_if_stronger_claim_needed": "One higher-speed AprilTag run with straight, turns, and curves.",
        },
        {
            "claim": "The model handles sustained turning/slip conditions.",
            "current_evidence": "Partially supported by supplemental turn calibration and diagnostics, but not final validation.",
            "support_level": "partial",
            "minimum_new_experiment_if_stronger_claim_needed": "Targeted figure-eight or repeated sustained-turn AprilTag run.",
        },
        {
            "claim": "GPS is required for UGV01 fidelity validation.",
            "current_evidence": "Not required for the current physical-virtual fidelity audit because AprilTags provide independent pose reference and the evaluated twin is prediction-only.",
            "support_level": "not required for this claim",
            "minimum_new_experiment_if_stronger_claim_needed": "Only needed for later GPS-sensor-trust/security claims, not for this asset-instantiation audit.",
        },
    ]
    write_csv(OUT_DIR / "ugv01_claim_evidence_matrix.csv", rows)


def build_existing_condition_comparison() -> list[dict]:
    specs = [
        {
            "condition_id": "carpet_strict_headline",
            "surface_condition": "low-speed carpet",
            "route_or_motion": "2.0 m x 1.0 m reference area, mixed low-speed motion",
            "folder": FITTED_FULL,
            "evidence_role": "strict headline",
            "surface_confidence": "explicit in filename/docs",
        },
        {
            "condition_id": "smooth_floor_trapezoid",
            "surface_condition": "smooth/kitchen-floor-like indoor surface",
            "route_or_motion": "trapezoid / turn-calibration motion",
            "folder": TRAPEZOID_FIDELITY,
            "evidence_role": "supplemental condition evidence",
            "surface_confidence": "docs/apriltag_turn_calibration.md calls this the recorded kitchen-floor run",
        },
        {
            "condition_id": "smooth_floor_square_1p5",
            "surface_condition": "indoor smooth-floor square run, exact surface label not embedded in artifact",
            "route_or_motion": "1.5 m square",
            "folder": TRIAL1_SQUARE_FIDELITY,
            "evidence_role": "supplemental condition evidence",
            "surface_confidence": "video/analysis artifact exists, but surface is not as explicitly named as the carpet run",
        },
    ]
    rows = []
    for spec in specs:
        summary = read_json(spec["folder"] / "fidelity_summary.json")
        rows.append(
            {
                "condition_id": spec["condition_id"],
                "surface_condition": spec["surface_condition"],
                "route_or_motion": spec["route_or_motion"],
                "evidence_role": spec["evidence_role"],
                "surface_confidence": spec["surface_confidence"],
                "source_path": rel(spec["folder"]),
                "tracking_source": summary["tracking_source"],
                "telemetry_source": summary["telemetry_source"],
                "samples": summary["evaluation_samples_approximately_10hz"],
                "directly_decoded_fraction": summary["directly_decoded_evaluation_fraction"],
                "sync_method": summary["synchronization_method"],
                "sync_correlation": summary["synchronization_correlation"],
                "sync_uncertainty_s": summary["synchronization_uncertainty_s"],
                "ate_rmse_m": summary["position_ate_rmse_m"],
                "position_median_m": summary["position_error_median_m"],
                "position_p95_m": summary["position_error_p95_m"],
                "position_max_m": summary["position_error_max_m"],
                "rpe1_rmse_m": summary["rpe_1s_rmse_m"],
                "heading_mae_deg": summary["heading_mae_deg"],
                "heading_p95_deg": summary["heading_p95_deg"],
                "truth_path_length_m": sum(p["truth_path_length_m"] for p in summary["per_interval"]),
                "estimated_path_length_m": sum(p["estimated_path_length_m"] for p in summary["per_interval"]),
                "path_length_ratio": (
                    sum(p["estimated_path_length_m"] for p in summary["per_interval"])
                    / sum(p["truth_path_length_m"] for p in summary["per_interval"])
                ),
                "interpretation_limit": summary["interpretation"],
            }
        )
    return rows


def write_condition_comparison(rows: list[dict]) -> None:
    write_csv(OUT_DIR / "ugv01_existing_apriltag_condition_comparison.csv", rows)

    labels = [r["condition_id"].replace("_", "\n") for r in rows]
    metrics = [
        ("ate_rmse_m", "ATE RMSE (m)"),
        ("rpe1_rmse_m", "RPE1 RMSE (m)"),
        ("heading_mae_deg", "Heading MAE (deg)"),
        ("position_p95_m", "Position p95 (m)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))
    colors = ["#1f5b89", "#2279b5", "#9aa7b5"]
    for ax, (key, title) in zip(axes, metrics):
        vals = [float(r[key]) for r in rows]
        ax.bar(range(len(rows)), vals, color=colors)
        ax.set_title(title)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Existing UGV01 AprilTag Conditions")
    fig.text(
        0.5,
        0.01,
        "Carpet is the strict headline run. Trapezoid and 1.5 m square are supplemental smooth-floor condition evidence.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(OUT_DIR / "ugv01_existing_apriltag_condition_comparison.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Existing UGV01 AprilTag Condition Comparison",
        "",
        "The repo does contain non-carpet AprilTag fidelity artifacts. They were not part of the strict headline split, but they are useful as supplemental evidence for a condition-dependent UGV01 discussion.",
        "",
        "| Condition | Role | ATE RMSE | RPE1 RMSE | Heading MAE | p95 position | Sync correlation | Main limitation |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['condition_id']} | {r['evidence_role']} | {float(r['ate_rmse_m']):.3f} m | "
            f"{float(r['rpe1_rmse_m']):.3f} m | {float(r['heading_mae_deg']):.1f} deg | "
            f"{float(r['position_p95_m']):.3f} m | {float(r['sync_correlation']):.3f} | "
            f"{r['surface_confidence']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- The trapezoid run is the best existing smooth/kitchen-floor-like supplemental condition: ATE and RPE1 are close to the carpet headline, but heading error is worse and sync uncertainty is larger.",
            "- The 1.5 m square run shows much larger global/tail error despite reasonable short-horizon RPE, making it useful for the local-versus-global fidelity story.",
            "- These runs can support a cautious UGV01 condition-dependent discussion now, but they should be labeled supplemental unless the manuscript explicitly accepts motion-correlation sync and repaired tracks as sufficient.",
        ]
    )
    (OUT_DIR / "ugv01_existing_apriltag_condition_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def plot_comparison(rows: list[dict]) -> None:
    plot_rows = [r for r in rows if r["stage_id"] in {"S1", "S2", "S3"}]
    labels = [r["stage_id"] for r in plot_rows]
    metrics = [
        ("ate_rmse_m", "ATE RMSE (m)"),
        ("rpe1_rmse_m", "RPE1 RMSE (m)"),
        ("heading_mae_deg", "Heading MAE (deg)"),
        ("dp_p95_m", "Position p95 (m)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    colors = ["#9aa7b5", "#2279b5", "#1f5b89"]
    for ax, (key, title) in zip(axes, metrics):
        vals = [float(r[key]) for r in plot_rows]
        ax.bar(labels, vals, color=colors[: len(vals)])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("UGV01 Asset-Specific Digital Twin Instantiation")
    fig.text(
        0.5,
        0.01,
        "S1=current UGV01 model; S2=asset-specific fitted strict windows; S3=asset-specific full repaired headline.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(OUT_DIR / "ugv01_instantiation_comparison.png", dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict]) -> None:
    s1 = next(r for r in rows if r["stage_id"] == "S1")
    s2 = next(r for r in rows if r["stage_id"] == "S2")
    s3 = next(r for r in rows if r["stage_id"] == "S3")

    def change(a: float, b: float) -> float:
        return 100.0 * (float(b) - float(a)) / float(a)

    text = f"""# UGV01 Asset-Specific Digital Twin Instantiation Summary

## Scientific Answer

Existing artifacts support a qualified asset-specific instantiation result:
binding the generic/current rover representation to this physical UGV01 and
calibrating low-speed carpet motion improves physical-virtual agreement on the
recorded AprilTag pilot. The strongest same-window comparison is:

| Metric | Current UGV01 model | Asset-specific fitted | Change |
|---|---:|---:|---:|
| ATE RMSE | {float(s1['ate_rmse_m']):.3f} m | {float(s2['ate_rmse_m']):.3f} m | {change(s1['ate_rmse_m'], s2['ate_rmse_m']):.1f}% |
| RPE1 RMSE | {float(s1['rpe1_rmse_m']):.3f} m | {float(s2['rpe1_rmse_m']):.3f} m | {change(s1['rpe1_rmse_m'], s2['rpe1_rmse_m']):.1f}% |
| Heading MAE | {float(s1['heading_mae_deg']):.1f} deg | {float(s2['heading_mae_deg']):.1f} deg | {change(s1['heading_mae_deg'], s2['heading_mae_deg']):.1f}% |

The current full repaired headline for the fitted asset-specific twin is:

| Metric | Value |
|---|---:|
| ATE RMSE | {float(s3['ate_rmse_m']):.3f} m |
| RPE1 RMSE | {float(s3['rpe1_rmse_m']):.3f} m |
| RPE5 RMSE | {float(s3['rpe5_rmse_m']):.3f} m |
| RPE10 RMSE | {float(s3['rpe10_rmse_m']):.3f} m |
| Heading MAE | {float(s3['heading_mae_deg']):.1f} deg |
| Position p95 / max | {float(s3['dp_p95_m']):.3f} / {float(s3['dp_max_m']):.3f} m |
| Heading p95 / max | {float(s3['dtheta_p95_deg']):.1f} / {float(s3['dtheta_max_deg']):.1f} deg |

## What Changed Between Stages

- The current comparator already has a UGV01 frame/sensor adapter, encoder signs,
  AprilTag frame alignment, and motion-correlation timing.
- The asset-specific fitted stage adds a carpet-specific distance scale, asymmetric
  effective track widths, and bounded gyro contribution recorded in
  `DigitalTwin/datasets/analysis/ugv01_apriltag_finetune_142023/temporal_calibration_summary.json`.
- The full repaired headline applies that candidate to the complete repaired
  142023 window.

## Interpretation

UGV01-specific calibration measurably improves the same-window development result,
especially heading and short-horizon path shape. The remaining weak point is that
this is one low-speed carpet recording with motion-correlation synchronization and
partly recovered AprilTag samples. The model is defensibly a twin of this specific
UGV01 only for the tested low-speed carpet condition. It is not yet evidence for
all surfaces, higher speeds, or sustained slip-heavy motion.

GPS is not required for this particular fidelity audit because the independent
reference is AprilTag pose. GPS would become necessary for later GPS-trust or
security claims, not for proving physical-virtual agreement in this run.

## Is New Physical Collection Scientifically Necessary?

New data are not necessary to report a careful existing-data asset-instantiation
analysis. New data are necessary for stronger claims:

- An untouched synchronized repeat enables a publication-grade held-out UGV01
  fidelity claim.
- A second surface enables cross-surface fidelity claims.
- A higher-speed run enables speed-regime claims.
- A sustained-turn or figure-eight run enables turn/slip robustness claims.

The minimum next experiment is therefore one clean synchronized UGV01 AprilTag
run with a deliberate visible sync event; add only one extra targeted condition
if the paper needs a generalization claim.
"""
    (OUT_DIR / "ugv01_asset_instantiation_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory()
    write_csv(OUT_DIR / "ugv01_asset_inventory.csv", inventory)
    write_inventory_markdown(inventory)

    stages = build_stages()
    (OUT_DIR / "ugv01_instantiation_stages.json").write_text(
        json.dumps({"schema": "ugv01_instantiation_stages_v1", "stages": stages}, indent=2),
        encoding="utf-8",
    )

    stage_rows = build_stage_metrics()
    write_csv(OUT_DIR / "ugv01_fidelity_by_stage.csv", stage_rows)
    plot_comparison(stage_rows)
    condition_rows = build_existing_condition_comparison()
    write_condition_comparison(condition_rows)
    write_reference_uncertainty(stage_rows)
    write_claim_matrix()
    write_summary(stage_rows)

    manifest = {
        "schema": "ugv01_asset_instantiation_audit_v1",
        "no_training_or_tuning_performed": True,
        "source_artifacts": {
            "strict_tracking": rel(STRICT_TRACKING),
            "strict_telemetry": rel(STRICT_TELEMETRY),
            "old_current": rel(OLD_CURRENT),
            "fitted_strict": rel(FITTED_STRICT),
            "fitted_full": rel(FITTED_FULL),
            "temporal_calibration": rel(TEMPORAL_CAL),
            "split_doc": rel(SPLIT_DOC),
            "finetune_doc": rel(FINETUNE_DOC),
            "turn_doc": rel(TURN_DOC),
        },
        "outputs": [
            "inventory.md",
            "ugv01_asset_inventory.csv",
            "ugv01_instantiation_stages.json",
            "ugv01_fidelity_by_stage.csv",
            "ugv01_instantiation_comparison.png",
            "ugv01_reference_uncertainty.md",
            "ugv01_claim_evidence_matrix.csv",
            "ugv01_asset_instantiation_summary.md",
            "ugv01_existing_apriltag_condition_comparison.csv",
            "ugv01_existing_apriltag_condition_comparison.md",
            "ugv01_existing_apriltag_condition_comparison.png",
        ],
    }
    (OUT_DIR / "ugv01_asset_instantiation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": rel(OUT_DIR), "stage_rows": len(stage_rows)}, indent=2))


if __name__ == "__main__":
    main()
