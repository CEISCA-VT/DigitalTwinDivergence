"""Rescore saved TerraSentia traces after correcting reference heading interpolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from DigitalTwin.analysis import aifarms_terrasentia_full_study as study
from DigitalTwin.analysis import aifarms_terrasentia_phase2 as adapter


def run(input_root: Path, output_root: Path) -> None:
    quality = pd.read_csv(output_root / "sequence_quality_summary.csv")
    physics_rows = []
    v2_rows = []
    aggregate_rows = []
    oracle_rows = []
    yaw_rows = []
    representative = []

    for sequence in study.SEQUENCES:
        sequence_dir = output_root / sequence
        saved = pd.read_csv(sequence_dir / "aligned_terrasentia_v2_inputs.csv")
        frames = adapter.load_sequence(input_root / sequence)
        corrected, _ = adapter.build_aligned(frames, adapter.make_grid(frames))
        if len(saved) != len(corrected) or not np.allclose(
            saved["bag_time_s"], corrected["bag_time_s"], atol=1e-6, rtol=0
        ):
            raise RuntimeError(f"Alignment changed for {sequence}; saved traces cannot be reused")
        for column in ["rtk_east_m", "rtk_north_m", "forward_motor_speed_mps", "imu_yaw_rate_radps"]:
            if not np.allclose(saved[column], corrected[column], atol=2e-6, rtol=0):
                raise RuntimeError(f"{sequence}: {column} changed beyond CSV precision")
        saved["reference_heading_rad"] = corrected["reference_heading_rad"]
        saved.to_csv(sequence_dir / "aligned_terrasentia_v2_inputs.csv", index=False)

        physics_trace, _, physics_profile = study.run_physics(saved, sequence)
        physics_trace.to_csv(sequence_dir / "physics_only_trace.csv", index=False)
        physics_rows.append({"sequence": sequence, "model": "physics_only", **physics_profile})

        oracle = study.run_oracles(saved, sequence)
        oracle_rows.append(oracle)
        cases = {row["case"]: row for row in oracle.to_dict("records")}
        a = cases["A_motor_forward_plus_imu_yaw"]["ATE_m"]
        b = cases["B_reference_forward_plus_imu_yaw_oracle"]["ATE_m"]
        c = cases["C_motor_forward_plus_reference_yaw_oracle"]["ATE_m"]
        yaw_rows.append({
            "sequence": sequence,
            "physics_ATE_m": a,
            "translation_replacement_ATE_m": b,
            "yaw_replacement_ATE_m": c,
            "translation_replacement_ate_improvement_m": a - b,
            "yaw_replacement_ate_improvement_m": a - c,
            "yaw_replacement_materially_greater": bool((a - c) > 1.5 * max(a - b, 1e-9)),
        })

        old_rows = pd.read_csv(sequence_dir / "checkpoint_level_v2_metrics.csv")
        if len(old_rows) != 30:
            raise RuntimeError(f"Expected 30 saved V2 traces for {sequence}")
        rows = []
        for old in old_rows.to_dict("records"):
            trace_path = Path(old["trace_file"])
            if not trace_path.is_file():
                trace_path = sequence_dir / trace_path.name
            trace = pd.read_csv(trace_path)
            if len(trace) != len(saved) or not np.allclose(trace["time_s"], saved["time_s"], atol=1e-6, rtol=0):
                raise RuntimeError(f"Saved V2 trace no longer aligns: {trace_path}")
            # Repropagate saved rate predictions only if their initial heading changed.
            if abs(float(trace["theta_T_rad"].iloc[0]) - float(saved["reference_heading_rad"].iloc[0])) > 1e-8:
                propagated = adapter.integrate_prediction(
                    saved,
                    trace["pred_v_T_mps"].to_numpy(float) - saved["forward_motor_speed_mps"].to_numpy(float),
                    trace["pred_omega_T_radps"].to_numpy(float) - saved["imu_yaw_rate_radps"].to_numpy(float),
                    label="saved_v2_rates",
                )
                for key in ("x_T_m", "y_T_m", "theta_T_rad"):
                    trace[key] = propagated[key]
                trace.to_csv(trace_path, index=False)
            profile, _ = study.evaluate_trace(
                saved, trace, model="frozen_v2_checkpoint", sequence=sequence,
                seed=None if pd.isna(old["seed"]) else int(old["seed"]),
                replicate=old["replicate"],
            )
            old.update(profile)
            rows.append(old)
            v2_rows.append(old)
        v2 = pd.DataFrame(rows)
        v2.to_csv(sequence_dir / "checkpoint_level_v2_metrics.csv", index=False)
        aggregate_rows.append(study.summarize_v2(sequence, v2, physics_profile))
        plot = study.representative_plots(sequence, output_root, saved, physics_trace, v2)
        if plot:
            representative.append(plot)

        # Only the heading diagnostics depend on the repaired EKF yaw interpolation.
        heading = adapter.heading_yaw_audit(saved, sequence_dir)
        for key, value in {
            "imu_vs_ekf_final_heading_diff_deg": heading["final_heading_difference_deg"],
            "imu_vs_ekf_p95_abs_heading_disagreement_deg": heading["heading_disagreement_p95_abs_deg"],
            "imu_minus_ekf_mean_signed_yaw_rate_residual_radps": heading["mean_signed_imu_minus_reference_yaw_rate_radps"],
            "imu_minus_ekf_accumulated_signed_yaw_residual_deg": heading["accumulated_signed_yaw_rate_residual_final_deg"],
        }.items():
            quality.loc[quality.sequence == sequence, key] = value

    physics = pd.DataFrame(physics_rows)
    v2_all = pd.DataFrame(v2_rows)
    v2_agg = pd.DataFrame(aggregate_rows)
    oracles = pd.concat(oracle_rows, ignore_index=True)
    yaw = pd.DataFrame(yaw_rows)
    local_global = study.local_global_summary(physics, v2_all)
    quality.to_csv(output_root / "sequence_quality_summary.csv", index=False)
    physics.to_csv(output_root / "per_sequence_physics_metrics.csv", index=False)
    v2_all.to_csv(output_root / "checkpoint_level_v2_metrics.csv", index=False)
    v2_agg.to_csv(output_root / "per_sequence_v2_aggregate_metrics.csv", index=False)
    oracles.to_csv(output_root / "oracle_diagnostics.csv", index=False)
    yaw.to_csv(output_root / "yaw_attribution_summary.csv", index=False)
    local_global.to_csv(output_root / "local_vs_global_comparison.csv", index=False)
    representative.append(study.plot_local_global(local_global, output_root))
    representative.append(study.plot_sequence_bars(physics, v2_agg, output_root))
    study.write_report(output_root, quality=quality, physics=physics, v2_agg=v2_agg,
                       yaw=yaw, local_global=local_global, plots=representative)
    (output_root / "reference_heading_rescore_manifest.json").write_text(json.dumps({
        "change": "unwrap dataset EKF yaw before timestamp interpolation",
        "primary_position_reference": "RTK ENU unchanged",
        "model": "saved frozen V2 rate predictions, no checkpoint inference or retraining",
        "sequences": study.SEQUENCES,
    }, indent=2) + "\n", encoding="utf-8")
    print(output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=study.DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("results/terrasentia_external_validation"))
    args = parser.parse_args()
    run(args.input_root, args.output_dir)


if __name__ == "__main__":
    main()
