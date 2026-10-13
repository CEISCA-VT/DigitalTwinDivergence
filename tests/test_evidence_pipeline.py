import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from DigitalTwin.analysis import run_doubly_nested_loso as runner
from DigitalTwin.analysis import service_evidence_from_bank as evidence

from DigitalTwin.analysis.deterministic_delivery_replay import sampled_delivery
from DigitalTwin.analysis.kaggle_trajectory_bank import merge_archives
from DigitalTwin.analysis.run_doubly_nested_loso import (
    SEQUENCES,
    Task,
    build_tasks,
    commands,
    task_paths,
    v1_complete,
)


def test_deterministic_delivery_replay_is_reproducible():
    time = np.arange(100, dtype=float) * 0.1
    condition = {"rate_hz": 5.0, "delay_ms": 50.0, "jitter_sd_ms": 20.0, "loss_probability": 0.1}
    first, first_stats = sampled_delivery(time, condition, 1234)
    second, second_stats = sampled_delivery(time, condition, 1234)
    np.testing.assert_array_equal(first, second)
    assert first_stats == second_stats
    assert 0 <= first_stats["realized_loss_fraction"] <= 1
    assert np.all((first < len(time)) & (first >= -1))


def test_doubly_nested_archive_merge_requires_and_verifies_audit(tmp_path):
    trajectory = b"time_s,gt_east_m\n0,0\n"
    digest = hashlib.sha256(trajectory).hexdigest()
    checkpoint = b"checkpoint"
    checkpoint_digest = hashlib.sha256(checkpoint).hexdigest()
    results = b"test,method,status\nbuilding00,gru_dual,ok\n"
    results_digest = hashlib.sha256(results).hexdigest()
    task = {"outer": "building00", "role": "outer_test", "target": "building00", "seed": 42}
    ledger = {"commit": "abc123", "shard_index": 0, "shard_count": 1,
              "total_tasks": 1, "shard_tasks": [task]}
    manifest_name = "study/outer_building00/outer_test/target_building00/v2/run_manifest.json"
    row = {**task, "status": "PASS", "v2_manifest": "outer_building00/outer_test/target_building00/v2/run_manifest.json",
           "v1_split": "outer_building00/outer_test/target_building00/v1_seed_42/fold_splits.json",
           "v1_checkpoint_sha256": checkpoint_digest, "v1_results_sha256": results_digest,
           "v2_trajectory_sha256": digest, "training": ["parking00"], "validation": ["parking01"],
           "normalization_fit_sequences": ["parking00"], "checkpoint_selection_sequences": ["parking01"]}
    audit = {"commit": "abc123", "complete": True, "rows": [row]}
    archive_path = tmp_path / "shard.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("study/task_ledger_shard_00.json", json.dumps(ledger))
        archive.writestr("study/shard_audit_00.json", json.dumps(audit))
        archive.writestr(manifest_name, json.dumps({"repo_commit": "abc123", "training_names": ["parking00"], "validation_names": ["parking01"],
                                                    "normalization_fit_sequences": ["parking00"],
                                                    "checkpoint_selection_sequences": ["parking01"],
                                                    "v1_checkpoint_sha256": checkpoint_digest,
                                                    "v1_results_sha256": results_digest}))
        archive.writestr("study/outer_building00/outer_test/target_building00/v1_seed_42/folds/building00/gru_dual.pt", checkpoint)
        archive.writestr("study/outer_building00/outer_test/target_building00/v1_seed_42/loso_results.csv", results)
        archive.writestr("study/outer_building00/outer_test/target_building00/v1_seed_42/fold_splits.json",
                         json.dumps([{"test": "building00", "train": ["parking00"], "validation": ["parking01"],
                                      "normalization_fit_sequences": ["parking00"],
                                      "checkpoint_selection_sequences": ["parking01"]}]))
        archive.writestr("study/outer_building00/outer_test/target_building00/v2/v2_evaluated_trajectory.csv", trajectory)
    result = merge_archives("doubly_nested", [archive_path], tmp_path / "merged")
    assert result["status"] == "complete"
    assert result["ready_for_evidence_analysis"]
    assert result["verified_trajectories"] == 1
    assert len(list((tmp_path / "merged" / "trajectory_bank").rglob("v2_evaluated_trajectory.csv"))) == 1


def test_single_shard_doubly_nested_task_graph_is_complete_and_exclusive():
    tasks = build_tasks((42, 1042, 2042))
    assert len(tasks) == 300
    assert len({task.task_id for task in tasks}) == 300
    for outer in SEQUENCES:
        fold = [task for task in tasks if task.outer == outer]
        assert len(fold) == 30
        outer_tasks = [task for task in fold if task.role == "outer_test"]
        qualification_tasks = [task for task in fold if task.role == "qualification_train"]
        assert len(outer_tasks) == 3
        assert len(qualification_tasks) == 27
        assert all(task.target == outer and task.extra_excluded is None for task in outer_tasks)
        assert all(task.target != outer and task.extra_excluded == outer for task in qualification_tasks)


def test_causal_nested_commands_apply_causal_mode_to_both_training_stages(tmp_path):
    task = Task("building00", "qualification_train", "building01", 42, "building00")
    args = type("Args", (), {
        "output_dir": tmp_path / "causal",
        "dataset_root": tmp_path / "dataset",
        "frozen_v1_dir": tmp_path / "frozen",
        "python": "python",
        "device": "cuda",
        "num_workers": 4,
        "feature_derivative_mode": "causal_backward",
    })()
    v1, v2 = commands(args, task)
    assert v1[v1.index("--feature-derivative-mode") + 1] == "causal_backward"
    assert v2[v2.index("--feature-derivative-mode") + 1] == "causal_backward"
    assert v1[v1.index("--additional-excluded-sequence") + 1] == "building00"
    assert v2[v2.index("--additional-excluded-sequence") + 1] == "building00"
    assert "--overwrite" in v2


def test_causal_nested_resume_rejects_centered_v1_artifacts(tmp_path):
    task = Task("building00", "outer_test", "building00", 42, None)
    paths = task_paths(tmp_path, task)
    paths["v1_checkpoint"].parent.mkdir(parents=True)
    paths["v1_checkpoint"].write_bytes(b"checkpoint")
    paths["v1_results"].write_text("test,method,status\nbuilding00,gru_dual,ok\n", encoding="utf-8")
    paths["v1_split"].write_text(json.dumps([{
        "test": "building00",
        "train": ["parking00"],
        "validation": ["parking01"],
        "normalization_fit_sequences": ["parking00"],
        "checkpoint_selection_sequences": ["parking01"],
        "feature_derivative_mode": "centered",
    }]), encoding="utf-8")
    assert v1_complete(paths, task, "centered")
    assert not v1_complete(paths, task, "causal_backward")


def _write_completed_task(output, task, mode="causal_backward", commit="abc123"):
    paths = task_paths(output, task)
    paths["v1_checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    paths["v1_checkpoint"].write_bytes(b"v1 checkpoint")
    paths["v1_results"].write_text(
        f"test_sequence,method,status\n{task.target},gru_dual,ok\n", encoding="utf-8"
    )
    excluded = {task.target, task.extra_excluded}
    eligible = [seq for seq in SEQUENCES if seq not in excluded]
    validation, training = eligible[:2], eligible[2:]
    runner.write_json(paths["v1_split"], [{
        "test": task.target, "train": training, "validation": validation,
        "normalization_fit_sequences": training,
        "checkpoint_selection_sequences": validation, "feature_derivative_mode": mode,
    }])
    directory = paths["v2"] / f"replicate_{task.seed}" / f"fold_01_{task.target}"
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "v2_checkpoint": "v2_slow_additive_yaw.pt",
        "evaluated_trajectory": "v2_evaluated_trajectory.csv",
        "prediction_trace": "v2_prediction_trace.csv",
        "run_summary": "run_summary.json",
    }
    for name in [*artifacts.values(), "fidelity_profile.json", "fidelity_timeseries.csv"]:
        (directory / name).write_bytes(b"test artifact")
    hashes = {key: runner.sha256_file(directory / name) for key, name in artifacts.items()}
    runner.write_json(directory / "run_manifest.json", {
        "status": "complete", "base_seed": task.seed, "test_sequence": task.target,
        "repo_commit": commit, "feature_derivative_mode": mode,
        "v1_checkpoint_source": str(paths["v1_checkpoint"].resolve()),
        "v1_results_source": str(paths["v1_results"].resolve()),
        "v1_checkpoint_sha256": runner.sha256_file(paths["v1_checkpoint"]),
        "v1_results_sha256": runner.sha256_file(paths["v1_results"]),
        "training_names": training, "validation_names": validation,
        "normalization_fit_sequences": training,
        "checkpoint_selection_sequences": validation, "artifacts_sha256": hashes,
    })
    runner.write_json(directory / "RUN_COMPLETE.json", {
        "status": "complete", "test_sequence": task.target, "base_seed": task.seed,
        "repo_commit": commit, "v2_checkpoint_sha256": hashes["v2_checkpoint"],
        "evaluated_trajectory_sha256": hashes["evaluated_trajectory"],
    })
    return paths, directory


def test_causal_nested_completion_checks_sources_markers_and_hashes(tmp_path):
    task = Task("building00", "outer_test", "building00", 42, None)
    paths, directory = _write_completed_task(tmp_path, task)
    assert runner.v2_complete(paths, task, "causal_backward")
    assert not runner.v2_complete(paths, task, "centered")
    paths["v1_results"].write_text("different results", encoding="utf-8")
    assert not runner.v2_complete(paths, task, "causal_backward")
    _write_completed_task(tmp_path, task)
    (directory / "v2_slow_additive_yaw.pt").write_bytes(b"changed checkpoint")
    assert not runner.v2_complete(paths, task, "causal_backward")
    _write_completed_task(tmp_path, task)
    (directory / "RUN_COMPLETE.json").unlink()
    assert not runner.v2_complete(paths, task, "causal_backward")


def test_causal_nested_resumes_partial_task_then_skips_verified_task(tmp_path, monkeypatch):
    task = Task("building00", "outer_test", "building00", 42, None)
    paths, directory = _write_completed_task(tmp_path, task)
    (directory / "RUN_COMPLETE.json").unlink()
    args = runner.parser().parse_args([
        "--output-dir", str(tmp_path), "--feature-derivative-mode", "causal_backward",
        "--outer-sequences", "building00", "--seeds", "42", "--max-tasks", "1",
    ])
    monkeypatch.setattr(runner, "validate_inputs", lambda args: None)
    monkeypatch.setattr(runner, "repository_state", lambda: ("abc123", False))
    monkeypatch.setattr(runner, "causal_feature_preflight", lambda: {"passed": True, "status": "PASS"})
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert command[2] == runner.V2_RUNNER
        assert "--overwrite" in command
        _write_completed_task(tmp_path, task)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(runner.subprocess, "run", execute)
    runner.run(args)
    assert len(calls) == 1
    manifest = runner.read_json(tmp_path / "merged_manifest.json")
    assert manifest["feature_derivative_mode"] == "causal_backward"
    assert manifest["verified_trajectories"] == 1
    assert not manifest["ready_for_evidence_analysis"]
    runner.run(args)
    assert len(calls) == 1
    (directory / "v2_prediction_trace.csv").write_bytes(b"changed trace")
    runner.run(args)
    assert len(calls) == 2


def test_existing_centered_bank_is_rejected_before_overwriting(tmp_path):
    path = tmp_path / "task_ledger.json"
    path.write_text(json.dumps({"feature_derivative_mode": "centered"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different derivative mode"):
        runner.validate_output_mode(tmp_path, "causal_backward")
    assert json.loads(path.read_text())["feature_derivative_mode"] == "centered"


def test_evidence_rejects_modified_trajectory_and_missing_causal_preflight(tmp_path):
    path = tmp_path / "trajectory.csv"
    path.write_bytes(b"trajectory")
    manifest = {
        "expected_total_tasks": 1, "feature_derivative_mode": "causal_backward",
        "causal_no_lookahead_preflight": {"passed": True},
        "trajectories": [{"outer": "building00", "role": "outer_test", "target": "building00",
                          "seed": 42, "trajectory": str(path), "sha256": runner.sha256_file(path)}],
    }
    evidence.validate_bank_sources(manifest)
    path.write_bytes(b"changed trajectory")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        evidence.validate_bank_sources(manifest)
    manifest["causal_no_lookahead_preflight"] = {}
    with pytest.raises(RuntimeError, match="no-lookahead"):
        evidence.validate_bank_sources(manifest)


def test_risk_plot_annotation_uses_new_results(tmp_path, monkeypatch):
    labels = []
    original = evidence.plt.Axes.annotate

    def annotate(self, text, *args, **kwargs):
        labels.append(text)
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(evidence.plt.Axes, "annotate", annotate)
    summary = pd.DataFrame([{
        "scope": "pooled", "method": "service_empirical", "target_acceptance": 0.25,
        "achieved_acceptance": 0.25, "selective_risk": 0.0875,
        "false_qualified_count": 7, "accepted_count": 80,
    }])
    evidence._plot_risk(summary, tmp_path / "risk.png")
    assert "7/80 (8.8%)" in labels
    assert "4/200 (2.0%)" not in labels
