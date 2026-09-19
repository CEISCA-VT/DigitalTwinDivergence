import hashlib
import json
import zipfile

import numpy as np

from DigitalTwin.analysis.deterministic_delivery_replay import sampled_delivery
from DigitalTwin.analysis.kaggle_trajectory_bank import merge_archives
from DigitalTwin.analysis.run_doubly_nested_loso import SEQUENCES, build_tasks


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
