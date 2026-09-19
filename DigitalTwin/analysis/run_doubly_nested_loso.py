"""Resumable, single-shard doubly nested i2Nav LOSO orchestration.

This module runs every outer-test and qualification-training V1->V2 pipeline
in one process.  It is intended for one long GPU allocation (for example one
Slurm job), not for statistical aggregation across independent workers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = (
    "building00", "building01", "building02", "parking00", "parking01",
    "parking02", "playground00", "street00", "street01", "street02",
)
DEFAULT_SEEDS = (42, 1042, 2042)
V1_RUNNER = "DigitalTwin.analysis.i2nav_loso_ablation"
V2_RUNNER = "DigitalTwin.analysis.i2nav_v2_full_loso"


@dataclass(frozen=True)
class Task:
    outer: str
    role: str
    target: str
    seed: int
    extra_excluded: str | None

    @property
    def task_id(self) -> str:
        return f"outer-{self.outer}__role-{self.role}__target-{self.target}__seed-{self.seed}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def repository_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
        check=True, capture_output=True, text=True,
    ).stdout.strip())
    return commit, dirty


def build_tasks(seeds: tuple[int, ...], outer_sequences: tuple[str, ...] = SEQUENCES) -> list[Task]:
    tasks = []
    for outer in outer_sequences:
        for seed in seeds:
            tasks.append(Task(outer, "outer_test", outer, seed, None))
        for target in SEQUENCES:
            if target == outer:
                continue
            for seed in seeds:
                tasks.append(Task(outer, "qualification_train", target, seed, outer))
    return sorted(tasks, key=lambda task: (task.outer, task.role, task.target, task.seed))


def task_paths(output: Path, task: Task) -> dict[str, Path]:
    parent = output / f"outer_{task.outer}" / task.role / f"target_{task.target}"
    v1 = parent / f"v1_seed_{task.seed}"
    return {
        "parent": parent,
        "v1": v1,
        "v2": parent / "v2",
        "v1_checkpoint": v1 / "folds" / task.target / "gru_dual.pt",
        "v1_results": v1 / "loso_results.csv",
        "v1_split": v1 / "fold_splits.json",
        "log": output / "logs" / f"{task.task_id}.log",
    }


def commands(args: argparse.Namespace, task: Task) -> tuple[list[str], list[str]]:
    paths = task_paths(args.output_dir, task)
    v1 = [
        args.python, "-m", V1_RUNNER, "--root", str(args.dataset_root),
        "--output-dir", str(paths["v1"]), "--folds", task.target,
        "--methods", "gru_dual", "--save-trajectories", "--seed", str(task.seed),
        "--device", args.device, "--num-workers", str(args.num_workers),
    ]
    v2 = [
        args.python, "-m", V2_RUNNER, "--root", str(args.dataset_root),
        "--frozen-v1-dir", str(args.frozen_v1_dir), "--output-dir", str(paths["v2"]),
        "--test-sequence", task.target, "--base-seed", str(task.seed),
        "--device", args.device, "--v1-checkpoint", str(paths["v1_checkpoint"]),
        "--v1-results-csv", str(paths["v1_results"]),
    ]
    if task.extra_excluded:
        v1.extend(("--additional-excluded-sequence", task.extra_excluded))
        v2.extend(("--additional-excluded-sequence", task.extra_excluded))
    return v1, v2


def v1_complete(paths: dict[str, Path], task: Task) -> bool:
    if not all(paths[name].is_file() for name in ("v1_checkpoint", "v1_results", "v1_split")):
        return False
    try:
        split_rows = read_json(paths["v1_split"])
        if len(split_rows) != 1 or split_rows[0].get("test") != task.target:
            return False
        with paths["v1_results"].open(newline="", encoding="utf-8") as stream:
            return any(row.get("method") == "gru_dual" and row.get("status") == "ok"
                       for row in csv.DictReader(stream))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def v2_manifests(paths: dict[str, Path], task: Task) -> list[tuple[Path, dict]]:
    found = []
    for path in paths["v2"].rglob(f"*_{task.target}/run_manifest.json") if paths["v2"].is_dir() else []:
        try:
            manifest = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if manifest.get("base_seed") == task.seed:
            found.append((path, manifest))
    return found


def v2_complete(paths: dict[str, Path], task: Task) -> bool:
    matches = v2_manifests(paths, task)
    if len(matches) != 1:
        return False
    path, manifest = matches[0]
    required = (
        "RUN_COMPLETE.json", "run_summary.json", "v2_slow_additive_yaw.pt",
        "v2_evaluated_trajectory.csv", "v2_prediction_trace.csv",
        "fidelity_profile.json", "fidelity_timeseries.csv",
    )
    return (all(path.with_name(name).is_file() for name in required)
            and manifest.get("status") == "complete"
            and Path(manifest.get("v1_checkpoint_source", "")).resolve() == paths["v1_checkpoint"].resolve()
            and Path(manifest.get("v1_results_source", "")).resolve() == paths["v1_results"].resolve())


def audit_task(output: Path, task: Task, expected_commit: str) -> dict:
    paths = task_paths(output, task)
    if not v1_complete(paths, task):
        raise RuntimeError(f"Incomplete V1 artifacts for {task.task_id}")
    split = read_json(paths["v1_split"])[0]
    used = set(split["train"]) | set(split["validation"])
    excluded = {task.target} | ({task.extra_excluded} if task.extra_excluded else set())
    if used & excluded:
        raise RuntimeError(f"V1 exclusion failure for {task.task_id}: {sorted(used & excluded)}")
    if set(split.get("normalization_fit_sequences", [])) != set(split["train"]):
        raise RuntimeError(f"V1 normalization provenance failure for {task.task_id}")
    if set(split.get("checkpoint_selection_sequences", [])) != set(split["validation"]):
        raise RuntimeError(f"V1 checkpoint-selection provenance failure for {task.task_id}")
    matches = v2_manifests(paths, task)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one V2 manifest for {task.task_id}; found {len(matches)}")
    manifest_path, manifest = matches[0]
    if manifest.get("repo_commit") != expected_commit:
        raise RuntimeError(f"V2 commit mismatch for {task.task_id}")
    v2_used = set(manifest.get("training_names", [])) | set(manifest.get("validation_names", []))
    if v2_used & excluded:
        raise RuntimeError(f"V2 exclusion failure for {task.task_id}: {sorted(v2_used & excluded)}")
    if set(manifest.get("normalization_fit_sequences", [])) != set(manifest.get("training_names", [])):
        raise RuntimeError(f"V2 normalization provenance failure for {task.task_id}")
    if set(manifest.get("checkpoint_selection_sequences", [])) != set(manifest.get("validation_names", [])):
        raise RuntimeError(f"V2 checkpoint-selection provenance failure for {task.task_id}")
    checkpoint_hash = sha256_file(paths["v1_checkpoint"])
    results_hash = sha256_file(paths["v1_results"])
    if manifest.get("v1_checkpoint_sha256") != checkpoint_hash or manifest.get("v1_results_sha256") != results_hash:
        raise RuntimeError(f"V1 source hash mismatch in V2 manifest for {task.task_id}")
    trajectory = manifest_path.with_name("v2_evaluated_trajectory.csv")
    trajectory_hash = sha256_file(trajectory)
    if manifest.get("artifacts_sha256", {}).get("evaluated_trajectory") != trajectory_hash:
        raise RuntimeError(f"V2 trajectory hash mismatch for {task.task_id}")
    return {
        **asdict(task), "task_id": task.task_id, "status": "PASS",
        "v1_split": str(paths["v1_split"].relative_to(output)),
        "v1_checkpoint_sha256": checkpoint_hash, "v1_results_sha256": results_hash,
        "v2_manifest": str(manifest_path.relative_to(output)),
        "v2_trajectory_sha256": trajectory_hash,
        "trajectory": portable_path(trajectory),
        "training": manifest["training_names"], "validation": manifest["validation_names"],
        "normalization_fit_sequences": manifest["normalization_fit_sequences"],
        "checkpoint_selection_sequences": manifest["checkpoint_selection_sequences"],
    }


def validate_inputs(args: argparse.Namespace) -> None:
    missing = [name for name in SEQUENCES if not (args.dataset_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing i2Nav sequence directories: {missing}")
    if not (args.frozen_v1_dir / "FROZEN_MANIFEST.json").is_file():
        raise FileNotFoundError(f"Missing frozen V1 manifest: {args.frozen_v1_dir / 'FROZEN_MANIFEST.json'}")
    for runner, option in ((V1_RUNNER, "--additional-excluded-sequence"), (V2_RUNNER, "--v1-checkpoint")):
        help_text = subprocess.run([args.python, "-m", runner, "--help"], cwd=REPO,
                                   check=True, capture_output=True, text=True).stdout
        if option not in help_text:
            raise RuntimeError(f"{runner} does not expose required option {option}")


def selected_tasks(tasks: list[Task], start: int, stop: int | None, max_tasks: int | None) -> list[Task]:
    selected = tasks[start:stop]
    return selected[:max_tasks] if max_tasks is not None else selected


def run(args: argparse.Namespace) -> None:
    args.dataset_root = args.dataset_root.resolve()
    args.frozen_v1_dir = args.frozen_v1_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    validate_inputs(args)
    commit, dirty = repository_state()
    if args.expected_commit and commit != args.expected_commit:
        raise RuntimeError(f"Repository commit {commit} does not match --expected-commit {args.expected_commit}")
    if dirty and not args.allow_dirty:
        raise RuntimeError("Tracked repository files are modified; commit them or pass --allow-dirty for a dry run")
    tasks = build_tasks(tuple(args.seeds), tuple(args.outer_sequences))
    active = selected_tasks(tasks, args.start_task, args.stop_task, args.max_tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = {
        "schema": "doubly_nested_loso_single_shard_ledger_v1", "repository_commit": commit,
        "repository_dirty_at_start": dirty, "total_tasks": len(tasks), "selected_tasks": len(active),
        "base_seeds": args.seeds, "outer_sequences": args.outer_sequences,
        "start_task": args.start_task, "stop_task": args.stop_task, "max_tasks": args.max_tasks,
        "tasks": [asdict(task) | {"task_id": task.task_id} for task in active],
    }
    write_json(args.output_dir / "task_ledger.json", ledger)
    if args.dry_run:
        preview = []
        for task in active[:min(3, len(active))]:
            v1, v2 = commands(args, task)
            preview.append({"task": asdict(task), "v1_command": v1, "v2_command": v2})
        write_json(args.output_dir / "dry_run_commands.json", preview)
        print(json.dumps({"status": "dry_run", "total_tasks": len(tasks), "selected_tasks": len(active),
                          "commit": commit, "output": str(args.output_dir)}, indent=2))
        return

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO) + os.pathsep + environment.get("PYTHONPATH", "")
    status_path = args.output_dir / "task_status.json"
    statuses = {row["task_id"]: row for row in read_json(status_path).get("tasks", [])} if status_path.exists() else {}
    started = time.time()
    for number, task in enumerate(active, 1):
        paths = task_paths(args.output_dir, task)
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        print(f"[{number}/{len(active)}] {task.task_id}", flush=True)
        v1_command, v2_command = commands(args, task)
        task_started = time.time()
        if not v1_complete(paths, task):
            with paths["log"].open("a", encoding="utf-8") as log:
                result = subprocess.run(v1_command, cwd=REPO, env=environment, text=True,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"V1 failed for {task.task_id}; inspect {paths['log']}")
        else:
            print("  V1 already complete", flush=True)
        if not v2_complete(paths, task):
            with paths["log"].open("a", encoding="utf-8") as log:
                result = subprocess.run(v2_command, cwd=REPO, env=environment, text=True,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"V2 failed for {task.task_id}; inspect {paths['log']}")
        else:
            print("  V2 already complete", flush=True)
        audit = audit_task(args.output_dir, task, commit)
        statuses[task.task_id] = {**audit, "elapsed_s": time.time() - task_started}
        write_json(status_path, {"schema": "doubly_nested_loso_task_status_v1",
                                 "repository_commit": commit, "tasks": list(statuses.values())})

    audited = []
    for task in tasks:
        paths = task_paths(args.output_dir, task)
        if v2_complete(paths, task):
            audited.append(audit_task(args.output_dir, task, commit))
    complete = len(audited) == len(tasks)
    full_design = (set(args.outer_sequences) == set(SEQUENCES)
                   and tuple(args.seeds) == DEFAULT_SEEDS)
    ready = complete and full_design
    write_json(args.output_dir / "full_audit.json", {
        "schema": "doubly_nested_loso_full_audit_v1", "repository_commit": commit,
        "complete": complete, "full_10_sequence_3_seed_design": full_design,
        "verified_tasks": len(audited), "expected_tasks": len(tasks), "rows": audited,
    })
    trajectories = [{key: row[key] for key in ("outer", "role", "target", "seed", "trajectory")}
                    | {"sha256": row["v2_trajectory_sha256"]} for row in audited]
    write_json(args.output_dir / "merged_manifest.json", {
        "schema": "verified_kaggle_trajectory_bank_v1", "study": "doubly_nested",
        "status": "complete" if ready else ("subset_complete" if complete else "incomplete"),
        "ready_for_evidence_analysis": ready,
        "repository_commits": [commit], "expected_total_tasks": len(tasks),
        "verified_trajectories": len(audited), "problems": [], "trajectories": trajectories,
        "wall_time_s": time.time() - started,
    })
    print(json.dumps({"status": "complete" if ready else ("subset_complete" if complete else "partial"), "verified": len(audited),
                      "expected": len(tasks), "output": str(args.output_dir)}, indent=2))


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--dataset-root", type=Path, default=REPO / "public_datasets/im2nav")
    command.add_argument("--frozen-v1-dir", type=Path, default=REPO / "results/i2nav_v1_frozen")
    command.add_argument("--output-dir", type=Path, default=REPO / "results/doubly_nested_loso_slurm")
    command.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    command.add_argument("--outer-sequences", nargs="+", choices=SEQUENCES, default=list(SEQUENCES))
    command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    command.add_argument("--num-workers", type=int, default=4)
    command.add_argument("--python", default=sys.executable)
    command.add_argument("--expected-commit")
    command.add_argument("--start-task", type=int, default=0)
    command.add_argument("--stop-task", type=int)
    command.add_argument("--max-tasks", type=int)
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--allow-dirty", action="store_true")
    return command


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
