"""Verify Kaggle shard archives and assemble immutable trajectory banks.

The importer trusts neither archive names nor directory layout.  Every copied
trajectory must be named by a passing shard-audit row and match the SHA-256
recorded by the Kaggle run manifest.  Incomplete studies still receive a
manifest and audit report, but are never labelled ready for analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS = {
    "doubly_nested": ROOT / "results/doubly_nested_loso",
    "causal_derivative": ROOT / "results/causal_derivative_loso",
}


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _json_members(archive: zipfile.ZipFile, stem: str) -> list[str]:
    return [name for name in archive.namelist() if PurePosixPath(name).name.startswith(stem) and name.endswith(".json")]


def _member_by_suffix(archive: zipfile.ZipFile, suffix: str) -> str:
    normalized = PurePosixPath(suffix).as_posix()
    matches = [name for name in archive.namelist() if PurePosixPath(name).as_posix().endswith(normalized)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one archive member ending in {normalized!r}; found {len(matches)}")
    return matches[0]


def _task_key(row: dict, study: str) -> tuple:
    if study == "doubly_nested":
        return row["outer"], row["role"], row["target"], int(row["seed"])
    return row.get("test", row.get("target")), int(row["seed"])


def _task_dict(key: tuple, study: str) -> dict:
    if study == "doubly_nested":
        return dict(zip(("outer", "role", "target", "seed"), key))
    return {"target": key[0], "seed": key[1]}


def _validate_row(row: dict, study: str) -> list[str]:
    errors: list[str] = []
    if row.get("status") != "PASS":
        errors.append("audit status is not PASS")
    if not row.get("v2_manifest") or not row.get("v2_trajectory_sha256"):
        errors.append("missing V2 manifest path or trajectory hash")
    if study == "doubly_nested":
        used = set(row.get("training", [])) | set(row.get("validation", []))
        if row.get("target") in used:
            errors.append("target sequence appears upstream")
        if row.get("outer") in used:
            errors.append("outer sequence appears upstream")
        if set(row.get("normalization_fit_sequences", [])) != set(row.get("training", [])):
            errors.append("normalization provenance differs from training split")
        if set(row.get("checkpoint_selection_sequences", [])) != set(row.get("validation", [])):
            errors.append("checkpoint-selection provenance differs from validation split")
    return errors


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def merge_archives(study: str, inputs: list[Path], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    bank = output / "trajectory_bank"
    bank.mkdir(parents=True, exist_ok=True)
    expected: dict[tuple, dict] = {}
    observed: dict[tuple, dict] = {}
    commits: set[str] = set()
    shard_counts: set[int] = set()
    shard_indices: set[int] = set()
    shard_sources: dict[int, str] = {}
    problems: list[str] = []
    archives_seen: list[dict] = []

    for input_path in inputs:
        candidates = sorted(input_path.glob("*.zip")) if input_path.is_dir() else [input_path]
        for path in candidates:
            try:
                with zipfile.ZipFile(path) as archive:
                    bad = archive.testzip()
                    if bad:
                        problems.append(f"{path}: corrupt member {bad}")
                        continue
                    audit_names = _json_members(archive, "shard_audit_")
                    ledger_names = _json_members(archive, "task_ledger_shard_")
                    if len(audit_names) != 1 or len(ledger_names) != 1:
                        problems.append(f"{path}: expected one audit and one ledger")
                        continue
                    audit = json.loads(archive.read(audit_names[0]))
                    ledger = json.loads(archive.read(ledger_names[0]))
                    if audit.get("commit") != ledger.get("commit"):
                        problems.append(f"{path}: audit and ledger commits differ")
                        continue
                    if study == "causal_derivative":
                        preflight_names = _json_members(archive, "causal_no_lookahead_preflight")
                        if len(preflight_names) != 1:
                            problems.append(f"{path}: missing unique numerical causality preflight")
                            continue
                        preflight = json.loads(archive.read(preflight_names[0]))
                        if not preflight.get("passed", False):
                            problems.append(f"{path}: numerical causality preflight did not pass")
                            continue
                    commit = audit.get("commit") or ledger.get("commit")
                    if commit:
                        commits.add(commit)
                    index = int(ledger["shard_index"])
                    count = int(ledger["shard_count"])
                    if index in shard_sources:
                        problems.append(f"duplicate shard index {index}: {shard_sources[index]} and {path}")
                        continue
                    shard_sources[index] = str(path)
                    shard_indices.add(index)
                    shard_counts.add(count)
                    tasks = ledger.get("shard_tasks", ledger.get("tasks", []))
                    for task in tasks:
                        key = _task_key(task, study)
                        if key in expected and expected[key] != task:
                            problems.append(f"conflicting ledger task {key}")
                        expected[key] = task
                    if not audit.get("complete", False):
                        problems.append(f"{path}: shard audit is incomplete")
                    archives_seen.append({"path": str(path), "shard_index": index, "tasks": len(tasks),
                                          "total_tasks": int(ledger.get("total_tasks", 0))})
                    for row in audit.get("rows", []):
                        key = _task_key(row, study)
                        row_errors = _validate_row(row, study)
                        if row_errors:
                            problems.extend(f"{path}:{key}: {message}" for message in row_errors)
                            continue
                        manifest_name = _member_by_suffix(archive, row["v2_manifest"])
                        run_manifest = json.loads(archive.read(manifest_name))
                        if run_manifest.get("repo_commit") != commit:
                            problems.append(f"{path}:{key}: V2 manifest repository commit mismatch")
                            continue
                        target = row.get("target", row.get("test"))
                        used = set(run_manifest.get("training_names", [])) | set(run_manifest.get("validation_names", []))
                        if target in used:
                            problems.append(f"{path}:{key}: V2 manifest includes target sequence upstream")
                            continue
                        if study == "doubly_nested" and row["outer"] in used:
                            problems.append(f"{path}:{key}: V2 manifest includes outer sequence upstream")
                            continue
                        if study == "causal_derivative" and run_manifest.get("feature_derivative_mode") != "causal_backward":
                            problems.append(f"{path}:{key}: V2 manifest is not causal_backward")
                            continue
                        if set(run_manifest.get("normalization_fit_sequences", [])) != set(run_manifest.get("training_names", [])):
                            problems.append(f"{path}:{key}: V2 normalization provenance mismatch")
                            continue
                        if set(run_manifest.get("checkpoint_selection_sequences", [])) != set(run_manifest.get("validation_names", [])):
                            problems.append(f"{path}:{key}: V2 checkpoint-selection provenance mismatch")
                            continue
                        split_parent = PurePosixPath(row["v1_split"]).parent
                        split_member = _member_by_suffix(archive, row["v1_split"])
                        split_rows = json.loads(archive.read(split_member))
                        matching_splits = [item for item in split_rows if item.get("test") == target]
                        if len(matching_splits) != 1:
                            problems.append(f"{path}:{key}: expected one matching V1 split row")
                            continue
                        split = matching_splits[0]
                        split_used = set(split.get("train", [])) | set(split.get("validation", []))
                        if target in split_used or (study == "doubly_nested" and row["outer"] in split_used):
                            problems.append(f"{path}:{key}: V1 split includes an excluded sequence")
                            continue
                        if set(split.get("normalization_fit_sequences", [])) != set(split.get("train", [])):
                            problems.append(f"{path}:{key}: V1 normalization provenance mismatch")
                            continue
                        if set(split.get("checkpoint_selection_sequences", [])) != set(split.get("validation", [])):
                            problems.append(f"{path}:{key}: V1 checkpoint-selection provenance mismatch")
                            continue
                        if study == "causal_derivative" and split.get("feature_derivative_mode") != "causal_backward":
                            problems.append(f"{path}:{key}: V1 split is not causal_backward")
                            continue
                        checkpoint_member = _member_by_suffix(
                            archive, str(split_parent / "folds" / str(target) / "gru_dual.pt")
                        )
                        results_member = _member_by_suffix(archive, str(split_parent / "loso_results.csv"))
                        with archive.open(checkpoint_member) as source:
                            checkpoint_hash = sha256_stream(source)
                        with archive.open(results_member) as source:
                            results_hash = sha256_stream(source)
                        if checkpoint_hash != row.get("v1_checkpoint_sha256"):
                            problems.append(f"{path}:{key}: archived V1 checkpoint hash mismatch")
                            continue
                        if results_hash != row.get("v1_results_sha256"):
                            problems.append(f"{path}:{key}: archived V1 results hash mismatch")
                            continue
                        if run_manifest.get("v1_checkpoint_sha256") != checkpoint_hash:
                            problems.append(f"{path}:{key}: V2 manifest V1-checkpoint provenance mismatch")
                            continue
                        if run_manifest.get("v1_results_sha256") != results_hash:
                            problems.append(f"{path}:{key}: V2 manifest V1-results provenance mismatch")
                            continue
                        trajectory_name = str(PurePosixPath(manifest_name).with_name("v2_evaluated_trajectory.csv"))
                        trajectory_name = _member_by_suffix(archive, trajectory_name)
                        with archive.open(trajectory_name) as source:
                            digest = sha256_stream(source)
                        if digest != row["v2_trajectory_sha256"]:
                            problems.append(f"{path}:{key}: trajectory SHA-256 mismatch")
                            continue
                        if key in observed and observed[key]["sha256"] != digest:
                            problems.append(f"{path}:{key}: conflicting duplicate trajectory")
                            continue
                        fields = _task_dict(key, study)
                        if study == "doubly_nested":
                            destination = (bank / f"outer_{fields['outer']}" / fields["role"] /
                                           f"replicate_seed_{fields['seed']}" / f"fold_{fields['target']}" /
                                           "v2_evaluated_trajectory.csv")
                        else:
                            destination = (bank / f"replicate_seed_{fields['seed']}" / f"fold_{fields['target']}" /
                                           "v2_evaluated_trajectory.csv")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        current_digest = None
                        if destination.exists():
                            with destination.open("rb") as current:
                                current_digest = sha256_stream(current)
                        if current_digest != digest:
                            with archive.open(trajectory_name) as source, tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as temporary:
                                shutil.copyfileobj(source, temporary)
                                temporary_path = Path(temporary.name)
                            temporary_path.replace(destination)
                        observed[key] = {**fields, "trajectory": _portable_path(destination), "sha256": digest,
                                         "archive": str(path), "v2_manifest_member": manifest_name}
            except (OSError, zipfile.BadZipFile, KeyError, ValueError, RuntimeError) as exc:
                problems.append(f"{path}: {type(exc).__name__}: {exc}")

    if shard_counts and len(shard_counts) == 1:
        missing_shards = sorted(set(range(next(iter(shard_counts)))) - shard_indices)
    else:
        missing_shards = []
        if len(shard_counts) > 1:
            problems.append(f"inconsistent shard counts: {sorted(shard_counts)}")
    missing_tasks = sorted(set(expected) - set(observed), key=str)
    unexpected_tasks = sorted(set(observed) - set(expected), key=str)
    if unexpected_tasks:
        problems.append(f"{len(unexpected_tasks)} audited trajectories have no ledger task")
    expected_total_from_ledger = max([entry["total_tasks"] for entry in archives_seen] + [0])
    complete = bool(expected_total_from_ledger) and len(observed) == expected_total_from_ledger and not missing_shards and not missing_tasks and not problems and len(commits) == 1
    status = "complete" if complete else ("contaminated" if problems or len(commits) > 1 else "incomplete")
    manifest = {
        "schema": "verified_kaggle_trajectory_bank_v1",
        "study": study,
        "status": status,
        "ready_for_evidence_analysis": complete,
        "repository_commits": sorted(commits),
        "expected_total_tasks": expected_total_from_ledger,
        "ledger_tasks_seen": len(expected),
        "verified_trajectories": len(observed),
        "missing_shards": missing_shards,
        "missing_tasks": [_task_dict(key, study) for key in missing_tasks],
        "problems": problems,
        "archives": archives_seen,
        "trajectories": [observed[key] for key in sorted(observed, key=str)],
    }
    (output / "merged_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = ["# Kaggle trajectory-bank audit", "", f"- Study: `{study}`", f"- Status: **{status.upper()}**",
              f"- Verified trajectories: `{len(observed)} / {expected_total_from_ledger or 'unknown'}`",
              f"- Repository commits: `{', '.join(sorted(commits)) or 'none'}`",
              f"- Missing shards: `{missing_shards or 'none'}`", "", "## Problems", ""]
    report.extend(f"- {problem}" for problem in problems)
    if not problems:
        report.append("- None detected in the supplied archives.")
    report += ["", "A bank is eligible for paper-grade analysis only when `ready_for_evidence_analysis` is true.",
               "Partial trajectories are retained for debugging but must not be used for headline counts."]
    (output / "trajectory_bank_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=tuple(DEFAULT_OUTPUTS), required=True)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Shard ZIP or directory containing ZIPs; repeatable")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = merge_archives(args.study, args.input, args.output or DEFAULT_OUTPUTS[args.study])
    print(json.dumps({key: manifest[key] for key in ("study", "status", "expected_total_tasks", "verified_trajectories")}, indent=2))


if __name__ == "__main__":
    main()
