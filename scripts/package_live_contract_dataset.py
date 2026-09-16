"""Package UGV01 live-contract trials into a public dataset layout.

This is a deterministic file organizer: it pairs each policy/trial video with
the matching dashboard telemetry CSV, copies available analysis summaries, and
emits dataset-level metadata for paper-facing use.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


POLICIES = {
    "Static_Low": ("static-low", "static_low"),
    "Static_High": ("static-high", "static_high"),
    "Aoi_Only": ("aoi-only", "aoi_only"),
    "Contract_Aware": ("contract-aware", "contract_aware"),
}


def read_expected_jsonl_map(source: Path) -> dict[tuple[str, int], str]:
    summary_path = source / "ugv01_live_20_trials" / "live_run_summary.csv"
    if not summary_path.exists():
        return {}
    with summary_path.open(newline="", encoding="utf-8-sig") as f:
        rows = csv.DictReader(f)
        mapping = {}
        for row in rows:
            policy = (row.get("policy") or "").strip()
            trial = try_float(row.get("trial"))
            log_file = (row.get("log_file") or "").strip()
            if policy and trial is not None and log_file:
                mapping[(policy, int(trial))] = Path(log_file).name
        return mapping


def inventory_jsonl(path: Path) -> dict[str, object]:
    records = 0
    policy = ""
    schema = ""
    times: list[float] = []
    gps_valid = 0
    first_keys = ""
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records += 1
            if not policy:
                policy = str(obj.get("policy") or "")
            if not schema:
                schema = str(obj.get("schema") or "")
            if not first_keys:
                first_keys = ";".join(obj.keys())
            point = obj.get("point") or {}
            if point.get("gps_valid") is True:
                gps_valid += 1
            for key in ("t_s", "time_s", "elapsed_s", "wall_time_s", "source_time_s"):
                value = try_float(point.get(key))
                if value is not None:
                    times.append(value)
                    break
    return {
        "jsonl_file": path.name,
        "source_path": path.as_posix(),
        "bytes": path.stat().st_size,
        "records": records,
        "duration_s": round(max(times) - min(times), 6) if len(times) >= 2 else None,
        "policy_in_file": policy,
        "schema": schema,
        "gps_valid_fraction": round(gps_valid / records, 6) if records else None,
        "top_level_keys": first_keys,
    }


def copy_jsonl_pool(raw_roots: list[Path], output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    pool_dir = output / "raw_jsonl_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    for root in raw_roots:
        if not root.exists():
            continue
        for src in sorted(root.rglob("*.jsonl")):
            unique_name = src.name
            if unique_name in seen:
                unique_name = f"{len(seen):04d}_{src.name}"
            seen.add(unique_name)
            dst = pool_dir / unique_name
            shutil.copy2(src, dst)
            row = inventory_jsonl(src)
            row["packaged_path"] = dst.as_posix()
            rows.append(row)
    if rows:
        write_csv(output / "jsonl_inventory.csv", rows)
    return rows


def try_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        value_f = float(text)
    except ValueError:
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def truthy(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def csv_stats(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {
            "telemetry_rows": 0,
            "telemetry_duration_s": None,
            "gps_valid_fraction": None,
            "satellite_median": None,
            "hdop_median": None,
        }

    time_values: list[float] = []
    for row in rows:
        for key in ("t", "time_s", "source_time_s", "timestamp_s"):
            value = try_float(row.get(key))
            if value is not None:
                time_values.append(value)
                break

    gps_valid: list[bool] = []
    for row in rows:
        value = truthy(row.get("gps_valid"))
        if value is not None:
            gps_valid.append(value)

    satellites: list[float] = []
    hdop: list[float] = []
    for row in rows:
        for sat_key in ("satellites", "gps_satellites", "sat"):
            value = try_float(row.get(sat_key))
            if value is not None:
                satellites.append(value)
                break
        for hdop_key in ("hdop", "gps_hdop"):
            value = try_float(row.get(hdop_key))
            if value is not None:
                hdop.append(value)
                break

    return {
        "telemetry_rows": len(rows),
        "telemetry_duration_s": (max(time_values) - min(time_values)) if len(time_values) >= 2 else None,
        "gps_valid_fraction": (sum(gps_valid) / len(gps_valid)) if gps_valid else None,
        "satellite_median": median(satellites) if satellites else None,
        "hdop_median": median(hdop) if hdop else None,
    }


def video_stats(path: Path) -> dict[str, object]:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return {
            "video_frames": None,
            "video_fps": None,
            "video_duration_s": None,
        }

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {
            "video_frames": None,
            "video_fps": None,
            "video_duration_s": None,
        }
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    return {
        "video_frames": frames if frames > 0 else None,
        "video_fps": fps if fps > 0 else None,
        "video_duration_s": (frames / fps) if frames > 0 and fps > 0 else None,
    }


def find_trial_file(policy_dir: Path, trial: int, suffix: str) -> Path | None:
    matches = sorted(policy_dir.glob(f"T{trial}_*{suffix}"))
    return matches[0] if matches else None


def find_jsonl(raw_roots: list[Path], expected_name: str | None) -> Path | None:
    if not expected_name:
        return None
    for root in raw_roots:
        if not root.exists():
            continue
        matches = [p for p in root.rglob("*.jsonl") if p.name == expected_name]
        if matches:
            return sorted(matches)[0]
    return None


def copy_if_present(src: Path | None, dst: Path | None) -> str:
    if src is None or dst is None:
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.as_posix()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compact_number(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 6)
    return value


def build_readme(dataset_name: str, rows: list[dict[str, object]], analysis_present: bool) -> str:
    by_policy: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy"])].append(row)

    lines = [
        f"# {dataset_name}",
        "",
        "This dataset packages the UGV01 smooth-floor live-contract validation trials.",
        "It is organized as paired video and dashboard telemetry for each policy/trial arm.",
        "",
        "## Layout",
        "",
        "- `trials/<policy>/trial_XX/video.mp4`: phone video used for AprilTag/visual reference.",
        "- `trials/<policy>/trial_XX/telemetry.csv`: dashboard-exported UGV01 telemetry.",
        "- `trials/<policy>/trial_XX/live_contract.jsonl`: live dashboard contract and resource-policy trace.",
        "- `analysis/`: generated live-contract summaries copied from the source trial analysis folder.",
        "- `dataset_manifest.csv`: one row per policy/trial with source paths and basic quality metadata.",
        "- `dataset_summary.json`: machine-readable aggregate summary.",
        "",
        "## Trial Matrix",
        "",
        "| Policy | Trials | Video duration mean (s) | Telemetry rows mean | GPS valid fraction mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    def mean_or_na(values: list[float], precision: int = 1) -> str:
        if not values:
            return "n/a"
        return f"{mean(values):.{precision}f}"

    for policy in sorted(by_policy):
        policy_rows = by_policy[policy]
        durations = [float(r["video_duration_s"]) for r in policy_rows if r.get("video_duration_s") not in ("", None)]
        telemetry_rows = [float(r["telemetry_rows"]) for r in policy_rows if r.get("telemetry_rows") not in ("", None)]
        gps = [float(r["gps_valid_fraction"]) for r in policy_rows if r.get("gps_valid_fraction") not in ("", None)]
        lines.append(
            f"| `{policy}` | {len(policy_rows)} | "
            f"{mean_or_na(durations, 1)} | {mean_or_na(telemetry_rows, 1)} | {mean_or_na(gps, 3)} |"
        )

    lines.extend(
        [
            "",
            "## Intended Use",
            "",
            "Use the videos plus telemetry CSV files for UGV01 physical-reference/digital-twin fidelity analysis.",
            "Use live-contract JSONL logs as the primary evidence for online contract state, policy decisions, AoI, and requested update-rate behavior.",
            "",
            "## Limitations",
            "",
            "- The current package is a single indoor smooth-floor dataset.",
            "- The packaged CSV telemetry is the stable source for the current AprilTag/fidelity tooling.",
            f"- Matching per-run JSONL files are present for {sum(bool(row.get('live_contract_jsonl_path')) for row in rows)}/{len(rows)} trials.",
            "- The canonical paper dataset contains only the matched final-trial logs; development/debug logs are excluded.",
            "- Requested policy rates were logged, but the delivered stream rate remained close to 1.5 Hz across policies; this package alone does not demonstrate delivered communication savings.",
            "- GPS was valid in the telemetry exports, but GPS is an operational sensor rather than independent ground truth.",
        ]
    )
    if not analysis_present:
        lines.append("- No generated live-contract analysis folder was present in the source tree.")
    return "\n".join(lines) + "\n"


def package_dataset(source: Path, output: Path, raw_roots: list[Path], *, include_debug_jsonl_pool: bool = False) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    dataset_name = output.name
    expected_jsonl = read_expected_jsonl_map(source)
    jsonl_inventory = copy_jsonl_pool(raw_roots, output) if include_debug_jsonl_pool else []

    for source_policy_dir, (policy, policy_folder) in POLICIES.items():
        policy_dir = source / source_policy_dir
        if not policy_dir.exists():
            raise FileNotFoundError(f"missing policy folder: {policy_dir}")

        for trial in range(1, 6):
            video_src = find_trial_file(policy_dir, trial, ".mp4")
            csv_src = find_trial_file(policy_dir, trial, ".csv")
            expected_jsonl_name = expected_jsonl.get((policy, trial), "")
            jsonl_src = find_jsonl(raw_roots, expected_jsonl_name)
            if video_src is None or csv_src is None:
                raise FileNotFoundError(f"missing video or csv for {source_policy_dir} T{trial}")

            trial_dir = output / "trials" / policy_folder / f"trial_{trial:02d}"
            video_dst = trial_dir / "video.mp4"
            csv_dst = trial_dir / "telemetry.csv"
            jsonl_dst = trial_dir / "live_contract.jsonl" if jsonl_src else None

            video_rel = copy_if_present(video_src, video_dst)
            csv_rel = copy_if_present(csv_src, csv_dst)
            jsonl_rel = copy_if_present(jsonl_src, jsonl_dst)

            stats = {
                **video_stats(video_src),
                **csv_stats(csv_src),
            }

            row: dict[str, object] = {
                "dataset_id": dataset_name,
                "policy": policy,
                "trial": trial,
                "physical_condition": "smooth_floor_mixed_motion",
                "wireless_condition": "wifi_baseline",
                "source_video": video_src.as_posix(),
                "source_telemetry_csv": csv_src.as_posix(),
                "source_live_contract_jsonl": jsonl_src.as_posix() if jsonl_src else "",
                "expected_live_contract_jsonl": expected_jsonl_name,
                "video_path": video_rel,
                "telemetry_csv_path": csv_rel,
                "live_contract_jsonl_path": jsonl_rel,
                **{k: compact_number(v) for k, v in stats.items()},
            }
            rows.append(row)

    analysis_src = source / "ugv01_live_20_trials"
    analysis_present = analysis_src.exists()
    if analysis_present:
        analysis_dst = output / "analysis"
        analysis_dst.mkdir(parents=True, exist_ok=True)
        for path in sorted(analysis_src.glob("*")):
            if path.is_file():
                shutil.copy2(path, analysis_dst / path.name)

    write_csv(output / "dataset_manifest.csv", rows)

    by_policy: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy"])].append(row)

    summary = {
        "schema": "ugv01_live_contract_public_dataset_v1",
        "dataset_id": dataset_name,
        "source_root": source.as_posix(),
        "output_root": output.as_posix(),
        "trial_count": len(rows),
        "policies": sorted(by_policy),
        "trials_per_policy": {policy: len(policy_rows) for policy, policy_rows in sorted(by_policy.items())},
        "has_per_trial_live_contract_jsonl": {
            policy: sum(1 for row in policy_rows if row["live_contract_jsonl_path"])
            for policy, policy_rows in sorted(by_policy.items())
        },
        "raw_jsonl_pool_count": len(jsonl_inventory),
        "exact_expected_jsonl_count": sum(1 for row in rows if row["live_contract_jsonl_path"]),
        "video_duration_s_total": round(
            sum(float(row["video_duration_s"]) for row in rows if row["video_duration_s"] not in ("", None)),
            3,
        ),
        "gps_valid_fraction_mean": round(
            mean(float(row["gps_valid_fraction"]) for row in rows if row["gps_valid_fraction"] not in ("", None)),
            6,
        ),
        "analysis_folder_copied": analysis_present,
        "notes": [
            "Dataset represents an indoor smooth-floor prospective live-contract trial set.",
            "Telemetry CSV is the primary source for current AprilTag/fidelity analysis tooling.",
            "All 20 trials include a matching JSONL contract log for online contract-policy analysis.",
        ],
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "README.md").write_text(build_readme(dataset_name, rows, analysis_present), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("live_contract_trials"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public_datasets") / "ugv01_live_contract",
    )
    parser.add_argument(
        "--jsonl-root",
        type=Path,
        action="append",
        default=[
            Path("raw_logs") / "new_trials",
            Path("raw_logs") / "live_validation",
            Path("live_contract_trials") / "json logs" / "new_trials",
            Path("live_contract_trials") / "json logs" / "live_validation",
        ],
    )
    parser.add_argument(
        "--include-debug-jsonl-pool",
        action="store_true",
        help="Copy unmatched development/debug JSONL logs into a separate pool (excluded from the paper dataset).",
    )
    args = parser.parse_args()

    summary = package_dataset(
        args.source,
        args.output,
        args.jsonl_root,
        include_debug_jsonl_pool=args.include_debug_jsonl_pool,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
