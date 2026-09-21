"""Replay measured Linux netem delivery traces over frozen twin trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from DigitalTwin.analysis import deterministic_delivery_replay as deterministic
from DigitalTwin.analysis import service_timing_budget_study as base


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE_ROOT = ROOT / "results/netem_delivery/captures"
DEFAULT_OUTPUT = ROOT / "results/netem_delivery/evidence"
REQUIRED_CONDITIONS = {"ideal", "practical", "degraded"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_trajectory_sources(bank_manifest: Path | None = None) -> list[dict]:
    if bank_manifest is None:
        return [
            {
                "sequence": path.parent.name.split("_", 2)[-1],
                "model_replicate": path.parent.parent.name,
                "path": path,
            }
            for path in base.discover()
        ]

    payload = json.loads(bank_manifest.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or not payload.get("ready_for_evidence_analysis"):
        raise RuntimeError(f"trajectory bank is not complete and analysis-ready: {bank_manifest}")
    rows = [row for row in payload.get("trajectories", []) if row.get("role") == "outer_test"]
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 doubly nested outer-test trajectories, found {len(rows)}")

    sources = []
    seen = set()
    for row in rows:
        path = Path(row["trajectory"])
        if not path.is_absolute():
            path = ROOT / path
        key = (str(row["outer"]), int(row["seed"]))
        if key in seen:
            raise RuntimeError(f"duplicate outer-sequence/seed trajectory: {key}")
        seen.add(key)
        if not path.is_file():
            raise FileNotFoundError(f"trajectory listed by bank manifest is missing: {path}")
        expected_hash = row.get("sha256")
        if expected_hash and _sha256(path) != expected_hash:
            raise RuntimeError(f"trajectory hash mismatch: {path}")
        sources.append(
            {
                "sequence": str(row["outer"]),
                "model_replicate": f"seed_{int(row['seed'])}",
                "path": path,
            }
        )
    return sources


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def discover_captures(root: Path) -> list[tuple[dict, pd.DataFrame, Path]]:
    if not root.is_absolute():
        root = ROOT / root
    captures = []
    for manifest_path in sorted(root.glob("*/replicate_*/capture_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ledger_path = manifest_path.parent / "packet_delivery_ledger.csv"
        ledger = pd.read_csv(ledger_path)
        if ledger.empty or not {"packet_id", "received", "measured_delay_ms"}.issubset(ledger.columns):
            raise RuntimeError(f"invalid packet ledger: {ledger_path}")
        captures.append((manifest, ledger, ledger_path))
    if not captures:
        raise RuntimeError(f"no netem captures found below {root}")
    conditions = {item[0]["condition"] for item in captures}
    missing = REQUIRED_CONDITIONS - conditions
    if missing:
        raise RuntimeError(f"missing required netem conditions: {sorted(missing)}")
    return captures


def measured_delivery(t: np.ndarray, ledger: pd.DataFrame, rate_hz: float) -> tuple[np.ndarray, dict]:
    dt = float(np.median(np.diff(t)))
    stride = max(1, int(round((1.0 / rate_hz) / dt)))
    source = np.arange(0, len(t), stride)
    pattern = ledger.sort_values("packet_id").reset_index(drop=True)
    if pattern.empty:
        raise ValueError("empty netem packet pattern")
    pattern_index = np.arange(len(source)) % len(pattern)
    selected = pattern.iloc[pattern_index]
    received = selected.received.astype(bool).to_numpy()
    delays_s = selected.measured_delay_ms.to_numpy(float) / 1000.0
    keep_source = source[received]
    keep_delay = delays_s[received]
    arrivals = t[keep_source] + keep_delay
    order = np.argsort(arrivals, kind="stable")
    arrivals = arrivals[order]
    keep_source = keep_source[order]
    delivered = np.full(len(t), -1, dtype=int)
    newest = -1
    cursor = 0
    for index, now in enumerate(t):
        while cursor < len(arrivals) and arrivals[cursor] <= now + 1e-12:
            newest = max(newest, int(keep_source[cursor]))
            cursor += 1
        delivered[index] = newest
    duration = max(float(t[-1] - t[0]), dt)
    stats = {
        "requested_rate_hz": rate_hz,
        "realized_arrival_rate_hz": float((arrivals <= t[-1] + 1e-12).sum() / duration),
        "requested_packets": int(len(source)),
        "delivered_packets": int(received.sum()),
        "lost_packets": int((~received).sum()),
        "realized_loss_fraction": float((~received).mean()),
        "delay_p50_ms": float(np.quantile(keep_delay, 0.50) * 1000) if len(keep_delay) else np.nan,
        "delay_p95_ms": float(np.quantile(keep_delay, 0.95) * 1000) if len(keep_delay) else np.nan,
        "delay_max_ms": float(np.max(keep_delay) * 1000) if len(keep_delay) else np.nan,
        "capture_packets": int(len(pattern)),
        "trace_repetitions": float(len(source) / len(pattern)),
    }
    return delivered, stats


def run(capture_root: Path, output: Path, bank_manifest: Path | None = None) -> None:
    captures = discover_captures(capture_root)
    sources = discover_trajectory_sources(bank_manifest)
    run_rows: list[dict] = []
    transport_rows: list[dict] = []
    capture_sources: list[dict] = []
    for manifest, ledger, ledger_path in captures:
        condition = manifest["condition"]
        rate_hz = float(manifest["requested_rate_hz"])
        transport_replicate = int(manifest["transport_replicate"])
        capture_sources.append({**manifest, "packet_ledger": _repo_relative(ledger_path)})
        for source in sources:
            path = source["path"]
            data = base.load(path)
            sequence = source["sequence"]
            model_replicate = source["model_replicate"]
            delivered, stats = measured_delivery(data["time_s"], ledger, rate_hz)
            transport_rows.append(
                {
                    "sequence": sequence,
                    "model_replicate": model_replicate,
                    "transport_replicate": transport_replicate,
                    "condition": condition,
                    **stats,
                }
            )
            for service in base.SERVICES:
                run_rows.append(
                    {
                        "sequence": sequence,
                        "model_replicate": model_replicate,
                        "transport_replicate": transport_replicate,
                        "condition": condition,
                        "service": service["service"],
                        **deterministic.evaluate_indices(data, service, delivered),
                    }
                )

    per_run = pd.DataFrame(run_rows)
    transport = pd.DataFrame(transport_rows)
    metrics = [
        "coverage",
        "physical_satisfaction",
        "freshness_satisfaction",
        "joint_satisfaction",
        "mean_aoi_s",
        "max_aoi_s",
    ]
    per_sequence = per_run.groupby(["sequence", "condition", "service"], as_index=False)[metrics].mean()
    per_sequence["qualified"] = (
        (per_sequence.coverage >= base.MIN_COVERAGE)
        & (per_sequence.physical_satisfaction >= base.TARGET)
        & (per_sequence.freshness_satisfaction >= base.TARGET)
        & (per_sequence.joint_satisfaction >= base.TARGET)
    ).astype(int)
    ledger = deterministic.build_ledger(per_sequence)

    output.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(output / "per_run_netem_contract_replay.csv", index=False)
    per_sequence.to_csv(output / "per_sequence_netem_contract_replay.csv", index=False)
    transport.to_csv(output / "netem_trace_application_statistics.csv", index=False)
    ledger.to_csv(output / "netem_delivery_case_ledger.csv", index=False)
    counts = ledger.classification.value_counts()
    summary = {
        "schema": "netem_delivery_replay_v1",
        "status": "complete",
        "source": "measured UDP arrivals over Linux veth shaped with tc netem",
        "receiver_policy": "newest source timestamp among packets arrived by each 10-Hz evaluation time",
        "trace_application": "captured packet delay/loss pattern repeats only when a frozen trajectory requests more packets than the capture contains",
        "inference_hierarchy": "timestamps within model and transport replicates within physical sequence; physical sequence is the primary unit",
        "physical_sequences": int(per_sequence.sequence.nunique()),
        "trajectory_source": str(bank_manifest) if bank_manifest else str(base.SOURCE),
        "model_replicates": int(per_run.model_replicate.nunique()),
        "transport_replicates": int(per_run.transport_replicate.nunique()),
        "delivery_remediable": int(counts.get("delivery_remediable", 0)),
        "persistent_under_ideal": int(counts.get("persistent_under_ideal", 0)),
        "already_qualified_degraded": int(counts.get("already_qualified_degraded", 0)),
        "practical_restores": int(ledger.practical_restores.sum()),
        "captures": capture_sources,
        "claim_boundary": "OS-level delivery emulation over precomputed frozen states; not live twin inference or measured communication savings",
    }
    (output / "netem_delivery_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Linux netem delivery replay",
        "",
        "Frozen states were delivered using measured UDP delay/loss traces captured over a Linux veth link shaped with `tc netem`.",
        "This is OS-level delivery emulation over precomputed states, not live twin inference.",
        "",
        "## Case ledger",
        "",
        f"- Delivery-remediable: **{summary['delivery_remediable']}/40**",
        f"- Persistent under ideal delivery: **{summary['persistent_under_ideal']}/40**",
        f"- Already qualified under degraded delivery: **{summary['already_qualified_degraded']}/40**",
        f"- Restored by practical delivery: **{summary['practical_restores']}**",
        "",
        "Transport repetitions quantify delivery variability; they are not additional physical-sequence replicates.",
    ]
    (output / "netem_delivery_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bank-manifest",
        type=Path,
        help="Completed doubly nested merged_manifest.json; only its 30 outer-test trajectories are replayed.",
    )
    args = parser.parse_args()
    run(args.capture_root, args.output, args.bank_manifest)


if __name__ == "__main__":
    main()
