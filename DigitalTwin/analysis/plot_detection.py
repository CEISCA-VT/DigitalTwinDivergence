"""Generate ROC and detection-probability plots from experiment CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .common import is_attack_row, parse_run_name, read_rows, write_rows


def roc_points(labels: list[int], scores: list[float]) -> list[dict[str, float]]:
    thresholds = sorted(set(scores), reverse=True)
    if thresholds:
        thresholds = [thresholds[0] + 1e-9] + thresholds + [thresholds[-1] - 1e-9]
    positives = sum(labels)
    negatives = len(labels) - positives
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        tp = sum(1 for label, score in zip(labels, scores) if label and score >= threshold)
        fp = sum(1 for label, score in zip(labels, scores) if not label and score >= threshold)
        rows.append(
            {
                "threshold": threshold,
                "tpr": tp / positives if positives else 0.0,
                "fpr": fp / negatives if negatives else 0.0,
            }
        )
    return rows


def read_pd_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="CSV files or glob patterns")
    parser.add_argument("--pd-summary", default="")
    parser.add_argument("--out-dir", default="DigitalTwin/datasets/analysis")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for item in args.inputs:
        matches = sorted(Path().glob(item)) if any(ch in item for ch in "*?[") else [Path(item)]
        paths.extend(path for path in matches if path.exists())

    labels: list[int] = []
    scores: list[float] = []
    for path in paths:
        for row in read_rows(path):
            labels.append(1 if is_attack_row(row) else 0)
            scores.append(float(row["mahalanobis"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roc = roc_points(labels, scores)
    write_rows(out_dir / "roc_points.csv", roc, ["threshold", "tpr", "fpr"])

    plt.figure()
    plt.plot([row["fpr"] for row in roc], [row["tpr"] for row in roc])
    plt.plot([0, 1], [0, 1], "--", color="0.5")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.tight_layout()
    plt.savefig(out_dir / "roc.png", dpi=160)
    plt.close()

    summary_path = Path(args.pd_summary) if args.pd_summary else out_dir / "pd_summary.csv"
    if summary_path.exists():
        rows = [row for row in read_pd_summary(summary_path) if row["attack"] == "step" and row["epsilon_m"]]
        grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for row in rows:
            key = (row["speed"], row["terrain"], row["latency"])
            grouped.setdefault(key, []).append(row)
        plt.figure()
        for key, group in sorted(grouped.items()):
            group = sorted(group, key=lambda row: float(row["epsilon_m"]))
            plt.plot(
                [float(row["epsilon_m"]) for row in group],
                [float(row["trial_pd"]) for row in group],
                marker="o",
                label=f"v={key[0]}, tau={key[1]}, l={key[2]}",
            )
        plt.xlabel("step bias magnitude (m)")
        plt.ylabel("empirical P_D")
        plt.ylim(-0.05, 1.05)
        if grouped:
            plt.legend(fontsize="small")
        plt.tight_layout()
        plt.savefig(out_dir / "pd_vs_step_magnitude.png", dpi=160)
        plt.close()

    print(out_dir)


if __name__ == "__main__":
    main()
