#!/usr/bin/env python3
"""
IoT resource characterization for the frozen Twin V2 checkpoint(s).

This script is intentionally conservative:
  * It measures checkpoint/state-dict resource quantities directly.
  * It infers recurrent input dimensions from PyTorch GRU weight shapes.
  * It reports model-side feature-tensor throughput and recurrent-history
    memory. These are NOT claimed to be wire/network bandwidth unless the
    user supplies --wire-bytes-per-update.
  * It does not invent inference latency. If you later have a callable
    one-step V2 inference function, benchmark that separately on the actual
    deployment machine.

Run from the repository root:
    python -m DigitalTwin.analysis.iot_resource_characterization

or:
    python DigitalTwin/analysis/iot_resource_characterization.py

Outputs:
    results/iot_resource_characterization/
      resource_summary.json
      resource_summary.csv
      checkpoint_inventory.csv
      recurrent_layers.csv
      resource_characterization_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    import torch
except Exception as exc:
    raise SystemExit(
        "PyTorch is required. Activate the same environment used for Twin V2 "
        f"and rerun. Import error: {exc}"
    )

CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin"}
DEFAULT_EXCLUDE_TOKENS = (
    "optimizer",
    "optim",
    "scheduler",
    "scaler",
    "aifarms",
    "terrasentia",
    "ugv01",
)

PARAMETER_BUFFER_TOKENS = ("running_mean", "running_var", "num_batches_tracked")


@dataclass
class TensorStats:
    tensor_count: int
    element_count: int
    byte_count: int
    parameter_like_element_count: int
    parameter_like_byte_count: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".", help="Repository root.")
    p.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Specific checkpoint path. Repeat for multiple checkpoints.",
    )
    p.add_argument(
        "--search-root",
        default="results",
        help="Root to recursively search when --checkpoint is omitted.",
    )
    p.add_argument(
        "--output",
        default="results/iot_resource_characterization",
        help="Output directory.",
    )
    p.add_argument("--update-hz", type=float, default=10.0)
    p.add_argument("--fast-context-s", type=float, default=2.0)
    p.add_argument("--slow-context-s", type=float, default=30.0)
    p.add_argument(
        "--scalar-bytes",
        type=int,
        default=4,
        help="Bytes per runtime feature scalar; FP32=4.",
    )
    p.add_argument(
        "--fast-feature-dim",
        type=int,
        default=None,
        help="Optional exact fast-branch feature dimension. If omitted, infer from GRU weights.",
    )
    p.add_argument(
        "--slow-feature-dim",
        type=int,
        default=None,
        help="Optional exact slow-branch feature dimension. If omitted, infer from GRU weights.",
    )
    p.add_argument(
        "--wire-bytes-per-update",
        type=float,
        default=None,
        help=(
            "Optional ACTUAL measured/serialized bytes delivered to the twin per update. "
            "If supplied, the script reports wire kB/s. Do not use an estimate here."
        ),
    )
    p.add_argument(
        "--max-checkpoints",
        type=int,
        default=200,
        help="Safety cap when auto-discovering checkpoints.",
    )
    return p.parse_args()


def repo_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def torch_load(path: Path) -> Any:
    # PyTorch 2.6 changed the default of weights_only. We explicitly allow
    # normal project checkpoints because these are local trusted artifacts.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_tensor_mapping(obj: Any) -> bool:
    if not isinstance(obj, Mapping) or not obj:
        return False
    pairs = list(obj.items())
    if not all(isinstance(k, str) for k, _ in pairs):
        return False
    tensor_values = sum(torch.is_tensor(v) for _, v in pairs)
    return tensor_values >= max(1, int(0.70 * len(pairs)))


def collect_tensor_mappings(obj: Any, prefix: str = "") -> List[Tuple[str, Mapping[str, Any]]]:
    """
    Find state-dict-like mappings while avoiding optimizer state dictionaries
    (which normally have integer keys).
    """
    found: List[Tuple[str, Mapping[str, Any]]] = []

    if isinstance(obj, torch.nn.Module):
        return [(prefix or "module", obj.state_dict())]

    if is_tensor_mapping(obj):
        found.append((prefix or "state_dict", obj))
        return found

    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = str(k)
            low = key.lower()
            if any(tok in low for tok in ("optimizer", "scheduler", "scaler")):
                continue
            subprefix = f"{prefix}.{key}" if prefix else key
            if isinstance(v, (Mapping, torch.nn.Module)):
                found.extend(collect_tensor_mappings(v, subprefix))
    return found


def flatten_state_dicts(mappings: List[Tuple[str, Mapping[str, Any]]]) -> Dict[str, torch.Tensor]:
    flat: Dict[str, torch.Tensor] = {}
    for namespace, mapping in mappings:
        for k, v in mapping.items():
            if not torch.is_tensor(v):
                continue
            full = f"{namespace}.{k}" if namespace else k
            # De-duplicate exact full names only.
            if full not in flat:
                flat[full] = v.detach().cpu()
    return flat


def tensor_stats(state: Mapping[str, torch.Tensor]) -> TensorStats:
    count = 0
    elems = 0
    nbytes = 0
    p_elems = 0
    p_bytes = 0

    for k, t in state.items():
        if not torch.is_tensor(t):
            continue
        count += 1
        n = int(t.numel())
        b = n * int(t.element_size())
        elems += n
        nbytes += b
        if not any(tok in k for tok in PARAMETER_BUFFER_TOKENS):
            p_elems += n
            p_bytes += b

    return TensorStats(count, elems, nbytes, p_elems, p_bytes)


def checkpoint_candidates(search_root: Path, max_count: int) -> List[Path]:
    if not search_root.exists():
        return []

    all_ckpts: List[Path] = []
    for p in search_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in CHECKPOINT_SUFFIXES:
            continue
        low = str(p).lower()
        # Prefer frozen V2 artifacts and avoid obvious external/UGV checkpoints.
        if "v2" not in low:
            continue
        if any(tok in low for tok in DEFAULT_EXCLUDE_TOKENS):
            continue
        all_ckpts.append(p)

    # Stable ordering helps reproducibility.
    all_ckpts = sorted(set(all_ckpts), key=lambda x: str(x).lower())
    return all_ckpts[:max_count]


GRU_WEIGHT_RE = re.compile(r"^(.*?)(?:\.|^)weight_ih_l(\d+)(?:_reverse)?$")


def infer_gru_layers(state: Mapping[str, torch.Tensor]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # Strip the discovery namespace when matching, but keep full key for provenance.
    for full_key, t in state.items():
        leaf_key = full_key
        # Find the final "weight_ih_lN" occurrence.
        m = re.search(r"(.*?)(weight_ih_l(\d+)(?:_reverse)?)$", leaf_key)
        if not m or t.ndim != 2:
            continue
        weight_name = m.group(2)
        layer_idx = int(m.group(3))
        rows_dim, input_dim = map(int, t.shape)
        if rows_dim % 3 != 0:
            # Likely not a GRU.
            continue
        hidden = rows_dim // 3
        prefix = m.group(1).rstrip(".")
        direction = "reverse" if weight_name.endswith("_reverse") else "forward"
        rows.append(
            {
                "prefix": prefix,
                "layer": layer_idx,
                "direction": direction,
                "input_dim": input_dim,
                "hidden_dim": hidden,
                "weight_key": full_key,
            }
        )

    rows.sort(key=lambda r: (r["prefix"], r["direction"], r["layer"]))
    return rows


def branch_specs(gru_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse GRU layer rows into one row per recurrent branch/direction and
    retain the layer-0 external input dimension.
    """
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in gru_rows:
        grouped.setdefault((row["prefix"], row["direction"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for (prefix, direction), rows in grouped.items():
        rows = sorted(rows, key=lambda r: r["layer"])
        layer0 = next((r for r in rows if r["layer"] == 0), rows[0])
        low = prefix.lower()
        if "slow" in low:
            kind = "slow"
        elif "fast" in low:
            kind = "fast"
        else:
            kind = "unclassified"
        out.append(
            {
                "prefix": prefix,
                "direction": direction,
                "kind": kind,
                "input_dim": int(layer0["input_dim"]),
                "hidden_dim": int(layer0["hidden_dim"]),
                "num_layers_detected": len({r["layer"] for r in rows}),
            }
        )
    return sorted(out, key=lambda r: (r["kind"], r["prefix"], r["direction"]))


def architecture_signature(state: Mapping[str, torch.Tensor]) -> List[Tuple[str, Tuple[int, ...]]]:
    """
    Compare checkpoints by leaf parameter name + shape, ignoring discovery namespace.
    """
    sig = []
    for k, t in state.items():
        # remove first discovery namespace segment when possible
        leaf = k.split(".", 1)[1] if "." in k else k
        sig.append((leaf, tuple(int(x) for x in t.shape)))
    return sorted(sig)


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = parse_args()
    root = Path(args.repo_root).resolve()
    outdir = repo_path(root, args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        ckpts = [repo_path(root, c).resolve() for c in args.checkpoint]
    else:
        ckpts = checkpoint_candidates(repo_path(root, args.search_root), args.max_checkpoints)

    if not ckpts:
        print("No V2 checkpoint files found.")
        print("Rerun with one or more explicit paths, for example:")
        print(
            "  python -m DigitalTwin.analysis.iot_resource_characterization "
            "--checkpoint path/to/frozen_v2_seed0.pt"
        )
        return 2

    inventory: List[Dict[str, Any]] = []
    representative_state: Optional[Dict[str, torch.Tensor]] = None
    representative_path: Optional[Path] = None
    reference_sig = None
    consistent = True
    load_errors = []

    for ckpt in ckpts:
        if not ckpt.exists():
            load_errors.append(f"{ckpt}: does not exist")
            continue
        try:
            obj = torch_load(ckpt)
            mappings = collect_tensor_mappings(obj)
            state = flatten_state_dicts(mappings)
            if not state:
                load_errors.append(f"{ckpt}: no state-dict-like tensor mapping found")
                continue
            st = tensor_stats(state)
            sig = architecture_signature(state)
            if reference_sig is None:
                reference_sig = sig
                representative_state = state
                representative_path = ckpt
            elif sig != reference_sig:
                consistent = False

            inventory.append(
                {
                    "checkpoint": str(ckpt.relative_to(root) if ckpt.is_relative_to(root) else ckpt),
                    "file_bytes": ckpt.stat().st_size,
                    "tensor_count": st.tensor_count,
                    "state_tensor_elements": st.element_count,
                    "state_tensor_bytes": st.byte_count,
                    "parameter_like_elements": st.parameter_like_element_count,
                    "parameter_like_bytes": st.parameter_like_byte_count,
                    "state_mapping_count": len(mappings),
                }
            )
        except Exception as exc:
            load_errors.append(f"{ckpt}: {type(exc).__name__}: {exc}")

    if representative_state is None or representative_path is None:
        print("Could not extract a model state dictionary from any checkpoint.")
        for e in load_errors:
            print("  ", e)
        return 3

    st = tensor_stats(representative_state)
    gru_rows = infer_gru_layers(representative_state)
    branches = branch_specs(gru_rows)

    # Save an inference-only state dict to obtain a direct serialized-size measurement.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "inference_state_dict.pt"
        torch.save(representative_state, tmp)
        inference_state_serialized_bytes = tmp.stat().st_size

    fast_dims = [b["input_dim"] for b in branches if b["kind"] == "fast"]
    slow_dims = [b["input_dim"] for b in branches if b["kind"] == "slow"]
    unknown_dims = [b["input_dim"] for b in branches if b["kind"] == "unclassified"]

    fast_dim = args.fast_feature_dim
    slow_dim = args.slow_feature_dim

    # If not overridden, infer conservatively from the first matching branch.
    if fast_dim is None and fast_dims:
        fast_dim = fast_dims[0]
    if slow_dim is None and slow_dims:
        slow_dim = slow_dims[0]

    fast_samples = int(round(args.fast_context_s * args.update_hz))
    slow_samples = int(round(args.slow_context_s * args.update_hz))

    # Model-side feature tensor calculations. These are not network payload claims.
    feature_dims_for_throughput = []
    if fast_dim is not None:
        feature_dims_for_throughput.append(int(fast_dim))
    if slow_dim is not None:
        feature_dims_for_throughput.append(int(slow_dim))
    if not feature_dims_for_throughput:
        # Fall back to detected external dims from every recurrent branch.
        feature_dims_for_throughput = [int(b["input_dim"]) for b in branches]

    feature_floats_per_update = int(sum(feature_dims_for_throughput))
    model_feature_bytes_per_update = feature_floats_per_update * args.scalar_bytes
    model_feature_bytes_per_s = model_feature_bytes_per_update * args.update_hz

    buffer_bytes = None
    buffer_assumption = ""
    if fast_dim is not None and slow_dim is not None:
        buffer_bytes = (
            fast_samples * fast_dim * args.scalar_bytes
            + slow_samples * slow_dim * args.scalar_bytes
        )
        buffer_assumption = (
            "One fast history buffer plus one slow history buffer using the supplied/"
            "inferred external feature dimensions."
        )
    elif branches:
        total = 0
        for b in branches:
            seconds = args.slow_context_s if b["kind"] == "slow" else args.fast_context_s
            samples = int(round(seconds * args.update_hz))
            total += samples * int(b["input_dim"]) * args.scalar_bytes
        buffer_bytes = total
        buffer_assumption = (
            "Upper-bound estimate treating each detected recurrent branch as having "
            "its own input-history buffer."
        )

    wire_bytes_per_s = (
        args.wire_bytes_per_update * args.update_hz
        if args.wire_bytes_per_update is not None
        else None
    )

    system = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }

    summary = {
        "representative_checkpoint": str(
            representative_path.relative_to(root)
            if representative_path.is_relative_to(root)
            else representative_path
        ),
        "checkpoint_count_loaded": len(inventory),
        "checkpoint_architecture_consistent": bool(consistent),
        "state_dict_tensor_count": st.tensor_count,
        "state_dict_tensor_elements": st.element_count,
        "state_dict_tensor_bytes": st.byte_count,
        "parameter_like_elements": st.parameter_like_element_count,
        "parameter_like_bytes": st.parameter_like_byte_count,
        "inference_only_serialized_state_dict_bytes": inference_state_serialized_bytes,
        "update_hz": args.update_hz,
        "update_period_ms": 1000.0 / args.update_hz,
        "fast_context_s": args.fast_context_s,
        "fast_context_samples": fast_samples,
        "slow_context_s": args.slow_context_s,
        "slow_context_samples": slow_samples,
        "fast_feature_dim": fast_dim,
        "slow_feature_dim": slow_dim,
        "model_feature_floats_per_update": feature_floats_per_update,
        "model_feature_bytes_per_update": model_feature_bytes_per_update,
        "model_feature_bytes_per_s": model_feature_bytes_per_s,
        "model_feature_kib_per_s": model_feature_bytes_per_s / 1024.0,
        "history_buffer_bytes": buffer_bytes,
        "history_buffer_kib": (buffer_bytes / 1024.0 if buffer_bytes is not None else None),
        "history_buffer_assumption": buffer_assumption,
        "wire_bytes_per_update_measured": args.wire_bytes_per_update,
        "wire_bytes_per_s_measured": wire_bytes_per_s,
        "wire_kib_per_s_measured": (
            wire_bytes_per_s / 1024.0 if wire_bytes_per_s is not None else None
        ),
        "runtime_sensor_modalities": ["wheel/odometry", "IMU"],
        "camera_required_for_v2_inference": False,
        "lidar_required_for_v2_inference": False,
        "radar_required_for_v2_inference": False,
        "gnss_required_for_v2_inference": False,
        "inference_latency_measured": False,
        "inference_latency_note": (
            "Not fabricated by this static checkpoint audit. Measure the actual one-step "
            "V2 inference callable on the deployment machine before putting latency in the paper."
        ),
        "system": system,
        "load_errors": load_errors,
    }

    (outdir / "resource_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_csv(outdir / "resource_summary.csv", [{k: v for k, v in summary.items() if k not in {"system", "load_errors", "runtime_sensor_modalities"}}])
    save_csv(outdir / "checkpoint_inventory.csv", inventory)
    save_csv(outdir / "recurrent_layers.csv", branches)

    mib = 1024.0 * 1024.0
    report = []
    report.append("# Twin V2 IoT Resource Characterization\n")
    report.append("## Scope\n")
    report.append(
        "This report measures the frozen checkpoint/state-dict footprint and model-side "
        "runtime feature-buffer cost. It does **not** treat model tensor throughput as "
        "wire/network bandwidth. Actual wire bandwidth is reported only when "
        "`--wire-bytes-per-update` is supplied from a measured serialized packet size.\n"
    )
    report.append("## Headline resource quantities\n")
    report.append(f"- Representative checkpoint: `{summary['representative_checkpoint']}`")
    report.append(f"- Frozen checkpoints inspected: **{len(inventory)}**")
    report.append(f"- Architecture consistent across inspected checkpoints: **{consistent}**")
    report.append(f"- Parameter-like state elements: **{st.parameter_like_element_count:,}**")
    report.append(f"- Parameter-like tensor memory: **{st.parameter_like_byte_count / mib:.3f} MiB**")
    report.append(f"- Inference-only serialized state dict: **{inference_state_serialized_bytes / mib:.3f} MiB**")
    report.append(f"- Twin update rate: **{args.update_hz:g} Hz** ({1000.0/args.update_hz:.1f} ms period)")
    report.append(f"- Runtime sensing: **wheel/odometry + IMU only**")
    report.append(f"- Camera/LiDAR/radar/GNSS required by V2 inference: **No**")
    if fast_dim is not None:
        report.append(f"- Fast recurrent external input dimension: **{fast_dim}**")
    if slow_dim is not None:
        report.append(f"- Slow recurrent external input dimension: **{slow_dim}**")
    report.append(
        f"- Model-side feature tensor throughput: **{model_feature_bytes_per_s/1024.0:.3f} KiB/s** "
        "(not wire bandwidth)"
    )
    if buffer_bytes is not None:
        report.append(f"- Recurrent input-history footprint: **{buffer_bytes/1024.0:.3f} KiB**")
        report.append(f"  - Assumption: {buffer_assumption}")
    if wire_bytes_per_s is not None:
        report.append(
            f"- Measured serialized input payload: **{wire_bytes_per_s/1024.0:.3f} KiB/s**"
        )
    else:
        report.append(
            "- Measured serialized network payload: **not supplied**; do not claim a network-bandwidth number from this run."
        )

    report.append("\n## Recurrent branches detected\n")
    if branches:
        report.append("| Branch | Kind | Input dim | Hidden dim | Layers | Direction |")
        report.append("|---|---:|---:|---:|---:|---|")
        for b in branches:
            report.append(
                f"| `{b['prefix']}` | {b['kind']} | {b['input_dim']} | {b['hidden_dim']} | "
                f"{b['num_layers_detected']} | {b['direction']} |"
            )
    else:
        report.append("No GRU `weight_ih_l*` tensors were detected in the extracted state dictionary.")

    report.append("\n## Machine / software provenance\n")
    for k, v in system.items():
        report.append(f"- {k}: `{v}`")

    report.append("\n## Paper-use boundary\n")
    report.append(
        "Safe claims from this report are checkpoint/model footprint, recurrent feature dimensions, "
        "history-buffer footprint, update rate, and the absence of camera/LiDAR/radar/GNSS inputs "
        "to V2 inference. Do **not** report inference latency or power from this script because they "
        "were not measured. Do **not** call model feature-tensor throughput 'network bandwidth'."
    )
    if load_errors:
        report.append("\n## Load warnings\n")
        report.extend([f"- {e}" for e in load_errors])

    (outdir / "resource_characterization_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(f"Wrote resource characterization to: {outdir}")
    print(f"Representative checkpoint: {summary['representative_checkpoint']}")
    print(f"Parameter-like elements: {st.parameter_like_element_count:,}")
    print(f"Inference-only state dict: {inference_state_serialized_bytes / mib:.3f} MiB")
    print(f"Model-side feature tensor throughput: {model_feature_bytes_per_s/1024.0:.3f} KiB/s")
    if buffer_bytes is not None:
        print(f"History buffer: {buffer_bytes/1024.0:.3f} KiB")
    if not consistent:
        print("WARNING: checkpoint architecture signatures were not identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
