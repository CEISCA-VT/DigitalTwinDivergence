# Linux Netem Delivery Study

This experiment sends timestamped frozen-state packet identifiers through a
Linux virtual Ethernet pair shaped by `tc netem`. It records actual monotonic
send/arrival timestamps and uses the measured delay/loss traces in the existing
service-contract evaluator.

The supported claim is **OS-level emulated delivery over precomputed frozen
states**. It is not live twin inference, measured energy savings, or a safety
experiment.

## Requirements

- Ubuntu or another Linux distribution with `iproute2` (`ip` and `tc`)
- `sudo` access or equivalent `CAP_NET_ADMIN`
- Python 3.10+ and this repository's `requirements.txt`
- For combined capture and replay: either the 30 frozen V2 trajectories under
  `results/i2nav_v2_full_loso/i2nav_v2_full_loso/`, or the completed doubly
  nested bank and its `merged_manifest.json`

Falcon compute nodes normally do not grant `sudo`; use a local Linux machine or
VM where you control the network namespace. The GPU is not used.

## Clone or update

```bash
git clone YOUR_REPOSITORY_URL DigitalTwinDivergence
cd DigitalTwinDivergence
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an existing checkout:

```bash
cd DigitalTwinDivergence
git pull
source .venv/bin/activate
```

The cleanest workflow is to capture network traces in the VM, copy only the
small capture directory back, and replay it locally against the audited doubly
nested bank. This avoids copying the 3.8-GB training bank into the VM.

Confirm the local doubly nested bank before starting the VM capture:

```powershell
python -c "from pathlib import Path; from DigitalTwin.analysis.netem_delivery_replay import discover_trajectory_sources; print(len(discover_trajectory_sources(Path(r'results\doubly_nested_loso_slurm\merged_manifest.json'))))"
```

The count must be `30`.

## Preflight

```bash
python -m pytest tests/test_netem_delivery.py tests/test_evidence_pipeline.py -q
sudo -v
tc -V
ip -Version
```

## Full recommended capture in the Linux VM

Three 180-second transport repetitions for all four conditions take about 36
minutes plus a few minutes of idle timeout and analysis:

```bash
bash scripts/linux/run_netem_delivery_study.sh --duration-s 180 --repetitions 3 --capture-only
```

The script may prompt once for the `sudo` password. It creates an isolated
network namespace and veth pair, applies `tc netem`, and removes both on exit.
It does not alter the host's physical Wi-Fi/Ethernet interface or default
route.

## Short smoke test

Run this first if the VM setup is new:

```bash
bash scripts/linux/run_netem_delivery_study.sh --duration-s 10 --repetitions 1 --capture-only --output results/netem_delivery_smoke
```

The smoke test validates plumbing only. Do not report its ledger as the paper
result because a ten-second transport trace is too short.

After the full capture, copy `results/netem_delivery/captures/` back into the
same path in the Windows checkout and run:

```powershell
python -m DigitalTwin.analysis.netem_delivery_replay --capture-root results\netem_delivery\captures --bank-manifest results\doubly_nested_loso_slurm\merged_manifest.json --output results\netem_delivery\evidence
```

This verifies trajectory hashes and replays only the 30 doubly nested
outer-test trajectories. The 270 qualification-training trajectories are not
used by the netem ledger.

## Outputs

Each condition/replicate contains:

- `sent_packets.csv`
- `received_packets.csv`
- `packet_delivery_ledger.csv`
- `capture_manifest.json`
- `tc_qdisc_before.txt` and `tc_qdisc_after.txt`

The final evidence directory contains:

- `per_run_netem_contract_replay.csv`
- `per_sequence_netem_contract_replay.csv`
- `netem_trace_application_statistics.csv`
- `netem_delivery_case_ledger.csv`
- `netem_delivery_manifest.json`
- `netem_delivery_summary.md`

Check that measured delay and loss agree reasonably with the configured values:

```bash
python - <<'PY'
import json
from pathlib import Path
for path in sorted(Path('results/netem_delivery/captures').glob('*/replicate_*/capture_manifest.json')):
    m = json.loads(path.read_text())
    print(m['condition'], m['transport_replicate'],
          f"loss={m['realized_loss_fraction']:.3f}",
          f"p50={m['delay_p50_ms']:.1f} ms",
          f"p95={m['delay_p95_ms']:.1f} ms")
PY
```

Transport repetitions quantify delivery variability; they are not additional
physical-sequence replicates. The physical sequence remains the primary
statistical unit. If a frozen trajectory is longer than a capture, the measured
packet pattern repeats, and this is recorded in
`netem_trace_application_statistics.csv`.

## Cleanup after interruption

The script has an exit trap. If the VM loses power while it is running, inspect
and remove only namespaces whose names start with `dtns_`:

```bash
ip netns list
sudo ip netns delete THE_EXACT_DTNS_NAME
```

Do not apply `tc` directly to the VM's real network interface for this study.
