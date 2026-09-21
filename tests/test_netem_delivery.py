import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np

from DigitalTwin.analysis import netem_delivery_replay as netem_replay
from DigitalTwin.analysis.netem_delivery_replay import discover_trajectory_sources, measured_delivery
from DigitalTwin.analysis.netem_transport import summarize_capture


def test_capture_summary_preserves_loss_and_measured_delay(tmp_path):
    sent = pd.DataFrame(
        {
            "packet_id": [0, 1, 2],
            "source_elapsed_s": [0.0, 0.2, 0.4],
            "send_monotonic_ns": [1_000_000_000, 1_200_000_000, 1_400_000_000],
            "datagram_bytes": [1024, 1024, 1024],
        }
    )
    received = pd.DataFrame(
        {
            "packet_id": [0, 2],
            "source_elapsed_s": [0.0, 0.4],
            "send_monotonic_ns": [1_000_000_000, 1_400_000_000],
            "receive_monotonic_ns": [1_050_000_000, 1_470_000_000],
            "datagram_bytes": [1024, 1024],
        }
    )
    sent_path = tmp_path / "sent.csv"
    received_path = tmp_path / "received.csv"
    sent.to_csv(sent_path, index=False)
    received.to_csv(received_path, index=False)
    qdisc = tmp_path / "tc_qdisc.txt"
    qdisc.write_text("qdisc netem 8001: root refcnt 2 limit 1000 delay 50ms 20ms loss 1%\n")
    summary = summarize_capture(sent_path, received_path, tmp_path, "practical", 1, 5, 50, 20, 1, qdisc)
    assert summary["lost_packets"] == 1
    assert summary["received_packets"] == 2
    assert summary["delay_p50_ms"] == 60.0
    assert summary["tc_qdisc_record"].startswith("qdisc netem")
    ledger = pd.read_csv(tmp_path / "packet_delivery_ledger.csv")
    assert ledger.received.tolist() == [True, False, True]


def test_measured_trace_replay_uses_captured_loss_and_delay():
    time_s = np.arange(0.0, 2.0, 0.1)
    capture = pd.DataFrame(
        {
            "packet_id": [0, 1, 2, 3],
            "received": [True, False, True, True],
            "measured_delay_ms": [50.0, np.nan, 50.0, 50.0],
        }
    )
    delivered, stats = measured_delivery(time_s, capture, rate_hz=5.0)
    assert delivered[0] == -1
    assert delivered[1] == 0
    assert stats["requested_packets"] == 10
    assert stats["lost_packets"] == 3
    assert np.isclose(stats["delay_p50_ms"], 50.0)
    assert np.all((delivered < len(time_s)) & (delivered >= -1))


def test_nested_bank_source_selection_uses_only_verified_outer_tests(tmp_path):
    trajectories = []
    for index in range(30):
        path = tmp_path / f"trajectory_{index}.csv"
        path.write_text("time_s\n0\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        trajectories.append(
            {
                "outer": f"sequence_{index // 3}",
                "role": "outer_test",
                "seed": (42, 1042, 2042)[index % 3],
                "trajectory": str(path),
                "sha256": digest,
            }
        )
    trajectories.append(
        {
            "outer": "sequence_0",
            "role": "qualification_train",
            "seed": 42,
            "trajectory": "not_used.csv",
        }
    )
    manifest = tmp_path / "merged_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "ready_for_evidence_analysis": True,
                "trajectories": trajectories,
            }
        ),
        encoding="utf-8",
    )
    sources = discover_trajectory_sources(manifest)
    assert len(sources) == 30
    assert {source["sequence"] for source in sources} == {f"sequence_{i}" for i in range(10)}


def test_capture_discovery_accepts_repo_relative_root(tmp_path, monkeypatch):
    capture_dir = tmp_path / "captures" / "ideal" / "replicate_01"
    capture_dir.mkdir(parents=True)
    (capture_dir / "capture_manifest.json").write_text(
        json.dumps({"condition": "ideal"}), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "packet_id": [0],
            "received": [True],
            "measured_delay_ms": [0.5],
        }
    ).to_csv(capture_dir / "packet_delivery_ledger.csv", index=False)

    monkeypatch.setattr(netem_replay, "ROOT", tmp_path)
    monkeypatch.setattr(netem_replay, "REQUIRED_CONDITIONS", {"ideal"})
    captures = netem_replay.discover_captures(Path("captures"))

    assert len(captures) == 1
    assert captures[0][0]["condition"] == "ideal"
