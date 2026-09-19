import pandas as pd
import numpy as np

from DigitalTwin.analysis.netem_delivery_replay import measured_delivery
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
