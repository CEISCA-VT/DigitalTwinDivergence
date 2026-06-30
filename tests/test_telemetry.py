from DigitalTwin.telemetry import TelemetryPacket


def test_packet_round_trip():
    packet = TelemetryPacket(
        seq=3,
        timestamp_us=123456,
        enc_left_ticks=10,
        enc_right_ticks=12,
        gps_lat_deg=40.0,
        gps_lon_deg=-74.0,
        gps_fix_type=3,
        gps_satellites=10,
        gps_hdop_cm=120,
    )
    parsed = TelemetryPacket.unpack(packet.pack())
    assert parsed.seq == packet.seq
    assert parsed.timestamp_us == packet.timestamp_us
    assert parsed.enc_left_ticks == packet.enc_left_ticks
    assert parsed.enc_right_ticks == packet.enc_right_ticks
