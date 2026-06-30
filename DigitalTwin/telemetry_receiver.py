"""UDP receiver for hardware telemetry.

The Arduino bridge currently sends hex text. Raw framed bytes are also accepted.
"""

from __future__ import annotations

import argparse
import socket

from .telemetry import TelemetryError, TelemetryPacket


def parse_datagram(data: bytes) -> TelemetryPacket:
    text = data.strip()
    try:
        return TelemetryPacket.from_hex(text.decode("ascii"))
    except (UnicodeDecodeError, ValueError, TelemetryError):
        return TelemetryPacket.unpack(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    print(f"listening on {args.host}:{args.port}")
    while True:
        data, addr = sock.recvfrom(4096)
        try:
            packet = parse_datagram(data)
            print(
                f"{addr[0]} seq={packet.seq} t={packet.timestamp_us} "
                f"ticks=({packet.enc_left_ticks},{packet.enc_right_ticks}) "
                f"gps=({packet.gps_lat_deg:.7f},{packet.gps_lon_deg:.7f})"
            )
        except TelemetryError as exc:
            print(f"bad packet from {addr[0]}: {exc}")


if __name__ == "__main__":
    main()
