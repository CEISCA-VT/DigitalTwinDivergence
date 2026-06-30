"""Telemetry packet schema and serialization helpers.

The firmware emits the same packed packet represented here.  During Week 0 the
synthetic simulator can produce identical frames, so hardware integration later
is mostly swapping the source of bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import zlib


MAGIC_BYTES = b"DTD1"
MAGIC_U32 = 0x31445444
VERSION = 1
HEADER_FORMAT = "<IBBH"
FRAME_FORMAT = "<IBBHIQiiffffddffBBHI"
FRAME_SIZE = struct.calcsize(FRAME_FORMAT)
CRC_SIZE = 4
PAYLOAD_LEN = FRAME_SIZE - struct.calcsize(HEADER_FORMAT) - CRC_SIZE
EARTH_RADIUS_M = 6_378_137.0


class TelemetryError(ValueError):
    """Raised when a telemetry frame is malformed."""


@dataclass(slots=True)
class TelemetryPacket:
    seq: int
    timestamp_us: int
    enc_left_ticks: int
    enc_right_ticks: int
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 9.81
    gyro_z: float = 0.0
    gps_lat_deg: float = 0.0
    gps_lon_deg: float = 0.0
    gps_speed_mps: float = 0.0
    gps_course_rad: float = 0.0
    gps_fix_type: int = 0
    gps_satellites: int = 0
    gps_hdop_cm: int = 9999
    flags: int = 0

    def pack(self) -> bytes:
        without_crc = struct.pack(
            FRAME_FORMAT[:-1],
            MAGIC_U32,
            VERSION,
            self.flags & 0xFF,
            PAYLOAD_LEN,
            self.seq & 0xFFFFFFFF,
            self.timestamp_us & 0xFFFFFFFFFFFFFFFF,
            int(self.enc_left_ticks),
            int(self.enc_right_ticks),
            float(self.accel_x),
            float(self.accel_y),
            float(self.accel_z),
            float(self.gyro_z),
            float(self.gps_lat_deg),
            float(self.gps_lon_deg),
            float(self.gps_speed_mps),
            float(self.gps_course_rad),
            int(self.gps_fix_type) & 0xFF,
            int(self.gps_satellites) & 0xFF,
            int(self.gps_hdop_cm) & 0xFFFF,
        )
        crc = zlib.crc32(without_crc) & 0xFFFFFFFF
        return without_crc + struct.pack("<I", crc)

    def to_hex(self) -> str:
        return self.pack().hex()

    @classmethod
    def unpack(cls, frame: bytes) -> "TelemetryPacket":
        if len(frame) != FRAME_SIZE:
            raise TelemetryError(f"expected {FRAME_SIZE} bytes, got {len(frame)}")

        fields = struct.unpack(FRAME_FORMAT, frame)
        (
            magic,
            version,
            flags,
            payload_len,
            seq,
            timestamp_us,
            enc_left_ticks,
            enc_right_ticks,
            accel_x,
            accel_y,
            accel_z,
            gyro_z,
            gps_lat_deg,
            gps_lon_deg,
            gps_speed_mps,
            gps_course_rad,
            gps_fix_type,
            gps_satellites,
            gps_hdop_cm,
            crc,
        ) = fields

        if magic != MAGIC_U32:
            raise TelemetryError(f"bad magic 0x{magic:08x}")
        if version != VERSION:
            raise TelemetryError(f"unsupported version {version}")
        if payload_len != PAYLOAD_LEN:
            raise TelemetryError(f"bad payload length {payload_len}")

        expected_crc = zlib.crc32(frame[:-CRC_SIZE]) & 0xFFFFFFFF
        if crc != expected_crc:
            raise TelemetryError(f"crc mismatch got 0x{crc:08x}, expected 0x{expected_crc:08x}")

        return cls(
            seq=seq,
            timestamp_us=timestamp_us,
            enc_left_ticks=enc_left_ticks,
            enc_right_ticks=enc_right_ticks,
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z,
            gyro_z=gyro_z,
            gps_lat_deg=gps_lat_deg,
            gps_lon_deg=gps_lon_deg,
            gps_speed_mps=gps_speed_mps,
            gps_course_rad=gps_course_rad,
            gps_fix_type=gps_fix_type,
            gps_satellites=gps_satellites,
            gps_hdop_cm=gps_hdop_cm,
            flags=flags,
        )

    @classmethod
    def from_hex(cls, text: str) -> "TelemetryPacket":
        return cls.unpack(bytes.fromhex(text.strip()))


def deserialize_stream(buffer: bytes) -> tuple[list[TelemetryPacket], bytes]:
    """Parse all complete frames from a byte buffer and return leftovers."""
    packets: list[TelemetryPacket] = []
    cursor = 0
    while True:
        start = buffer.find(MAGIC_BYTES, cursor)
        if start < 0:
            return packets, b""
        if len(buffer) - start < FRAME_SIZE:
            return packets, buffer[start:]
        candidate = buffer[start : start + FRAME_SIZE]
        try:
            packets.append(TelemetryPacket.unpack(candidate))
            cursor = start + FRAME_SIZE
        except TelemetryError:
            cursor = start + 1


def local_xy_to_gps(x_m: float, y_m: float, origin_lat_deg: float, origin_lon_deg: float) -> tuple[float, float]:
    lat0 = math.radians(origin_lat_deg)
    lat = origin_lat_deg + math.degrees(y_m / EARTH_RADIUS_M)
    lon = origin_lon_deg + math.degrees(x_m / (EARTH_RADIUS_M * math.cos(lat0)))
    return lat, lon


def gps_to_local_xy(lat_deg: float, lon_deg: float, origin_lat_deg: float, origin_lon_deg: float) -> tuple[float, float]:
    lat0 = math.radians(origin_lat_deg)
    x = math.radians(lon_deg - origin_lon_deg) * EARTH_RADIUS_M * math.cos(lat0)
    y = math.radians(lat_deg - origin_lat_deg) * EARTH_RADIUS_M
    return x, y
