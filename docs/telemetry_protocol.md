# Telemetry Protocol

The firmware and Python simulator use the same packed little-endian packet.
Packets may be transported as raw bytes over Serial/UDP or hex text over bridge
interfaces that are safer with printable strings.

## Frame

```text
magic           uint32   0x31445444, bytes "DTD1"
version         uint8    currently 1
flags           uint8    reserved
payload_len     uint16   currently 64
seq             uint32
timestamp_us    uint64
enc_left_ticks  int32
enc_right_ticks int32
accel_x         float32  m/s^2
accel_y         float32  m/s^2
accel_z         float32  m/s^2
gyro_z          float32  rad/s
gps_lat_deg     float64
gps_lon_deg     float64
gps_speed_mps   float32
gps_course_rad  float32
gps_fix_type    uint8    0 none, 2 2D, 3 3D
gps_satellites  uint8
gps_hdop_cm     uint16   HDOP * 100
crc32           uint32   CRC of all preceding bytes
```

Total frame size: 76 bytes. The IMU and timing fields feed the proposal's
rolling uncertainty features `sigma_IMU` and `Delta t_k`; the GPS/encoder
residual feeds `r_k`.

## Python Round Trip

```python
from DigitalTwin.telemetry import TelemetryPacket

packet = TelemetryPacket(seq=1, timestamp_us=100000, enc_left_ticks=10, enc_right_ticks=11)
raw = packet.pack()
parsed = TelemetryPacket.unpack(raw)
```

For bridge-safe text:

```python
line = packet.to_hex()
parsed = TelemetryPacket.from_hex(line)
```

## Integration Rule

The Arduino, simulator, WiFi bridge, and logger must all preserve this packet
contract. If a new sensor is added later, increment `version` and document the
new frame instead of silently changing field order.
