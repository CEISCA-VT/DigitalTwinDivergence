# Log Data Dictionary

This document defines the main fields used in the pre-battery stationary and
bench telemetry logs. The active embedded source is
`ugv01_gps_dev/General_Driver`, and the verified BN220 wiring remains:

```text
BN220 white -> UGV01 RX
BN220 red   -> 5V
BN220 black -> GND
```

## Field Conventions

- `firmware-side`: emitted by the ESP32 sketch
- `edge-side`: added by the laptop/logger
- `derived`: computed later during replay or analysis
- Blank values are expected when a field is not available in an older log
  schema, when GPS is invalid, or when an alarm did not fire.

## Bench Telemetry Log (`raw_logs/telemetry/*.csv`)

| Field | Meaning | Units | Source | Blank expected |
| --- | --- | --- | --- | --- |
| `sample_idx` | Logger cycle index | count | edge-side | no |
| `cycle_ok` | HTTP cycle succeeded | bool | edge-side | no |
| `error` | Logger error text | text | edge-side | yes |
| `t_wall_unix_s` | Wall-clock capture time | s | edge-side | no |
| `t_cycle_start_ns` | Monotonic cycle start time | ns | edge-side | no |
| `t_edge_send_ns` | Monotonic HTTP send time | ns | edge-side | older logs only |
| `t_edge_rx_ns` | Monotonic HTTP receive time | ns | edge-side | no |
| `t_edge_mid_ns` | Midpoint between send and receive | ns | edge-side | older logs only |
| `http_latency_ms` | End-to-end HTTP request latency | ms | edge-side | no |
| `rover_millis_s` | Firmware `millis()` mapped to seconds | s | edge-side | newer logs only |
| `source_sample_time_s` | Firmware sample time in seconds | s | firmware-side mirrored at edge | newer logs only |
| `edge_arrival_time_s` | Edge receive timestamp in seconds | s | edge-side | newer logs only |
| `estimate_time_s` | Time used for EKF/analysis estimate | s | edge-side | newer logs only |
| `alarm_time_s` | Time of detector alarm | s | edge-side/derived | yes |
| `clock_offset_s` | Session offset estimate between firmware clock and edge time | s | edge-side | newer logs only |
| `clock_calibrated` | Offset estimate has enough samples | bool | edge-side | newer logs only |
| `clock_calibration_samples` | Samples used by session calibration | count | edge-side | newer logs only |
| `packet_loss_count` | Sequence gaps before this row | count | edge-side | newer logs only |
| `stale_packet` | Sequence not newer than previous row | bool | edge-side | newer logs only |
| `queue_depth` | Buffered items waiting after delivery | count | edge-side | newer logs only |
| `T` | Telemetry command/feedback ID | count | firmware-side | no |
| `seq` | Firmware telemetry sequence | count | firmware-side | newer logs only |
| `sample_ms` | Firmware sample timestamp from `millis()` | ms | firmware-side | newer logs only |
| `send_ms` | Firmware send timestamp from `millis()` | ms | firmware-side | newer logs only |
| `millis` | Firmware `millis()` value | ms | firmware-side | no |
| `L`, `R` | Left/right commanded or measured speed channels | firmware units | firmware-side | no |
| `enc_left`, `enc_right` | Left/right encoder counts | ticks | firmware-side | newer logs only |
| `v` | Battery voltage | V | firmware-side | no |
| `r`, `p`, `y` | Roll, pitch, yaw | deg | firmware-side | no |
| `ax`, `ay`, `az` | Accelerometer channels | sensor units from firmware | firmware-side | no |
| `gx`, `gy`, `gz` | Gyroscope channels | sensor units from firmware | firmware-side | no |
| `mx`, `my`, `mz` | Magnetometer channels | sensor units from firmware | firmware-side | no |
| `temp` | IMU or board temperature | C | firmware-side | no |
| `gps_valid` | GPS fix validity | bool | firmware-side | no |
| `gps_age_ms` | GPS data age | ms | firmware-side | no |
| `gps_fix_type` | Fix class, typically `0` or `3` here | enum | firmware-side | newer logs only |
| `lat`, `lon` | GPS coordinates | deg | firmware-side | yes when invalid |
| `sat` | Satellites used/seen | count | firmware-side | yes when invalid |
| `hdop` | Horizontal dilution of precision | unitless | firmware-side | yes when invalid |
| `alt_m` | Altitude | m | firmware-side | yes when invalid |
| `speed_mps` | GPS speed | m/s | firmware-side | yes when invalid |
| `course_deg` | GPS course over ground | deg | firmware-side | yes when invalid |
| `gps_chars` | NMEA characters processed | count | firmware-side | no |
| `gps_sentences` | Sentences contributing to fixes | count | firmware-side | no |
| `gps_failed_checksums` | Failed NMEA checksum count | count | firmware-side | no |

## Stationary Multi-Request Log (`raw_logs/static/*.csv`)

| Field | Meaning | Units | Source | Blank expected |
| --- | --- | --- | --- | --- |
| `sample_idx` | Logger cycle index | count | edge-side | older logs only |
| `cycle_ok` | All requested subqueries succeeded | bool | edge-side | older logs may omit |
| `error` | Logger error text | text | edge-side | yes |
| `t_cycle_start_ns` | Monotonic cycle start time | ns | edge-side | newer static logs only |
| `t_edge_rx_ns` | Monotonic receive time | ns | edge-side | no |
| `t_wall_unix_s` | Wall-clock capture time | s | edge-side | no |
| `imu_http_latency_ms` | IMU request latency | ms | edge-side | newer static logs only |
| `base_http_latency_ms` | Base request latency | ms | edge-side | newer static logs only |
| `gps_http_latency_ms` | GPS request latency | ms | edge-side | newer static logs only |
| `imu_*` | IMU telemetry from a dedicated request | mixed | firmware-side | no in IMU-capable logs |
| `base_*` | Base telemetry from a dedicated request | mixed | firmware-side | no in base-capable logs |
| `gps_T` | GPS feedback ID | count | firmware-side | yes in oldest static log |
| `gps_valid` | GPS fix validity | bool | firmware-side | yes in oldest static log |
| `gps_age_ms` | GPS data age | ms | firmware-side | yes in oldest static log |
| `gps_lat`, `gps_lon` | GPS coordinates | deg | firmware-side | yes when invalid |
| `gps_sat` | Satellite count | count | firmware-side | yes when invalid |
| `gps_hdop` | Horizontal dilution of precision | unitless | firmware-side | yes when invalid |
| `gps_alt_m` | Altitude | m | firmware-side | yes when invalid |
| `gps_speed_mps` | GPS speed | m/s | firmware-side | yes when invalid |
| `gps_course_deg` | Course over ground | deg | firmware-side | yes when invalid |
| `gps_chars` | NMEA characters processed | count | firmware-side | yes in oldest static log |
| `gps_sentences` | Fix-contributing sentence count | count | firmware-side | yes in oldest static log |
| `gps_failed_checksums` | Failed NMEA checksum count | count | firmware-side | yes in oldest static log |

## Replay Output Highlights

The replayed digital-twin CSV adds derived fields such as:

- `gps_x_m`, `gps_y_m`
- `ekf_x_m`, `ekf_y_m`, `ekf_theta_rad`
- `innovation_x_m`, `innovation_y_m`
- `mahalanobis`, `threshold`, `lambda_max_s`, `epsilon_min_m`
- `confidence`, `envelope_region`
- `q_xx`, `q_yy`, `q_tt`

These are derived from the raw hardware logs and are checked by
`python -m DigitalTwin.analysis.review_hardware_replay`.
