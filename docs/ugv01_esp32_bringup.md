# UGV01 ESP32 Bring-Up

The UGV01 uses its onboard ESP32 driver board as the lower-level controller.
It is not the Arduino bridge path in `firmware/sketch/`.

## First Bench Session

1. Charge the UGV01 and keep the tracks lifted/off the ground for command tests.
2. Power on the rover.
3. Check the OLED screen:
   - AP mode should show Wi-Fi hotspot `UGV`.
   - Default AP password is `12345678`.
   - New firmware shows `Version: 0.9`.
   - The voltage line should look healthy before driving.
4. Connect the laptop Wi-Fi to `UGV`.
5. Open Chrome at:

```text
http://192.168.4.1
```

6. Use the web UI first. Verify voltage, heading, IP/MAC, and basic stop/speed
   controls.

## HTTP JSON Smoke Tests

From this repo:

```powershell
python firmware/python/ugv01_http_ctrl.py --cmd '{"T":130}'
python firmware/python/ugv01_http_ctrl.py --cmd '{"T":126}'
python firmware/python/ugv01_http_ctrl.py --cmd '{"T":1,"L":0,"R":0}'
```

GPS telemetry commands added in `ugv01_gps_dev`:

```powershell
python firmware/python/ugv01_http_ctrl.py --cmd '{"T":146}'
python firmware/python/ugv01_http_ctrl.py --cmd '{"T":147}'
```

`T:146` returns GPS-only data. `T:147` returns combined base, encoder, IMU,
voltage, GPS, sequence, and firmware timing fields.

## BN-220 GPS Wiring Note

The firmware keeps the GPS on the UGV01 RX path that was already verified on
the bench. Do not move wires for the telemetry changes. The only GPS setting in
`ugv01_gps_dev/General_Driver/ugv_config.h` is the baud rate:

```cpp
#define GPS_BAUD 9600
```

Your working BN-220 wiring should stay:

```text
BN-220 white wire -> UGV01 RX
BN-220 red wire -> UGV01 5V
BN-220 black wire -> UGV01 GND
```

If `gps_chars` increases in `T:146` or `T:147`, the GPS serial path is working.

## Active Firmware Path

For this project, the active lower-controller firmware is:

```text
ugv01_gps_dev/General_Driver
```

This is the firmware tree that contains the BN220 telemetry additions,
`T:146`, `T:147`, sequence counters, and firmware-side timing fields. Keep the
GPS on the original verified RX path and do not rewire it for these features.

## Flashing And Arduino Settings

Use Arduino IDE with the ESP32 board package installed. The upstream base for
`ugv01_gps_dev` is recorded in:

```text
ugv01_gps_dev/SOURCE_COMMIT.txt
```

Project setup:

1. Open `ugv01_gps_dev/General_Driver/General_Driver.ino`.
2. Confirm required libraries from `ugv01_gps_dev/README.md` are installed.
3. Select an ESP32 board profile compatible with the Waveshare General Driver.
4. Use the board defaults that already compile this sketch in the current repo
   unless a board-specific serial/flash issue forces a change.
5. Confirm `GPS_BAUD` remains `9600`.
6. Compile before upload.
7. Upload over the ESP32 USB/programming path.

Bench acceptance after flashing:

- The OLED boots normally.
- The web UI still loads at `192.168.4.1` in AP mode.
- `{"T":130}` and `{"T":126}` still return valid base/IMU payloads.
- `{"T":146}` returns GPS-only fields.
- `{"T":147}` returns combined telemetry with `seq`, `sample_ms`, `send_ms`,
  encoder counts, IMU fields, and GPS fields.

Useful commands from the UGV01 wiki:

```json
{"T":130}
{"T":126}
{"T":1,"L":0.0,"R":0.0}
{"T":1,"L":0.1,"R":0.1}
{"T":13,"X":0.1,"Z":0.0}
```

The rover has heartbeat safety: if movement commands stop for about 3 seconds,
the chassis should stop automatically. Still, keep the tracks lifted for first
tests and always send a stop command before touching the chassis.

## Network Modes

Default AP mode:

```text
SSID: UGV
Password: 12345678
Robot IP: 192.168.4.1
```

To connect the rover to a known Wi-Fi network, send this from the web UI after
replacing the STA SSID and password:

```json
{"T":404,"ap_ssid":"UGV","ap_password":"12345678","sta_ssid":"your_ssid","sta_password":"password"}
```

After that, the OLED `ST` line should show the router-assigned IP. Use that IP
with `--ip` in the HTTP helper.

## Research Pipeline Note

For the current project, the next milestone is to retrieve UGV01 feedback
reliably and map it into the `DigitalTwin.telemetry.TelemetryPacket` fields.
Do not start attack trials until base feedback, IMU feedback, timing, and
encoder-derived motion are repeatable.

## `T:146` And `T:147` Definitions

`T:146` is GPS-only telemetry intended for bench checking the BN220 receive
path and NMEA decode health. Important fields:

- `seq`: firmware telemetry sequence
- `sample_ms`: firmware sample timestamp from `millis()`
- `send_ms`: firmware send timestamp from `millis()`
- `valid`, `fix_type`, `lat`, `lon`, `sat`, `hdop`
- `chars`, `sentences`, `failed_checksums`

`T:147` is the combined telemetry message used by `bench_logger.py`. Important
fields:

- firmware timing: `seq`, `sample_ms`, `send_ms`, `millis`
- base state: `L`, `R`, `enc_left`, `enc_right`, `v`
- IMU attitude/raw motion: `r`, `p`, `y`, `ax`, `ay`, `az`, `gx`, `gy`, `gz`,
  `mx`, `my`, `mz`, `temp`
- GPS state: `gps_valid`, `gps_fix_type`, `lat`, `lon`, `sat`, `hdop`, `alt_m`,
  `speed_mps`, `course_deg`, `gps_chars`, `gps_sentences`,
  `gps_failed_checksums`

The log-field data dictionary and edge-derived timing fields are documented in
`docs/log_data_dictionary.md`.
