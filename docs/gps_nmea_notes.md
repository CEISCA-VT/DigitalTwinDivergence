# GPS Notes for BN-220 / NMEA / UBX

The BN-220 commonly outputs NMEA sentences over UART. Useful sentences:

`GGA`: fix quality, latitude, longitude, satellites, HDOP, altitude.

`RMC`: recommended minimum data, UTC time, validity, latitude, longitude, speed
over ground, course over ground, date.

`VTG`: course over ground and ground speed.

For this project, parse these fields into the frozen telemetry packet:

```text
gps_lat_deg
gps_lon_deg
gps_speed_mps
gps_course_rad
gps_fix_type
gps_satellites
gps_hdop_cm
```

NMEA latitude and longitude are not decimal degrees. They use `ddmm.mmmm` for
latitude and `dddmm.mmmm` for longitude. Convert with:

```text
decimal_deg = degrees + minutes / 60
```

Apply negative signs for south and west hemispheres.

UBX binary configuration can later be used to increase update rate or disable
unneeded NMEA sentences. That is a hardware-integration task, not a Week 0
dependency.
