#include <TinyGPSPlus.h>

TinyGPSPlus gps;
uint32_t telemetrySeq = 0;

void gpsInit() {
  Serial.begin(GPS_BAUD);
}

void gpsPoll() {
  while (Serial.available() > 0) {
    gps.encode(static_cast<char>(Serial.read()));
  }
}

void appendGPSFields(const char *prefix) {
  String key;

  key = String(prefix) + "valid";
  jsonInfoHttp[key] = gps.location.isValid();
  key = String(prefix) + "age_ms";
  jsonInfoHttp[key] = gps.location.age();
  key = String(prefix) + "fix_type";
  jsonInfoHttp[key] = gps.location.isValid() ? 3 : 0;

  key = String(prefix) + "lat";
  if (gps.location.isValid()) {
    jsonInfoHttp[key] = gps.location.lat();
  } else {
    jsonInfoHttp[key] = nullptr;
  }

  key = String(prefix) + "lon";
  if (gps.location.isValid()) {
    jsonInfoHttp[key] = gps.location.lng();
  } else {
    jsonInfoHttp[key] = nullptr;
  }

  key = String(prefix) + "sat";
  jsonInfoHttp[key] = gps.satellites.isValid() ? gps.satellites.value() : -1;

  key = String(prefix) + "hdop";
  jsonInfoHttp[key] = gps.hdop.isValid() ? gps.hdop.value() / 100.0 : -1.0;

  key = String(prefix) + "alt_m";
  jsonInfoHttp[key] = gps.altitude.isValid() ? gps.altitude.meters() : -1.0;

  key = String(prefix) + "speed_mps";
  jsonInfoHttp[key] = gps.speed.isValid() ? gps.speed.mps() : -1.0;

  key = String(prefix) + "course_deg";
  jsonInfoHttp[key] = gps.course.isValid() ? gps.course.deg() : -1.0;

  key = String(prefix) + "chars";
  jsonInfoHttp[key] = gps.charsProcessed();
  key = String(prefix) + "sentences";
  jsonInfoHttp[key] = gps.sentencesWithFix();
  key = String(prefix) + "failed_checksums";
  jsonInfoHttp[key] = gps.failedChecksum();
}

void getGPSData() {
  unsigned long sampleMs = millis();
  jsonInfoHttp.clear();

  jsonInfoHttp["T"] = FEEDBACK_GPS_DATA;
  jsonInfoHttp["seq"] = telemetrySeq++;
  jsonInfoHttp["sample_ms"] = sampleMs;
  appendGPSFields("");
  jsonInfoHttp["send_ms"] = millis();

  String payload;
  serializeJson(jsonInfoHttp, payload);
  Serial.println(payload);
}

void getAllTelemetryData() {
  unsigned long sampleMs = millis();
  jsonInfoHttp.clear();

  jsonInfoHttp["T"] = FEEDBACK_ALL_TELEMETRY;
  jsonInfoHttp["seq"] = telemetrySeq++;
  jsonInfoHttp["sample_ms"] = sampleMs;
  jsonInfoHttp["millis"] = sampleMs;

  // Wheel/base feedback
  jsonInfoHttp["L"] = speedGetA;
  jsonInfoHttp["R"] = speedGetB;
  jsonInfoHttp["enc_left"] = static_cast<long>(encoderA.getCount());
  jsonInfoHttp["enc_right"] = static_cast<long>(encoderB.getCount());
  jsonInfoHttp["v"] = loadVoltage_V;

  // IMU attitude
  jsonInfoHttp["r"] = icm_roll;
  jsonInfoHttp["p"] = icm_pitch;
  jsonInfoHttp["y"] = icm_yaw;

  // IMU raw-ish motion fields
  jsonInfoHttp["ax"] = ax;
  jsonInfoHttp["ay"] = ay;
  jsonInfoHttp["az"] = az;

  jsonInfoHttp["gx"] = gx;
  jsonInfoHttp["gy"] = gy;
  jsonInfoHttp["gz"] = gz;

  jsonInfoHttp["mx"] = mx;
  jsonInfoHttp["my"] = my;
  jsonInfoHttp["mz"] = mz;

  jsonInfoHttp["temp"] = temp;

  appendGPSFields("gps_");
  if (gps.location.isValid()) {
    jsonInfoHttp["lat"] = gps.location.lat();
    jsonInfoHttp["lon"] = gps.location.lng();
  } else {
    jsonInfoHttp["lat"] = nullptr;
    jsonInfoHttp["lon"] = nullptr;
  }
  jsonInfoHttp["sat"] = gps.satellites.isValid() ? gps.satellites.value() : -1;
  jsonInfoHttp["hdop"] = gps.hdop.isValid() ? gps.hdop.value() / 100.0 : -1.0;
  jsonInfoHttp["send_ms"] = millis();

  String payload;
  serializeJson(jsonInfoHttp, payload);
  Serial.println(payload);
}
