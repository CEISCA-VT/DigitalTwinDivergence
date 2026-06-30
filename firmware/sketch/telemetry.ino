#include <Arduino_RouterBridge.h>
#include <math.h>
#include <string.h>

static const uint32_t TELEMETRY_MAGIC = 0x31445444UL; // bytes "DTD1"
static const uint8_t TELEMETRY_VERSION = 1;

struct __attribute__((packed)) TelemetryPacket {
  uint32_t magic;
  uint8_t version;
  uint8_t flags;
  uint16_t payload_len;
  uint32_t seq;
  uint64_t timestamp_us;
  int32_t enc_left_ticks;
  int32_t enc_right_ticks;
  float accel_x;
  float accel_y;
  float accel_z;
  float gyro_z;
  double gps_lat_deg;
  double gps_lon_deg;
  float gps_speed_mps;
  float gps_course_rad;
  uint8_t gps_fix_type;
  uint8_t gps_satellites;
  uint16_t gps_hdop_cm;
  uint32_t crc32;
};

static_assert(sizeof(TelemetryPacket) == 76, "TelemetryPacket size must match Python parser");

uint32_t calculation_index = 0;

uint32_t crc32_update(uint32_t crc, const uint8_t data) {
  crc ^= data;
  for (uint8_t i = 0; i < 8; i++) {
    const uint32_t mask = -(crc & 1UL);
    crc = (crc >> 1) ^ (0xEDB88320UL & mask);
  }
  return crc;
}

uint32_t crc32_bytes(const uint8_t* data, size_t length) {
  uint32_t crc = 0xFFFFFFFFUL;
  for (size_t i = 0; i < length; i++) {
    crc = crc32_update(crc, data[i]);
  }
  return ~crc;
}

TelemetryPacket build_packet() {
  TelemetryPacket packet;
  memset(&packet, 0, sizeof(packet));

  packet.magic = TELEMETRY_MAGIC;
  packet.version = TELEMETRY_VERSION;
  packet.flags = 0;
  packet.payload_len = sizeof(TelemetryPacket) - 8 - sizeof(packet.crc32);
  packet.seq = calculation_index;
  packet.timestamp_us = (uint64_t)millis() * 1000ULL;
  packet.enc_left_ticks = (int32_t)(calculation_index * 12);
  packet.enc_right_ticks = (int32_t)(calculation_index * 11);
  packet.accel_x = 0.02f * sin(calculation_index * 0.5f);
  packet.accel_y = 0.01f * cos(calculation_index * 0.5f);
  packet.accel_z = 9.81f + (0.45f * sin(calculation_index * 2.0f));
  packet.gyro_z = 0.1f * sin(calculation_index * 0.1f);

  // Placeholder GPS fields until the BN-220 is wired in.
  packet.gps_lat_deg = 40.0000000;
  packet.gps_lon_deg = -74.0000000;
  packet.gps_speed_mps = 0.20f;
  packet.gps_course_rad = 0.0f;
  packet.gps_fix_type = 3;
  packet.gps_satellites = 10;
  packet.gps_hdop_cm = 120;

  packet.crc32 = crc32_bytes((const uint8_t*)&packet, sizeof(packet) - sizeof(packet.crc32));
  calculation_index++;
  return packet;
}

String bytes_to_hex(const uint8_t* data, size_t length) {
  const char hex[] = "0123456789abcdef";
  String out;
  out.reserve(length * 2);
  for (size_t i = 0; i < length; i++) {
    out += hex[(data[i] >> 4) & 0x0F];
    out += hex[data[i] & 0x0F];
  }
  return out;
}

// Bridge-safe serializer: returns printable hex for WiFi relay scripts.
String get_telemetry() {
  TelemetryPacket packet = build_packet();
  return bytes_to_hex((const uint8_t*)&packet, sizeof(packet));
}

// Serial serializer: call this from loop if raw UART streaming is preferred.
void write_telemetry_serial() {
  TelemetryPacket packet = build_packet();
  Serial.write((const uint8_t*)&packet, sizeof(packet));
}

void setup() {
  Bridge.begin();
  Bridge.provide("get_telemetry", get_telemetry);
}

void loop() {
  delay(10);
}
