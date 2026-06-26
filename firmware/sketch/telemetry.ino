#include <Arduino_RouterBridge.h>
#include <math.h>

uint32_t calculation_index = 0;

// Expose this telemetry function over the internal RPC Bridge to the Linux MPU
String get_telemetry() {
  uint32_t timestamp_ms = millis();
  int32_t enc_left = (int32_t)(calculation_index * 12);
  int32_t enc_right = (int32_t)(calculation_index * 11);
  
  float accel_x = 0.02 * sin(calculation_index * 0.5);
  float accel_y = 0.01 * cos(calculation_index * 0.5);
  float accel_z = 9.81 + (0.45 * sin(calculation_index * 2.0));
  float gyro_z = 0.1 * sin(calculation_index * 0.1);
  
  calculation_index++;
  
  // Return a clean CSV string to ensure zero byte-alignment errors across the RPC
  return String(timestamp_ms) + "," + 
         String(enc_left) + "," + 
         String(enc_right) + "," + 
         String(accel_x, 4) + "," + 
         String(accel_y, 4) + "," + 
         String(accel_z, 4) + "," + 
         String(gyro_z, 4);
}

void setup() {
  // Initialize the native dual-core RPC router bridge
  Bridge.begin();
  
  // Register the function name so the Qualcomm Linux processor can call it
  Bridge.provide("get_telemetry", get_telemetry);
}

void loop() {
  // Zephyr RTOS handles the RPC threads natively. 
  // Keep the loop unblocked with a minor yield.
  delay(10);
}