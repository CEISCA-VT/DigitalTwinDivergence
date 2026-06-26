import time
import socket
from arduino.app_utils import App, Bridge

print("[*] Initializing MPU Wireless Telemetry Relay...")

# Target Destination: Your Laptop's Wireless IP address and selected UDP Port
LAPTOP_IP = "10.0.0.150"
UDP_PORT = 5005

# Initialize a low-latency UDP Network Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def loop():
    """This function is called repeatedly by the App framework."""
    try:
        # Call the C++ function on the STM32 MCU over the internal RPC bridge
        telemetry_data = Bridge.call("get_telemetry")
        
        if telemetry_data:
            # Broadcast the telemetry string over Wi-Fi directly to the laptop
            sock.sendto(telemetry_data.encode('utf-8'), (LAPTOP_IP, UDP_PORT))
    except Exception as e:
        print(f"[!] Bridge Relay Error: {e}")
        
    # Enforce strict 10Hz sampling intervals (100ms)
    time.sleep(0.1)

# Execute the application loop context
App.run(user_loop=loop)