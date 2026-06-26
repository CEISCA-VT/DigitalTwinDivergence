import time
import math
import collections

# --- TRAJECTORY CONFIGURATION ---
SQUARE_SIDE = 2.0  # 2 meters
TARGET_SPEED = 0.2  # 0.2 m/s
TIME_STEP = 0.111  # 111ms matching your terminal log

class Week1SimulationEngine:
    def __init__(self):
        # Core Kinematic States
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.sim_time = 111258 # Initial baseline timestamp from your log
        
        # Trajectory Tracker
        self.distance_traveled = 0.0
        self.current_leg = 0 # 0=Straight, 1=Turn, 2=Straight, etc.
        self.leg_stage = "STRAIGHT" # "STRAIGHT" or "TURN"
        
        # --- FAULT INJECTION STATE ---
        self.bias_vector = [0.0, 0.0] # [x_bias, y_bias]
        self.freeze_active = False
        self.replay_active = False
        
        # 20-Second Rolling Replay Buffer (20s / 0.111s = ~180 frames)
        self.replay_buffer = collections.deque(maxlen=180)

    def update_kinematics(self):
        """Generates a perfect 2m x 2m square trajectory"""
        if self.freeze_active:
            # Hard coordinate freezing: Stop updating states, return 0 for ticks
            return self.x + self.bias_vector[0], self.y + self.bias_vector[1], self.theta, 0, 0
        # Basic differential drive simulation variables
        v = 0.0
        omega = 0.0

        if self.leg_stage == "STRAIGHT":
            v = TARGET_SPEED
            omega = 0.0
            self.distance_traveled += v * TIME_STEP
            if self.distance_traveled >= SQUARE_SIDE:
                self.leg_stage = "TURN"
                self.distance_traveled = 0.0
        
        elif self.leg_stage == "TURN":
            v = 0.0
            omega = math.pi / 2.0  # 90 deg/sec turn rate
            self.distance_traveled += abs(omega) * TIME_STEP
            if self.distance_traveled >= (math.pi / 2.0):  # Completed 90 degrees
                self.leg_stage = "STRAIGHT"
                self.distance_traveled = 0.0
                self.current_leg = (self.current_leg + 1) % 4

        # Propagate states using non-linear kinematics
        self.theta += omega * TIME_STEP
        self.x += v * math.cos(self.theta) * TIME_STEP
        self.y += v * math.sin(self.theta) * TIME_STEP
        
        # Dynamic telemetry variables to mimic hardware ticks
        delta_l = 12 if v > 0 else (10 if omega > 0 else 0)
        delta_r = 12 if v > 0 else (-10 if omega > 0 else 0)

        # Apply Fault Injection Biases seamlessly if triggered
        output_x = self.x + self.bias_vector[0]
        output_y = self.y + self.bias_vector[1]

        # Record into rolling replay buffer
        state_frame = (output_x, output_y, self.theta, delta_l, delta_r)
        
        if self.replay_active and len(self.replay_buffer) == self.replay_buffer.maxlen:
            # Override current state with old data from 20 seconds ago
            return self.replay_buffer.popleft()
        else:
            self.replay_buffer.append(state_frame)
            return output_x, output_y, self.theta, delta_l, delta_r

    def trigger_step_bias(self, x_err, y_err):
        self.bias_vector = [x_err, y_err]
        print(f"\n[CRITICAL] Step Bias Vector Injected: (+{x_err}m, +{y_err}m)")

    def trigger_coordinate_freeze(self, status: bool):
        self.freeze_active = status
        print(f"\n[CRITICAL] Hard Coordinate Freeze State changed to: {status}")

    def trigger_replay_attack(self, status: bool):
        self.replay_active = status
        print(f"\n[CRITICAL] 20-Second Rolling Replay Buffer Playback: {status}")

# --- EXECUTION LOOP ---
if __name__ == "__main__":
    sim = Week1SimulationEngine()
    print("Time(ms)   Δ Left    Δ Right   Pose X (m)    Pose Y (m)    Heading (rad)")
    print("-" * 75)
    
    for step in range(1, 100):
        sim.sim_time += 111
        x, y, th, dl, dr = sim.update_kinematics()
        
        # Showcase a dynamic failure injection trigger halfway through testing
        if step == 30:
            sim.trigger_step_bias(0.5, -0.5)
        if step == 60:
            sim.trigger_coordinate_freeze(True)
            
        print(f"{sim.sim_time:<10} {dl:<9} {dr:<9} {x:<13.4f} {y:<13.4f} {th:<12.4f}")
        time.sleep(0.05)