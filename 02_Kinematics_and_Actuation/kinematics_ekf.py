import numpy as np
import matplotlib.pyplot as plt
import math

class KinematicEKF:
    def __init__(self, wheel_radius, track_width):
        self.r = wheel_radius
        self.L = track_width
        
        # State Vector: [x, y, theta]
        self.X = np.array([[0.0], [0.0], [0.0]])
        self.P = np.eye(3) * 0.1  # Initial Covariance
        
        # Noise tuning
        self.Q = np.eye(3) * 0.05  # Process Noise (Trust in encoders)
        self.R = np.array([[0.5]]) # Sensor Noise (Trust in beacon)
        
        # Static lab beacon coordinate (mx, my)
        self.beacon = np.array([5.0, 5.0])

    def forward_kinematics(self, omega_L, omega_R):
        """Translates encoder speeds to chassis velocities."""
        v = (self.r * (omega_R + omega_L)) / 2.0
        omega = (self.r * (omega_R - omega_L)) / self.L
        return v, omega

    def ekf_predict(self, v, omega, dt):
        """Moves the state and expands the uncertainty banana (Module 1 + 2)."""
        theta = self.X[2, 0]
        
        # Non-Linear State Move
        self.X[0, 0] += v * math.cos(theta) * dt
        self.X[1, 0] += v * math.sin(theta) * dt
        self.X[2, 0] += omega * dt
        
        # Motion Jacobian (G)
        G = np.array([
            [1.0, 0.0, -v * math.sin(theta) * dt],
            [0.0, 1.0,  v * math.cos(theta) * dt],
            [0.0, 0.0,  1.0]
        ])
        
        self.P = G @ self.P @ G.T + self.Q

    def ekf_update(self, measured_range):
        """Corrects the state using the distance to the beacon."""
        x, y = self.X[0, 0], self.X[1, 0]
        mx, my = self.beacon[0], self.beacon[1]
        
        # Expected distance to beacon
        expected_range = math.sqrt((mx - x)**2 + (my - y)**2)
        
        # Measurement Jacobian (H) - The derivative of the range formula
        dr_dx = -(mx - x) / expected_range if expected_range > 0 else 0.0
        dr_dy = -(my - y) / expected_range if expected_range > 0 else 0.0
        H = np.array([[dr_dx, dr_dy, 0.0]])
        
        # Standard EKF Update Math
        y_innov = np.array([[measured_range]]) - expected_range
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.X = self.X + K @ y_innov
        self.P = (np.eye(3) - K @ H) @ self.P

# --- Integration Simulation ---
if __name__ == "__main__":
    robot = KinematicEKF(wheel_radius=0.05, track_width=0.20)
    dt = 0.1
    
    # Tracking arrays
    true_x, true_y = [0.0], [0.0]
    ekf_x, ekf_y = [0.0], [0.0]
    
    # Simulate 100 steps of driving
    for step in range(100):
        # 1. Physical Reality: Clean target speeds
        target_v = 1.0
        target_w = 0.2
        true_theta = true_x[-1] * 0.0 # Simplified true state tracking
        
        # Calculate true position (perfect physics)
        tx = true_x[-1] + target_v * math.cos(step * target_w * dt) * dt
        ty = true_y[-1] + target_v * math.sin(step * target_w * dt) * dt
        true_x.append(tx)
        true_y.append(ty)
        
        # 2. Hardware Flaw: Encoders read slightly wrong (Wheel slip)
        # Using Inverse Kinematics to find perfect wheel speeds, then adding error
        perfect_wL = (target_v - (target_w * 0.2 / 2)) / 0.05
        perfect_wR = (target_v + (target_w * 0.2 / 2)) / 0.05
        
        noisy_wL = perfect_wL - 0.5 # Left wheel slips
        noisy_wR = perfect_wR
        
        # 3. Perception: Measure distance to beacon with some sensor noise
        true_range = math.sqrt((5.0 - tx)**2 + (5.0 - ty)**2)
        measured_range = true_range + np.random.normal(0, 0.2)
        
        # --- THE EKF PIPELINE ---
        v, omega = robot.forward_kinematics(noisy_wL, noisy_wR)
        robot.ekf_predict(v, omega, dt)
        
        # Apply the sensor correction every 5 steps
        if step % 5 == 0:
            robot.ekf_update(measured_range)
            
        ekf_x.append(robot.X[0, 0])
        ekf_y.append(robot.X[1, 0])

    plt.figure(figsize=(10, 8))
    plt.plot(true_x, true_y, 'g--', label="True Physical Path")
    plt.plot(ekf_x, ekf_y, 'b-', linewidth=2, label="EKF Filtered Path")
    plt.plot(5.0, 5.0, 'r*', markersize=15, label="Known Beacon")
    plt.title("Capstone: Kinematics + EKF Integration")
    plt.legend()
    plt.grid()
    plt.savefig("capstone_integration.png")
   