import numpy as np
import matplotlib.pyplot as plt
import math

class OdometryTracker:
    def __init__(self, wheel_radius, track_width):
        """Initialize physical chassis constraints and starting state."""
        self.r = wheel_radius
        self.L = track_width
        
        # State Vector: [x, y, theta]
        self.state = np.array([0.0, 0.0, 0.0])
        
        # Tracking history for plotting
        self.history_x = [0.0]
        self.history_y = [0.0]

    def forward_kinematics(self, omega_L, omega_R):
        """
        Convert raw wheel encoder speeds into chassis velocities.
        """
        v = (self.r * (omega_R + omega_L)) / 2.0
        omega = (self.r * (omega_R - omega_L)) / self.L
        return v, omega

    def update_state(self, v, omega, dt):
        """
        Update the robot's X, Y, and Theta coordinates using Dead Reckoning.
        """
        current_theta = self.state[2]
        
        x_new = self.state[0] + v * math.cos(current_theta) * dt
        y_new = self.state[1] + v * math.sin(current_theta) * dt
        theta_new = current_theta + omega * dt
        
        new_x = 0.0
        new_y = 0.0
        new_theta = 0.0
        
        self.state = np.array([new_x, new_y, new_theta])
        
        # Save history
        self.history_x.append(new_x)
        self.history_y.append(new_y)

# --- Simulation Execution ---
if __name__ == "__main__":
    # Standard laboratory robot parameters
    robot = OdometryTracker(wheel_radius=0.05, track_width=0.20)
    dt = 0.1
    
    # Simulated encoder data (rad/s) for 50 time steps
    # The right wheel spins slightly slower, simulating the error discussed previously
    encoder_left = [10.0] * 50
    encoder_right = [9.5] * 50
    
    for w_L, w_R in zip(encoder_left, encoder_right):
        # Step 1: Read encoders and calculate velocities
        v, omega = robot.forward_kinematics(w_L, w_R)
        
        # Step 2: Update internal map
        robot.update_state(v, omega, dt)
        
    # Plot the resulting internal map trajectory
    plt.plot(robot.history_x, robot.history_y, label="Dead Reckoning Path", color='b')
    plt.title("Odometry Tracking (Wheel Slip Simulation)")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.legend()
    plt.grid()
    plt.savefig("odemetry_tracker_1")
