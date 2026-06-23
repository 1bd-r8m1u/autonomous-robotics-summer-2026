class DiffDriveKinematics:
    def __init__(self, wheel_radius, track_width):
        """Initialize the physical constraints of the chassis."""
        self.r = wheel_radius
        self.L = track_width

    def inverse_kinematics(self, v, omega):
        """
        Convert target chassis velocities into individual wheel speeds.
        v: target linear velocity (m/s)
        omega: target angular velocity (rad/s)
        Returns: (omega_L, omega_R) in rad/s
        """
        omega_L = (v - (omega * self.L / 2.0)) / self.r
        omega_R = (v + (omega * self.L / 2.0)) / self.r
        return omega_L, omega_R

    def forward_kinematics(self, omega_L, omega_R):
        """
        Convert actual wheel speeds back into chassis velocities.
        Useful for odometry (tracking where the robot actually went).
        """
        v = self.r * (omega_R + omega_L) / 2.0
        omega = self.r * (omega_R - omega_L) / self.L
        return v, omega

# --- Testing Block ---
if __name__ == "__main__":
    # Physical specs of a typical small lab robot (like a TurtleBot)
    robot = DiffDriveKinematics(wheel_radius=0.033, track_width=0.160)

    # The navigation stack commands a smooth right turn
    target_v = 0.5        # m/s
    target_omega = -1.0    # rad/s (negative is right turn)

    print(f"Commanded Trajectory -> v: {target_v} m/s, omega: {target_omega} rad/s")

    # Calculate required wheel speeds
    w_L, w_R = robot.inverse_kinematics(target_v, target_omega)
    print(f"Calculated Wheel Speeds -> Left: {w_L:.2f} rad/s, Right: {w_R:.2f} rad/s")
    
    # Verify via forward kinematics
    v_verify, omega_verify = robot.forward_kinematics(w_L, w_R)
    print(f"Verified via Forward Kinematics -> v: {v_verify:.2f} m/s, omega: {omega_verify:.2f} rad/s")
