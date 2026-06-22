import numpy as np
import matplotlib.pyplot as plt

# 1. Physical Parameters
dt = 0.1
num_steps = 100
landmark_pos = np.array([5.0, 5.0])  # Static beacon coordinate (mx, my)

# 2. State Initialization [x, y, theta, velocity]
X = np.array([[0.0], [0.0], [0.0], [2.0]])
P = np.eye(4) * 1.0
Q = np.eye(4) * 0.05
R = np.array([[0.1]])  # Sensor variance for range tracking

def predict_step(X_state, P_cov, v, omega, dt):
    """Moves the robot and updates the uncertainty banana."""
    theta = X_state[2, 0]
    
    # Non-Linear State Move
    X_next = np.array([
        [X_state[0, 0] + v * np.cos(theta) * dt],
        [X_state[1, 0] + v * np.sin(theta) * dt],
        [X_state[2, 0] + omega * dt],
        [v]
    ])
    
    # 4x4 Motion Jacobian Matrix (G)
    G = np.eye(4)
    G[0, 2] = -v * np.sin(theta) * dt  # dx / d_theta
    G[0, 3] = np.cos(theta) * dt       # dx / dv
    G[1, 2] = v * np.cos(theta) * dt   # dy / d_theta
    G[1, 3] = np.sin(theta) * dt       # dy / dv
    
    P_next = G @ P_cov @ G.T + Q
    return X_next, P_next

def update_step(X_pred, P_pred, z_range, landmark):
    """Linearizes the measurement model and applies the correction pass."""
    mx, my = landmark[0], landmark[1]
    x, y = X_pred[0, 0], X_pred[1, 0]
    
    # Expected Range (Non-linear geometric distance to beacon)
    expected_range = np.sqrt((mx - x)**2 + (my - y)**2)
    
    # --- LINEARIZATION OF MEASUREMENT (The H Jacobian) ---
    # The derivative of sqrt((mx-x)^2 + (my-y)^2) with respect to x and y
    dr_dx = -(mx - x) / expected_range
    dr_dy = -(my - y) / expected_range
    
    # Formulate H matrix: [dr_dx, dr_dy, dr_dtheta, dr_dv]
    H = np.array([[dr_dx, dr_dy, 0.0, 0.0]])
    
    # --- STANDARD KALMAN FILTER OPERATIONS ---
    y_innovation = np.array([[z_range]]) - expected_range
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    X_updated = X_pred + K @ y_innovation
    P_updated = (np.eye(4) - K @ H) @ P_pred
    return X_updated, P_updated

# 3. Execution Simulation Loop
# True control inputs (constant velocity and slight turn rate)
v_true = 2.0
omega_true = 0.1

# Storage for plotting results
true_trajectory = []
estimated_trajectory = []

# Generate a true ground truth state for simulation comparison
X_true = np.copy(X)

for step in range(num_steps):
    # --- SIMULATE GROUND TRUTH & NOISY MEASUREMENT ---
    # Update true state (no noise)
    theta_true = X_true[2, 0]
    X_true = np.array([
        [X_true[0, 0] + v_true * np.cos(theta_true) * dt],
        [X_true[1, 0] + v_true * np.sin(theta_true) * dt],
        [X_true[2, 0] + omega_true * dt],
        [v_true]
    ])
    
    # Generate actual range to landmark with Gaussian sensor noise
    true_range = np.sqrt((landmark_pos[0] - X_true[0, 0])**2 + (landmark_pos[1] - X_true[1, 0])**2)
    z_noisy_range = true_range + np.random.normal(0, np.sqrt(R[0, 0]))
    
    # --- EKF STEPS ---
    # 1. Predict
    X_pred, P_pred = predict_step(X, P, v_true, omega_true, dt)
    
    # 2. Update (Correct)
    X, P = update_step(X_pred, P_pred, z_noisy_range, landmark_pos)
    
    # Record positions
    true_trajectory.append((X_true[0, 0], X_true[1, 0]))
    estimated_trajectory.append((X[0, 0], X[1, 0]))

# Convert logs to numpy arrays for plotting
true_trajectory = np.array(true_trajectory)
estimated_trajectory = np.array(estimated_trajectory)

# 4. Visualization
plt.figure(figsize=(10, 6))
plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'g-', label='True Path')
plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 'b--', label='EKF Estimate')
plt.scatter(landmark_pos[0], landmark_pos[1], color='red', marker='*', s=150, label='Landmark Beacon')
plt.title('Extended Kalman Filter (EKF) Localization')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('EKF_Localization_1')
