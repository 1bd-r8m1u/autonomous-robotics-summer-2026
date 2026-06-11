import numpy as np
import matplotlib.pyplot as plt

# 1. Simulation Setup
dt = 0.1
num_steps = 100
true_velocity = 2.0  # Constant velocity (m/s)

# Generate true positions and add random noise
time = np.arange(num_steps) * dt
true_positions = true_velocity * time
sensor_variance = 0.5
noisy_measurements = true_positions + np.random.normal(0, np.sqrt(sensor_variance), num_steps)

# 2. Initialize Kalman Filter Matrices
X = np.array([[0.0],   # Initial Position Guess
              [0.0]])  # Initial Velocity Guess

P = np.array([[10.0, 0.0],
              [0.0, 10.0]])

F = np.array([[1.0,  dt],
              [0.0, 1.0]])

H = np.array([[1.0, 0.0]])

R = np.array([[sensor_variance]])

Q = np.array([[0.01, 0.0],
              [0.0, 0.01]])

I = np.eye(2)

# Tracking arrays
est_positions = []
est_velocities = []

# 3. The 2D Kalman Loop
for z in noisy_measurements:
    # --- PHASE 1: PREDICT ---
    X_matrix_predict = F @ X
    P_matrix_predict = F @ P @ F.T + Q
    
    # --- PHASE 2: UPDATE ---
    # y = z - Hx
    y = np.array([[z]]) - (H @ X_matrix_predict)
    
    # TODO: Calculate S (Innovation Covariance) Hint: S = H @ P_predict @ H.T + R
    S = H @ P_matrix_predict @ H.T + R
    
    # TODO: Calculate Kalman Gain K. Hint: Use np.linalg.inv(S) to invert S
    K = P_matrix_predict @ H.T @ np.linalg.inv(S)
    
    # TODO: Update the state vector X
    X = X_matrix_predict + K @ y
    
    # TODO: Update the covariance matrix P
    P = (I - K @ H) @ P_matrix_predict
    
    # Save current estimates
    est_positions.append(X[0, 0])
    est_velocities.append(X[1, 0])

# 4. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(time, true_positions, 'g--', label='True Position')
ax1.plot(time, noisy_measurements, 'ro', alpha=0.3, label='Noisy Laser Measurement')
ax1.plot(time, est_positions, 'b-', linewidth=2, label='Kalman Position Estimate')
ax1.set_ylabel('Position (meters)')
ax1.legend()

ax2.plot(time, [true_velocity]*num_steps, 'g--', label='True Velocity')
ax2.plot(time, est_velocities, 'b-', linewidth=2, label='Kalman Velocity Estimate')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_xlabel('Time (seconds)')
ax2.legend()

plt.savefig('kalman_2d_1.png')
