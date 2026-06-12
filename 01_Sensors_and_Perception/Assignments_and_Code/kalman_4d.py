import numpy as np
import matplotlib.pyplot as plt

dt = 0.2
num_steps = 50

# 1. Initialize Matrices
X = np.array([[10.0],  # x
              [4.8],  # y
              [3],  # x_dot
              [-1]]) # y_dot

P = np.eye(4) * 10.0

F = np.array([[1.0, 0.0,  dt, 0.0],
              [0.0, 1.0, 0.0,  dt],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])

H = np.array([[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0]])

R = np.eye(2) * 0.5
Q = np.eye(4) * 0.1
I = np.eye(4)

# Simulated raw sensor measurements [x, y]
simulated_measurements = [np.array([[0.6], [0.1]]), 
                          np.array([[1.1], [-0.3]]), 
                          np.array([[1.7], [-0.4]])]

# 2. Execution Loop for first 3 steps
for step, z in enumerate(simulated_measurements):
    # --- PREDICT ---
    X_pred = F @ X
    P_pred = F @ P @ F.T + Q
    
    # --- UPDATE ---
    # TODO: Calculate the Innovation Vector (y). Hint: z is a 2x1 vector. H @ X_pred is a 2x1 vector.
    y = z - (H @ X_pred)
    
    # TODO: Calculate Innovation Covariance (S). Formula: S = H @ P_pred @ H.T + R
    S = H @ P_pred @ H.T + R
    
    # TODO: Calculate Kalman Gain (K). Formula: K = P_pred @ H.T @ inv(S)
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    # TODO: Compute the updated State Vector (X)
    X = X_pred + K @ y
    
    # TODO: Compute the updated Covariance Matrix (P)
    P = (I - K @ H) @ P_pred
    
    print(f"Step {step+1} Estimated Position X: {X[0,0]:.2f}, Y: {X[1,0]:.2f} | Estimated Velocity X_dot: {X[2,0]:.2f}")
