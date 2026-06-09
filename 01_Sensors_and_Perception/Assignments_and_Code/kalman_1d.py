import numpy as np
import matplotlib.pyplot as plt

# 1. Setup simulated true state and noisy measurements
true_value = 25.0  # e.g., Constant target temperature
num_steps =100
np.random.seed(42)

# Generate noisy sensor measurements (variance = 4.0)
sensor_variance = 4.0
measurements = true_value + np.random.normal(0, np.sqrt(sensor_variance), num_steps)

# 2. Initialize Kalman Filter variables
initial_estimate = 20.0  # The robot starts with a bad guess
initial_variance = 20.0  # The robot is highly uncertain at first
motion_variance = 0.1   # The target doesn't move much, very low process noise

# Arrays to store results for plotting
estimates = []
variances = []

current_estimate = initial_estimate
current_variance = initial_variance

# 3. The Kalman Filter Loop
for z in measurements:
    # --- PHASE 1: PREDICT ---
    # Assume static target for this assignment, so prediction equals past state
    prediction = current_estimate
    prediction_variance = current_variance + motion_variance
    
    # --- PHASE 2: UPDATE (Write your math here) ---
    # TODO: Calculate Kalman Gain (K)
    kalman_gain = prediction_variance / (prediction_variance + sensor_variance)
    # TODO: Calculate current_estimate using the update equation
    current_estimate = prediction + kalman_gain * (z - prediction)
    # TODO: Calculate current_variance using the covariance update equation
    current_variance = (1 - kalman_gain) * prediction_variance
    
    # Store results
    estimates.append(current_estimate)
    variances.append(current_variance)

# 4. Data Visualization
plt.figure(figsize=(10,6))
plt.plot(range(num_steps), [true_value]*num_steps, 'g--', label='Ground Truth')
plt.plot(range(num_steps), measurements, 'ro', alpha=0.5, label='Noisy Measurements')
plt.plot(range(num_steps), estimates, 'b-', linewidth=2, label='Kalman Filter Estimate')
plt.title('1D Kalman Filter Implementation')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.savefig('kalman_plot_2.png')
