import math 
import random
import matplotlib.pyplot as plt

# Generate the Data
num_of_points  = 100
time_steps = [i for i in range (num_of_points)]

# Generating the sine wave with added random noise

raw_data = []
for t in time_steps:
  # Base sine wave calculation
  sine_value = math.sin(2*math.pi*t / 50)
  noise = random.gauss(0, 0.15)
  raw_data.append(sine_value + noise)

# Making the Filter
window_size = 5
filtered_data = []

for i in range (len(raw_data)):
  # Boundaries for the Window
  start_index = max (0, i - window_size + 1)
  end_index = i + 1

  window = raw_data[start_index:end_index]

  window_average = sum(window) / len(window)
  filtered_data.append(window_average)

# Plotting the results
plt.figure(figsize=(10, 6))

plt.plot(time_steps, raw_data, label='Raw Noisy Data', color='gray', alpha=0.6, linestyle='--')
plt.scatter(time_steps, raw_data, color='red', alpha=0.7, label='Sensor Samples')

plt.plot(time_steps, filtered_data, label=f'Filtered Data (Window={window_size})', color='blue', linewidth=2)

plt.title('Sensor Data Smoothing via a Moving-Average Filter')
plt.xlabel('Time Step')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig('filter_results.png', dpi=300, bbox_inches='tight')
print("Plot successfully saved to filter_results.png")
