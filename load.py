import rosbag

# Replace with your actual bag file path
bag_path = '2025-04-09-15-38-20.bag'

import numpy as np

# Open the bag file
bag = rosbag.Bag(bag_path)

# Target topic
topic_name = '/vicon/lcarr_quad/odom'

# Prepare list to collect data
data = []

for _, msg, _ in bag.read_messages(topics=[topic_name]):
    pos_x = msg.pose.pose.position.x
    pos_y = msg.pose.pose.position.y
    twist_x = msg.twist.twist.linear.x
    twist_y = msg.twist.twist.linear.y
    data.append([pos_x, pos_y, twist_x, twist_y])

bag.close()

# Convert to NumPy array
array = np.array(data)
num_msgs = array.shape[0]
start_idx = int(0.01 * num_msgs)
end_idx = int(0.99 * num_msgs)
array = array[start_idx:end_idx]
# Print shape and optionally preview
print("Shape of result array:", array.shape)
print("First 5 rows:\n", array[:5])


import matplotlib.pyplot as plt

# Extract columns
pos_x = array[:, 0]
pos_y = array[:, 1]
vel_x = array[:, 2]
vel_y = array[:, 3]

# Plot
plt.figure(figsize=(10, 6))
plt.title("Trajectory with Velocity Vectors")
plt.xlabel("Position X")
plt.ylabel("Position Y")

# Trajectory
plt.plot(pos_x, pos_y, label="Trajectory", linewidth=2)

# Velocity vectors
skip = max(len(pos_x) // 50, 1)  # Skip some to avoid clutter
plt.quiver(
    pos_x[::skip],
    pos_y[::skip],
    vel_x[::skip],
    vel_y[::skip],
    angles='xy', scale_units='xy', scale=1, color='r', alpha=0.6, label="Velocity"
)

plt.axis('equal')
plt.grid(True)
plt.legend()

# Expand Y-axis limits
plt.ylim(0, 10)


#plt.tight_layout()
plt.show()


from scipy.io import savemat
import numpy as np

# --- Step 1: Trim 5% from both ends ---
num_msgs = array.shape[0]

print("num_msgs ", num_msgs)


# --- Step 2: Subsample ---
subsample_rate = 5  # Keep every 5th message (adjust as needed)
subsampled_array = array[::subsample_rate]

print(subsampled_array.shape)

# # Create a dictionary to store in the .mat file
# data_dict = {
#     'data': subsampled_array.T  # Shape: (N, 4)
# }

#cut half of the data, keep the first half
subsampled_array = subsampled_array[:len(subsampled_array)//2]
print("subsampled_array shape ", subsampled_array.shape)



cell_wrapped = np.empty((1, 1), dtype=object)
cell_wrapped[0, 0] = subsampled_array.T

# Save to .mat file
savemat('trajectory_data.mat', {'data': cell_wrapped})

print("Saved matrix as a single cell to trajectory_data.mat")

print("Saved to trajectory_data.mat")

