import rosbag
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt
import os



def reparameterize_trajectory(array, ratio = 0.7):



    # reparameter the velocity to make the final velocity to be 0, and it should be kinda smooth

    # Extract velocity components
    new_vel_x = array[:, 2]
    new_vel_y = array[:, 3]

    # Number of steps
    n = int(len(new_vel_x) * ratio)

    print("n: ", n)
    print("length of new_vel_x: ", len(new_vel_x))

    # Create a taper function that starts at 1 and goes to 0 linearly
    taper = np.linspace(1, 0, len(new_vel_x) - n)

    new_vel_x[n: ] = new_vel_x[n:] * taper
    new_vel_y[n: ] = new_vel_y[n:] * taper

    # If needed, update the array
    array[:, 2] = new_vel_x
    array[:, 3] = new_vel_y


    return array






def process(bag_name):

    bag_path = './bag/' + bag_name + '.bag'

    # Open the bag file
    if not bag_path.endswith('.bag'):
        raise ValueError("The provided file is not a .bag file.")
        
    bag = rosbag.Bag(bag_path, 'r')

    #list all topics

    # if any of the topic has "/odom" in it, save to topic_name
    topic_name = '/vicon/lcarr_quad/odom'
    print("Topics in the bag file:")
    for topic in bag.get_type_and_topic_info()[1].keys():
        print(topic)

        if "/odom" in topic:
            topic_name = topic
            print("Found odom topic: ", topic_name)
            break
        else:
            print("No odom topic found, using default: ", topic_name)


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
    start_idx = int(0.05 * num_msgs)
    end_idx = int(0.95 * num_msgs)
    array = array[start_idx:end_idx]
    # Print shape and optionally preview
    print("Shape of result array:", array.shape)
    print("First 5 rows:\n", array[:5])





    if bag_name == "first_flight":

        print("Processing first_flight bag file")
        print("length of array: ", len(array))

        #cut half of the data, keep the last half
        array = array[array.shape[0] // 2: int(array.shape[0] * 0.9)]

    if bag_name == "circle_twice":

        #cut half of the data, keep the 40% of the data
        array = array[:int(array.shape[0] * 0.4)]
        
    
    array = reparameterize_trajectory(array, ratio = 0.8)



    # Extract columns
    pos_x = array[:, 0]
    pos_y = array[:, 1]
    vel_x = array[:, 2]
    vel_y = array[:, 3]

    # Plot
    plt.figure(figsize=(5, 6))
    plt.title("Trajectory: " + bag_name)
    plt.xlabel("X1")
    plt.ylabel("X2")

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
    plt.xlim(-4, 4)

    # save this image
    folder = './img/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(folder + 'trajectory_plot' + bag_name + '.png', dpi=300)

    #plt.tight_layout()
    plt.show()




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
    cell_wrapped = np.empty((1, 1), dtype=object)
    cell_wrapped[0, 0] = subsampled_array.T

    # Save to .mat file
    folder = './trajectory_data/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    savemat(folder + 'trajectory_data' + bag_name + '.mat', {'data': cell_wrapped})



    print("Saved matrix as a single cell to trajectory_data.mat")
    print("Saved to trajectory_data" + bag_name + ".mat")





if __name__ == "__main__":


        # Replace with your actual bag file path
    bags_path = './bag/'
    bag_names = []
    #traverse the directory and find all .bag files
    for root, dirs, files in os.walk(bags_path):
        for file in files:
            if file.endswith('.bag'):

                bag_name = file.split('.')[0]

                bag_names.append(bag_name)
                print("Found bag file: ", file)
                #print("Found bag file: ", os.path.join(root, file))
    # print("Found bag file: ", os.path.join(root, file))


    print("Found bag files: ", bag_names)

    for bag_name in bag_names:
        print("Processing bag file: ", bag_name)
        process(bag_name)
        print("Done processing bag file: ", bag_name)
        print("=========================================")

   
