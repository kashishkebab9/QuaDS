import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import bagpy
from bagpy import bagreader   # this contains a class that does all the hard work of reading bag files

import os
import sys

bag_dir = os.path.join(os.path.dirname(__file__), '..', 'bags')
fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
bag_files = [f for f in os.listdir(bag_dir) if f.endswith('.bag')]

print("Choose the bag file you want.")
for i, bagf in enumerate(bag_files):
    print(str(i)+": "+bagf)
bag_idx = int(input("Enter the number corresponding to the bag file: "))
# bag_idx = 0

bag_file = os.path.join(bag_dir, bag_files[bag_idx])

b = bagreader(bag_file)   # This creates an object that we name 'b' that contains all the information from your bag

csvfiles = {}     # To avoid mixing up topics, we save each topic as an individual csv file, since some topics might have the same headers!
for topic in b.topics:
    data = b.message_by_topic(topic)
    csvfiles[topic] = data

##################### READ TOPICS

state = pd.read_csv(csvfiles['crazy00/odom'])   # The topic "odom" contains all the state information we need

vicon_time = state['Time'] - b.start_time   # Here we are extracting time and subtracting the start time of the .bag file

# Position
x = state['pose.pose.position.x']
y = state['pose.pose.position.y']
z = state['pose.pose.position.z']

# Velocity
xdot = state['twist.twist.linear.x']
ydot = state['twist.twist.linear.y']
zdot = state['twist.twist.linear.z']

# Angular Velocity (w.r.t. body frames x, y, and z)
wx = state['twist.twist.angular.x']
wy = state['twist.twist.angular.y']
wz = state['twist.twist.angular.z']

# Orientation (measured as a unit quaternion)
qx = state['pose.pose.orientation.x']
qy = state['pose.pose.orientation.y']
qz = state['pose.pose.orientation.z']
qw = state['pose.pose.orientation.w']

# If you want to use Rotation, these lines might be useful
q = np.vstack((qx,qy,qz,qw)).T      # Stack the quaternions, shape -> (N,4)
rot = Rotation.from_quat(q[0,:])    # This should be familiar from the simulator

control = pd.read_csv(csvfiles['crazy00/so3cmd_to_crazyflie/cmd_vel_fast'])   # The topic "so3cmd_to_crazyflie/cmd_vel_fast" has our control inputs

# Different topics publish at different rates and times... we need to make sure the times is synced up between topics
control_time = control['Time'] - b.start_time

# Coefficients below are used to convert thrust PWM (sent to Crazyflie) into Newtons (what your controller computes)
c1 = -0.6709 
c2 = 0.1932
c3 = 13.0652
cmd_thrust = (((control['linear.z']/60000 - c1) / c2)**2 - c3)/1000*9.81

# Orientation is sent to the Crazyflie as Euler angles (pitch and roll, specifically)
roll = control['linear.x']
pitch = control['linear.y']
yaw = np.zeros(pitch.shape)  # Here we assume 0 yaw.
cmd_q = Rotation.from_euler('zyx', np.transpose([yaw,roll,pitch]), degrees=True).as_quat()  # Generate quaternions from E

#################### PLOTTING

# It's often useful to save the objects associated with a figure and its axes
(fig_pos, axes_pos) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Position vs Time', figsize=(10, 6))

ax = axes_pos[0]    # Select the first plot

# You can plot using multiple lines if you want it to be readable
ax.plot(vicon_time, x, 'r.', markersize=2)   
ax.plot(vicon_time, y, 'g.', markersize=2)
ax.plot(vicon_time, z, 'b.', markersize=2)
ax.legend(('x', 'y', 'z'), loc='upper right')   # Set a legend
ax.set_ylabel('position, m')                    # Set a y label
ax.grid('major')                                # Put on a grid
ax.set_title('Position')                        # Plot title

ax = axes_pos[1]    # Select the second plot

# Or to be more efficient you can plot everything with one line...
ax.plot(vicon_time, xdot, 'r.', vicon_time, ydot, 'g.', vicon_time, zdot, 'b.', markersize=2)
ax.legend(('x','y','z'), loc='upper right')
ax.set_ylabel('velocity, m/s')
ax.grid('major')
ax.set_title('Velocity')
ax.set_xlabel("time, s")

# Orientation and Angular Velocity vs. Time
(fig_rot, axes_rot) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Orientation vs Time', figsize=(10, 6))

ax = axes_rot[0]
ax.plot(control_time, cmd_q[:,0], 'r', control_time, cmd_q[:,1], 'g',
        control_time, cmd_q[:,2], 'b', control_time, cmd_q[:,3], 'k')
ax.plot(vicon_time, q[:,0], 'r.',  vicon_time, q[:,1], 'g.',
        vicon_time, q[:,2], 'b.',  vicon_time, q[:,3],'k.', markersize=2)
ax.legend(('i', 'j', 'k', 'w'), loc='upper right')
ax.set_ylabel('quaternion')
ax.grid('major')
ax.set_title('Orientation')

ax = axes_rot[1]
ax.plot(vicon_time, wx, 'r.', vicon_time, wy, 'g.', vicon_time, wz, 'b.',markersize=2)
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('angular velocity, rad/s')
ax.set_xlabel('time, s')
ax.grid('major')
ax.set_title('Body Rates')

# Commands vs. Time
(fig_cmd, axes_cmd) = plt.subplots(nrows=1, ncols=1, sharex=True, num='Commands vs Time', figsize=(10, 6))
ax = axes_cmd
ax.plot(control_time, cmd_thrust, 'k.-', markersize=5)
ax.set_ylabel('thrust, N')
ax.set_xlabel('time, s')
ax.grid('major')
ax.set_title('Commanded Thrust')

# 3D position.
fig_3d = plt.figure(figsize=(10, 10))
ax = fig_3d.add_subplot(projection='3d')

ax.view_init(elev=30, azim=-45)
ax.set_xlabel('x position, m')
ax.set_ylabel('y position, m')
ax.set_zlabel('z position, m')
ax.plot3D(x,y,z,'k.-',markersize=5)
ax.scatter3D(x[0],y[0],z[0], marker='o', c='r', s=60)
ax.scatter3D(x.iloc[-1],y.iloc[-1],z.iloc[-1], marker='o', c='g', s=60)
ax.legend(('Trajectory','Start','End'))
ax.set_title("Crazyflie Trajectory")

############# SAVE FIGS

fig_pos.savefig(os.path.join(fig_dir, 'pos_vel.png'))
fig_rot.savefig(os.path.join(fig_dir, 'rotations.png'))
fig_cmd.savefig(os.path.join(fig_dir, 'cmd_thrust.png'))
fig_3d.savefig(os.path.join(fig_dir, "position3d.png"))
plt.show()