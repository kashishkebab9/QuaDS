#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist
from std_msgs.msg import Bool
from lpv_ds_class import LPV_DS_Model


class LPVDS_ROS_Node:
    def __init__(self, gmm_path):


        rospy.init_node('lpvds_ros_node')

        #print path
        rospy.loginfo(f"Loading GMM from: {gmm_path}")

        # Load LPV-DS model
        mat_path = rospy.get_param("~gmm", gmm_path)
        self.model = LPV_DS_Model(mat_path)

        self.latest_odom = None
        self.trigger_received = False

        self.init_pos = np.array([[self.model.Xi_ref[0, 0]], [self.model.Xi_ref[1, 0]]])
        #print("The initial position is ", self.init_pos)

        print("The initial position is ", self.init_pos)

        # Subscribers
        rospy.Subscriber('/hummingbird/ground_truth/odometry', Odometry, self.odom_callback)
        rospy.Subscriber('/trigger_cmd', Bool, self.trigger_callback)

        # Publisher
        self.cmd_pub = rospy.Publisher('/hummingbird/command/trajectory', MultiDOFJointTrajectory, queue_size=10)

        self.rate = rospy.Rate(10)
        rospy.loginfo("LPV-DS ROS node initialized.")
        self.run_loop()

    def odom_callback(self, msg):
        #print("Odometry received")  
        self.latest_odom = msg

    def trigger_callback(self, msg):
        if msg.data:
            self.trigger_received = True
            rospy.loginfo("Trigger received!")

    def publish_command(self):
        if self.latest_odom is None:
            rospy.logwarn("No odometry received yet.")
            return

        pos = self.latest_odom.pose.pose.position
        x = np.array([[pos.x], [pos.y]])

        relative_x = x + self.init_pos
        x_dot = self.model.evaluate_lpv_ds(relative_x)

        print("evaluate the x and the x_dot is ", relative_x, x_dot)

        #clip the x_dot to be in the range of [-1, 1]
        x_dot = np.clip(x_dot, -1, 1)

        # Build trajectory message
        traj_msg = MultiDOFJointTrajectory()
        traj_msg.header.stamp = rospy.Time.now()
        traj_msg.joint_names = ["base_link"]

        dt = 0.1

        point = MultiDOFJointTrajectoryPoint()
        


        twist = Twist()
        twist.linear.x = float(x_dot[0])
        twist.linear.y = float(x_dot[1])
        twist.linear.z = 0.0
        point.velocities.append(twist)

        point.accelerations.append(Twist())


        pos = Transform()
        pos.translation.x = float(x[0]) + dt * float(x_dot[0])
        pos.translation.y = float(x[1]) + dt * float(x_dot[1])
        pos.translation.z =1.0

        point.transforms.append(pos)


        point.time_from_start = rospy.Duration(0.1)
        traj_msg.points.append(point)

        self.cmd_pub.publish(traj_msg)
        rospy.loginfo(f"Published velocity: {x_dot.flatten()}")

    def run_loop(self):
        while not rospy.is_shutdown():
            #print("Waiting for trigger...")
            if self.trigger_received:
                self.publish_command()
            self.rate.sleep()
        
        
        rospy.loginfo("Shutting down LPV-DS ROS node.")


if __name__ == '__main__':

    gmm_path = "gmm_mat_data/gmm_trajectory_dataline_backward.mat"
    if not gmm_path.endswith('.mat'):
        raise ValueError("The provided file is not a .mat file.")

    try:
        LPVDS_ROS_Node(gmm_path)
    except rospy.ROSInterruptException:
        pass
