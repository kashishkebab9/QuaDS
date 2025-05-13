#!/usr/bin/python3
'''
Run quadrotor controllers for trajectory tracking. 
Adapted from code from Laura Jarin-Lipschitz written for MEAM 620 at the University of Pennsylvania.
'''

# ROS imports
import rospy
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from kr_mav_msgs.msg import TRPYCommand
from kr_tracker_msgs.srv import Transition
from kr_tracker_msgs.msg import TrackerStatus
from std_srvs.srv import Trigger, SetBool
from rotorpy_hardware.msg import ExtendedSO3Command

# General Python imports
from scipy.spatial.transform import Rotation
import numpy as np
import os
import sys
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import struct

# RotorPy imports
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj, ThreeDCircularTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.speed_traj import ConstantSpeed
from rotorpy.trajectories.minsnap import MinSnap
from rotorpy.world import World
from lpv_ds_class import LPV_DS_Model

# Relative imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.trajectories.hover_traj import HoverTraj

class ControlCrazyflie:
    def __init__(self):

        self.mav_name = rospy.get_namespace()
        self.mass = rospy.get_param(self.mav_name+"mass")

        # Crazyflie thrust coefficients
        self.c1 = rospy.get_param(self.mav_name+"/so3cmd_to_crazyflie/c1")
        self.c2 = rospy.get_param(self.mav_name+"/so3cmd_to_crazyflie/c2")
        self.c3 = rospy.get_param(self.mav_name+"/so3cmd_to_crazyflie/c3")
        self.k_v = 10
        self.g = 9.81
        # Yaw P controller.
        self.kyaw = 10
        self.control = {}

        # Update quad params. 
        quad_params['mass'] = self.mass

        try:
            self.Dmax = rospy.get_param(self.mav_name+"/range_max")
        except:
            self.Dmax = None

        self.t_init = None
        self.t = 0
        self.control_input = {}# Thrust to PWM conversion. Converts thrust in grams to PWM signal. 

        self.flat_output = {}
        self.state = {}

        self.controller = SE3Control(quad_params)

        self.trajectory = None
        self.student_in_control = True
        self.failed = False
        self.traj_initialization()

        self.lpvds = LPV_DS_Model("/home/dcist/ws_kr/src/rotorpy_hardware_dev/src/gmm_trajectory_datacircle_twice.mat")
        
        try:
            if hasattr(self.controller, "obstacles"):
                self.obstacle_avoidance = True
            else:
                self.obstacle_avoidance = False
        except Exception as e:
            self.obstacle_avoidance = False  # In case of any unexpected error
        
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.eso3_cmd_pub = rospy.Publisher('extended_so3_cmd', ExtendedSO3Command, queue_size=1)
        rospy.Subscriber("odom", Odometry, self.mocap_callback, queue_size=1)
        rospy.Subscriber("trackers_manager/status", TrackerStatus, self.status_callback, queue_size=1)
        rospy.Subscriber("lidar_raycast", PointCloud2, self.lidar_callback, queue_size=1)

    def traj_initialization(self):

        rospy.loginfo("Waiting for VICON")
        msg = rospy.wait_for_message("odom", Odometry)
        pos = msg.pose.pose.position
        self.mocap_callback(msg)
        rospy.logwarn("Got VICON. Creating trajectory.")
        start = np.array([pos.x, pos.y, pos.z])

        self.trajectory = HoverTraj(0.7, 3.0, 1.0)
        self.trajectory = TwoDLissajous(A=0.25, B=0.25, a=2, b=1, delta=0, height=0.5, yaw_bool=False)
        self.trajectory = CircularTraj(center=np.array([0.70, -3.27, 1.2]), radius=0.75, freq=0.3, yaw_bool=False)
        self.trajectory = ThreeDCircularTraj(center=np.array([0.0, 3.0, 0.75]), radius=np.array([1.5, 0, 0]), freq=np.array([0.1, 0.2, 0.13]))

        rospy.loginfo("Finished running waypoint trajectory initialization")

    def status_callback(self, msg):
        if msg.tracker == "kr_trackers/NullTracker":
            self.student_in_control = True
        else:
            self.student_in_control = False

    def mocap_callback(self, msg):
        '''
        Callback for motion capture data for sensing the state of the quadrotor. 
            x, position, m
            v, linear velocity, m/s
            q, quaternion [i,j,k,w]
            w, angular velocity, rad/s
        '''
        
        # Get the position, linear velocity, quaternion, and angular velocity
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        lin_vel = msg.twist.twist.linear
        ang_vel = msg.twist.twist.angular

        # Update the state
        self.state['x'] = np.array([pos.x, pos.y, pos.z])
        self.state['v'] = np.array([lin_vel.x, lin_vel.y, lin_vel.z])
        self.state['q'] = np.array([q.x, q.y, q.z, q.w])
        self.state['w'] = np.array([ang_vel.x, ang_vel.y, ang_vel.z])

    def lidar_callback(self, msg):

        if self.obstacle_avoidance:
            if self.controller.obstacles is None:
                self.controller.obstacles = np.zeros((msg.height*msg.width, 3))

            for i in range(msg.height * msg.width):  # TODO: Ignore nans or infinities? 
                # Unpack each point
                point_data = msg.data[(i * msg.point_step):((i + 1) * msg.point_step)]
                point = np.array(struct.unpack('fff', point_data[:msg.point_step]))  # assuming x, y, z are the first three floats

                self.controller.obstacles[i, :] = point

        return 

    def update_control(self):

        #         self.state['x'] = np.array([pos.x, pos.y, pos.z])
        x = self.state['x'][:2].reshape(2, 1)
        x_dot = self.lpvds.evaluate_lpv_ds(x)
        x_dot = np.array([float(x_dot[0]), float(x_dot[1]), 0])
        #x_dot = np.clip(x_dot, -1.0, 1.0)

        self.flat_output['x'] =  self.state['x'] 
        self.flat_output['x_dot'] = np.array([x_dot[0], x_dot[1], 0])
        self.flat_output['x_ddot'] = np.zeros((3,))
        self.flat_output['x_dddot'] = np.zeros((3,))
        self.flat_output['x_ddddot'] = np.zeros((3,))
        self.flat_output['yaw'] = 0.0
        self.flat_output['yaw_dot'] = 0.0
        self.flat_output['yaw_ddot'] = 0.0
        #self.control = self.controller.update(self.t, self.state, self.flat_output)
        v_err = self.state['v'] - self.flat_output['x_dot']

        # Get desired acceleration based on P control of velocity error. 
        kp = 8
        a_cmd = -self.k_v*v_err - kp *np.array([0, 0, self.state['x'][2] -1])

        # Get desired force from this acceleration. 
        F_des = self.mass*(a_cmd + np.array([0, 0, self.g]))

        R = Rotation.from_quat(self.state['q']).as_matrix()
        b3 = R @ np.array([0, 0, 1])
        cmd_thrust = np.dot(F_des, b3)
        self.control['cmd_thrust'] = cmd_thrust
    
        # Follow rest of SE3 controller to compute cmd moment. 

        # Desired orientation to obtain force vector.
        b3_des = F_des/np.linalg.norm(F_des)
        c1_des = np.array([1, 0, 0])
        b2_des = np.cross(b3_des, c1_des)/np.linalg.norm(np.cross(b3_des, c1_des))
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.stack([b1_des, b2_des, b3_des]).T

        # Orientation error.
        S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
        att_err = np.array([-S_err[1,2], S_err[0,2], -S_err[0,1]])
        cmd_q = Rotation.from_matrix(R_des).as_quat()
        self.control['cmd_q'] = cmd_q

    def run_control(self): 
        '''
        Run the control loop. 

        geometry_msgs/Twist is the type for cmd_vel, the crazyflie control
        topic. The format of this is as follows:
        linear.y = roll     [-30 to 30 degrees]         (may be negative)
        linear.x = pitch    [-30 to 30 degrees]         (may be negative)
        linear.z = thrust   [0 to 60,000]               (motors stiction around 2000)
        angular.z = yawrate [-200 to 200 degrees/second] (note this is not yaw!)
        '''

        if self.t_init is None:
            self.t_init = rospy.Time.now()
        else:
            self.t = (rospy.Time.now() - self.t_init).to_sec()
        self.update_control()

        # Desired TRPY
        thrust_des_newtons = self.control['cmd_thrust']
        thrust_des_grams = thrust_des_newtons/9.81*1000  # Have to convert thrust to grams
        q_des = self.control['cmd_q']
        R_des = Rotation.from_quat(q_des).as_matrix()
        eul_des = Rotation.from_quat(q_des).as_euler('ZXY', degrees=True)
        yaw_des = eul_des[0]

        # Current RPY
        R_cur = Rotation.from_quat(self.state['q'])
        eul_cur = R_cur.as_euler('ZXY', degrees=True)
        yaw_cur = eul_cur[0]

        # Map the desired onto the current body frame based on yaw
        R_z = Rotation.from_rotvec((yaw_cur - yaw_des)*(np.pi/180)*np.array([0,0,1])).as_matrix()
        R_des_new = R_des@R_z

        pitch_des = -np.arcsin(R_des_new[2,0])*180/np.pi
        roll_des = np.arctan2(R_des_new[2,1], R_des_new[2,2])*180/np.pi

        if self.c3 + thrust_des_grams < 0:
            rospy.logerr("Thrust too negative")
            thrust_des_grams = 0
        thrust_pwm = self.c1 + self.c2 * (self.c3 + thrust_des_grams)**.5
        thrust_pwm = min(thrust_pwm, 0.9)

        # Scale to full range
        thrust_pwm_max = 60000

        e_yaw = (yaw_des - yaw_cur)
        if e_yaw > 180:
            e_yaw -= 360
        elif e_yaw < -180:
            e_yaw += 360

        # Set command msg
        cmd = Twist()
        cmd.linear.x = pitch_des
        cmd.linear.y = roll_des
        cmd.linear.z = thrust_pwm * thrust_pwm_max
        cmd.angular.z = (-self.kyaw * e_yaw)  # desired yaw rate

        # Extended SO3 command message.
        so3_cmd = ExtendedSO3Command()
        so3_cmd.header.stamp = rospy.Time.now()
        
        so3_cmd.force.x = 0
        so3_cmd.force.y = 0
        so3_cmd.force.z = thrust_des_newtons
        so3_cmd.orientation.x, so3_cmd.orientation.y, so3_cmd.orientation.z, so3_cmd.orientation.w = q_des
        so3_cmd.angular_velocity.x, so3_cmd.angular_velocity.y, so3_cmd.angular_velocity.z = np.zeros((3,))
        so3_cmd.position.x, so3_cmd.position.y, so3_cmd.position.z = self.flat_output['x']
        so3_cmd.velocity.x, so3_cmd.velocity.y, so3_cmd.velocity.z = self.flat_output['x_dot']
        so3_cmd.acceleration.x, so3_cmd.acceleration.y, so3_cmd.acceleration.z = self.flat_output['x_ddot']
        so3_cmd.jerk.x, so3_cmd.jerk.y, so3_cmd.jerk.z = self.flat_output['x_dddot']
        so3_cmd.snap.x, so3_cmd.snap.y, so3_cmd.snap.z = self.flat_output['x_ddddot']
        so3_cmd.yaw = self.flat_output['yaw']
        so3_cmd.yaw_dot = self.flat_output['yaw_dot']
        # Publish
        self.cmd_pub.publish(cmd)
        self.eso3_cmd_pub.publish(so3_cmd)

    def transition_to_null_tracker(self):
        '''
        Tell quadrotor_control to relinquish control to the student
        '''
        service_name = self.mav_name + '/trackers_manager/transition'
        try:
            service_proxy = rospy.ServiceProxy(service_name, Transition)
            resp = service_proxy('kr_trackers/NullTracker')
            rospy.loginfo(resp.success)
            rospy.loginfo(resp.message)
        except rospy.ServiceException as e:
            print("Service call failed: %s", e)

    def motors_off(self):

        try:
            srv = rospy.ServiceProxy('mav_services/motors', SetBool)
            resp = srv(False)
            rospy.loginfo(resp)
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e)
            return False

if __name__ == "__main__":

    rospy.init_node('control_crazyflie')

    cc = ControlCrazyflie()
    print("created cc obj")
    rospy.on_shutdown(cc.motors_off)

    while not rospy.is_shutdown() and not cc.failed and cc.trajectory is None:
        rospy.sleep(1)

    if cc.trajectory: 
        print("in here now!")
        cc.transition_to_null_tracker()
        print("transitioned")
        r = rospy.Rate(100) 
        while not rospy.is_shutdown() and not cc.student_in_control:
            r.sleep()
        while not rospy.is_shutdown() and cc.student_in_control:
            cc.run_control()
            r.sleep()
        rospy.spin()
