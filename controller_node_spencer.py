#!/usr/bin/python3
'''
Run quadrotor controllers for trajectory tracking. 
Adapted from code from Laura Jarin-Lipschitz written for MEAM 620 at the University of Pennsylvania.
'''

# ROS imports
import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from kr_tracker_msgs.srv import Transition
from kr_tracker_msgs.msg import TrackerStatus
from kr_mav_msgs.msg import SO3Command
from std_srvs.srv import Trigger, SetBool
from rotorpy_hardware.msg import ExtendedSO3Command
from rotorpy_hardware.msg import WindShaper

# General Python imports
from scipy.spatial.transform import Rotation
import numpy as np
import sys
import os
windshape_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'windshape_api' ))
sys.path.append(windshape_path)
from copy import deepcopy
import struct

# RotorPy imports
# from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj, ThreeDCircularTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.speed_traj import ConstantSpeed
# from rotorpy.trajectories.minsnap import MinSnap
from rotorpy.world import World

# Relative imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.trajectories.hover_traj import HoverTraj
from scripts.trajectories.circular_rampup_traj import ThreeDRampedCircularTraj, ThreeDSteppedCircularTraj
from scripts.trajectories.speed_traj import SpeedSweepTraj
from scripts.trajectories.minsnap import MinSnap
from scripts.controllers.se3_control import SE3Control

from scripts.wind_control.windcontroller import WindShapeController
from scripts.wind_control.speed_sweep import SpeedSweep
from scripts.wind_control.speed_sinusoid import SpeedSinusoid

from scripts.trajectories.yaw_trajs import *

class ControlCrazyflie:
    def __init__(self):

        self.mav_name = rospy.get_namespace()
        self.mass = rospy.get_param(self.mav_name+"/mass")

        # Crazyflie thrust coefficients
        self.c1 = rospy.get_param(self.mav_name+"/so3cmd_to_crazyflie/c1")
        self.c2 = rospy.get_param(self.mav_name+"/so3cmd_to_crazyflie/c2")
        self.c3 = rospy.get_param(self.mav_name+"/so3cmd_to_crazyflie/c3")

        # Load windshape controls. 
        self.windshaper_controller = None
        self.only_windshape = False


        ################### Wind Probe ONLY Experiments
        # levels=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 0]
        # power_traj = SpeedSweep(levels=levels, duration=15)
        # power_traj = SpeedSinusoid(min_power=10, max_power=60, period=20, num_periods=5)

        ################### Crazyflie + WindShape experiments

        # ### Experiment 1
        # levels=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 0]
        # power_traj = SpeedSweep(levels=levels, duration=15)

        # ### Experiment 2
        # power_traj = SpeedSinusoid(min_power=10, max_power=60, period=20, num_periods=5)

        # ### Experiment 3
        # traj = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=6)
        # levels = [10, 35, 60]
        # duration = []
        # for i in range(traj.t_keyframes.size - 1):
        #     one_pass_time = 2*(1/traj.all_freqs[i])  # two periods. 
        #     duration.extend([one_pass_time]*len(levels))
        # levels = levels*traj.num_freq
        # levels.append(0)
        # duration.append(10)
        # print(levels)
        # power_traj = SpeedSweep(levels=levels, duration=duration)

        # ### Experiment 4
        # traj = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=6)
        # period = 20
        # num_periods = ((traj.t_keyframes[-1] // period) + 1)*period
        # power_traj = SpeedSinusoid(min_power=10, max_power=60, period=period, num_periods=num_periods)

        # ### Experiment 5
        # traj = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=6)
        # levels = [10, 35, 60]
        # duration = []
        # for i in range(traj.t_keyframes.size - 1):
        #     one_pass_time = 2*(1/traj.all_freqs[i])  # two periods. 
        #     duration.extend([one_pass_time]*len(levels))
        # levels = levels*traj.num_freq
        # levels.append(0)
        # duration.append(10)
        # print(levels)
        # power_traj = SpeedSweep(levels=levels, duration=duration)

        # ### Experiment 6
        # power_traj = SpeedSinusoid(min_power=10, max_power=60, period=20, num_periods=5)


        ################### Crazyflie BRUSHLESS + WindShape experiments

        # ### Experiment 1
        # levels=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 0]
        # power_traj = SpeedSweep(levels=levels, duration=15)

        # ### Experiment 2
        # power_traj = SpeedSinusoid(min_power=10, max_power=60, period=30, num_periods=5)

        # ### Experiment 3
        # traj = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=6)
        # levels = [10, 35, 60]
        # duration = []
        # for i in range(traj.t_keyframes.size - 1):
        #     one_pass_time = 2*(1/traj.all_freqs[i])  # two periods. 
        #     duration.extend([one_pass_time]*len(levels))
        # levels = levels*traj.num_freq
        # levels.append(0)
        # duration.append(10)
        # print(levels)
        # power_traj = SpeedSweep(levels=levels, duration=duration)

        # ### Experiment 4
        # traj = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=6)
        # period = 20
        # num_periods = ((traj.t_keyframes[-1] // period) + 1)*period
        # power_traj = SpeedSinusoid(min_power=10, max_power=60, period=period, num_periods=num_periods)

        # ### Experiment 5
        # traj = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=6)
        # levels = [10, 35, 60]
        # duration = []
        # for i in range(traj.t_keyframes.size - 1):
        #     one_pass_time = 2*(1/traj.all_freqs[i])  # two periods. 
        #     duration.extend([one_pass_time]*len(levels))
        # levels = levels*traj.num_freq
        # levels.append(0)
        # duration.append(10)
        # print(levels)
        # power_traj = SpeedSweep(levels=levels, duration=duration)

        # ### Experiment 6
        # power_traj = SpeedSinusoid(min_power=10, max_power=60, period=20, num_periods=5)

        # ### Experiment 7 
        # levels=[0, 5, 10, 15, 20, 25, 30, 0]
        # power_traj = SpeedSweep(levels=levels, duration=20)

        # ### Experiment 8
        # levels=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 0]
        # power_traj = SpeedSweep(levels=levels, duration=20)

        # ### Experiment 9 
        # levels=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 0]
        # power_traj = SpeedSweep(levels=levels, duration=20)

        # # ### Experiment 10
        # traj = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0.0, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=4, yaw_traj=Spinning(yaw_rate=np.pi/4, dir=1, yaw_init=0))
        # levels = [10, 35, 60]
        # duration = []
        # for i in range(traj.t_keyframes.size - 1):
        #     one_pass_time = 2*(1/traj.all_freqs[i])  # two periods. 
        #     duration.extend([one_pass_time]*len(levels))
        # levels = levels*traj.num_freq
        # levels.append(0)
        # duration.append(10)
        # print(levels)
        # power_traj = SpeedSweep(levels=levels, duration=duration)

        # ### Experiment 11
        # levels=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 0]
        # power_traj = SpeedSweep(levels=levels, duration=20)

        # ### Experiment 12
        # levels=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 0]
        # power_traj = SpeedSweep(levels=levels, duration=20)

        # ### Experiment 13
        # levels=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 0]
        # power_traj = SpeedSweep(levels=levels, duration=20)


        # ### Experiment 14 DEMO
        # levels = [25]
        # power_traj = SpeedSweep(levels=levels, duration=100)

        # ### Experiment 15
        # levels = [40, 0]
        # power_traj= SpeedSweep(levels=levels, duration=100)

        # self.windshaper_controller = WindShapeController(power_traj, update_freq=10)

        # Yaw P controller.
        self.kyaw = 8

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
        self.controller.kp_pos = np.array([10, 10, 20])
        self.controller.kd_pos = np.array([8, 8, 11])
        self.controller.ki_pos = np.array([0, 0, 0])

        self.trajectory = None
        self.student_in_control = True
        self.failed = False
        self.traj_initialization()
        
        try:
            if hasattr(self.controller, "obstacles"):
                self.obstacle_avoidance = True
            else:
                self.obstacle_avoidance = False
        except Exception as e:
            self.obstacle_avoidance = False  # In case of any unexpected error

        self.cmd_pub = rospy.Publisher('so3_cmd', SO3Command, queue_size=10)
        self.so3_cmd_extended_pub = rospy.Publisher('extended_so3_cmd_extended', ExtendedSO3Command, queue_size=1)
        if self.windshaper_controller is not None:
            self.windshaper_pub = rospy.Publisher('/windshaper', WindShaper, queue_size=1)

        if not self.only_windshape:
            rospy.Subscriber("odom", Odometry, self.mocap_callback, queue_size=1)
            rospy.Subscriber("trackers_manager/status", TrackerStatus, self.status_callback, queue_size=10)
            rospy.Subscriber("lidar_raycast", PointCloud2, self.lidar_callback, queue_size=1)

        if self.obstacle_avoidance:
            self.marker_rate = 10  # Hz
            self.mppi_traj_markers = rospy.Publisher('mppi_traj_markers', MarkerArray, queue_size=1)
            self.marker_timer = rospy.Timer(rospy.Duration(1.0 / self.marker_rate), self.visualize_mppi_trajectories)

            self.traj_des_markers = rospy.Publisher('traj_des_markers', MarkerArray, queue_size=1)
            self.traj_des_timer = rospy.Timer(rospy.Duration(1.0 / self.marker_rate), self.visualize_traj_des)

    def traj_initialization(self):

        if not self.only_windshape:
            rospy.logwarn("Waiting for VICON")
            msg = rospy.wait_for_message("odom", Odometry)
            pos = msg.pose.pose.position
            self.mocap_callback(msg)
            rospy.logwarn("Got VICON. Creating trajectory.")
            start = np.array([pos.x, pos.y, pos.z])

        # self.trajectory = ThreeDSteppedCircularTraj(center=np.array([5.5, 0.0, 1.0]), radius=np.array([0, 1.0, 0]), startfreq=0.1, endfreq=0.3, num_freq=5, num_periods_per_freq=3)
        # self.trajectory = ThreeDRampedCircularTraj(center=np.array([4.67, 0.0, 1.15]), radius=np.array([0, 0.0, 0]), startfreq=np.array([0.0, 0.05, 0.0]), endfreq=np.array([0.0,0.4,0.0]), freq_vel=0.01)

        ################### Crazyflie + WindShape experiments

        ### Experiment 1 
        # self.trajectory = HoverTraj(4.21, 0.49, 1.15)

        # # ### Experiment 2
        # self.trajectory = HoverTraj(4.21, 0.49, 1.15)

        # # ### Experiment 3
        # self.trajectory = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=6)

        # # ### Experiment 4
        # self.trajectory = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=6)

        # # ### Experiment 5
        # self.trajectory = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0.85, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=6)

        # # ### Experiment 6
        # self.trajectory = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0.85, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=6)


        ################### Crazyflie BRUSHLESS + WindShape experiments

        ### Experiment 1 
        # self.trajectory = HoverTraj(4.21, 0.49, 1.15)

        # # ### Experiment 2
        # self.trajectory = HoverTraj(4.21, 0.49, 1.15)

        # # ### Experiment 3
        # self.trajectory = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=6)

        # # ### Experiment 4
        # self.trajectory = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=6)

        # # ### Experiment 5
        # self.trajectory = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0.85, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=6)

        # # ### Experiment 6
        # self.trajectory = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0.85, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=6)

        # # ### Experiment 7 
        # yaw_angles = np.array([0, 45, 90])*np.pi/180
        # duration = 20/3
        # repeat = True
        # self.trajectory = HoverTraj(4.21, 0.49, 1.15, yaw_traj=SteppedYaw(yaw_angles, duration, lowpass_tau=1, sampling_rate=100, repeat=True))

        # # ### Experiment 8
        # self.trajectory = HoverTraj(4.21, 0.49, 1.15, yaw_traj=SteppedYaw(yaw_angles=[0, np.pi/4], duration=10, lowpass_tau=1, sampling_rate=100, repeat=True))

        # # ### Experiment 9
        # self.trajectory = HoverTraj(4.21, 0.49, 1.15, yaw_traj=Spinning(yaw_rate=np.pi/4, dir=1, yaw_init=0))

        # # ### Experiment 10
        # self.trajectory = ThreeDSteppedCircularTraj(center=np.array([4.21, 0.49, 1.15]), radius=np.array([0.0, 0.85, 0]), startfreq=0.05, endfreq=0.2, num_freq=3, num_periods_per_freq=4, yaw_traj=Spinning(yaw_rate=np.pi/4, dir=1, yaw_init=0))

        # # ### Experiment 11
        # self.trajectory = HoverTraj(4.21, 0.49, 1.15, yaw_traj=Spinning(yaw_rate=3*np.pi/4, dir=1, yaw_init=0))

        # # ### Experiment 12
        # self.trajectory = HoverTraj(4.21, 0.49, 1.15)

        # # ### Experiment 13
        # self.trajectory = HoverTraj(4.21, 0.49, 1.15, yaw_traj=SteppedYaw(yaw_angles=[0, np.pi/4], duration=10, lowpass_tau=1, sampling_rate=100, repeat=True))

        # # ### Experiment 15
        # self.trajectory = ThreeDSteppedCircularTraj(center=np.array([5.04, 0.51, 0.78]), radius=np.array([1.25, 1.25, 0]), startfreq=0.05, endfreq=0.15, num_freq=3, num_periods_per_freq=2)


        ####### DEMO1  - 02/04/2025

        # waypoints = np.array([[10.34, -2.89, 1.81], 
        #                       [9.0, -1.17, 0.69], 
        #                       [7.83, 0.95, 0.44], 
        #                       [6.55, 3.45, 0.48],
        #                       [5.68, 2.50, 1.53],
        #                       [6.27, 0.93, 1.84],
        #                       [6.55, -0.24, 2.23],
        #                       [5.94, 0.03, 3.09], 
        #                       [5.1, -0.2, 2.32],
        #                       [4.2, -1.40, 1.28],
        #                       [5.88, -1.94, 1.18],
        #                       [10.34, -2.89, 1.81]])
        # self.trajectory = MinSnap(waypoints, v_max=3, v_avg=1.35, v_start=[0, 0, 0], v_end=[0, 0, 0], loop=True)

        ######  DEMO2  - 03/12/2025
        # waypoints = np.array([[5.82, 0.58, 0.55], 
        #                       [6.45, 0.58, 1.31],
        #                       [6.07, 1.93, 1.30],
        #                       [4.13, 1.81, 1.21],
        #                       [4.14, 0.32, 0.96],
        #                       [5.52, -0.44, 0.93],
        #                       [4.56, -1.66, 1.13],
        #                       [4.35, -0.29, 1.70],
        #                       [5.08, 0.49, 2.30],
        #                       [5.80, 0.56, 1.12],
        #                       [5.82, 0.58, 0.55]])
        # self.trajectory = MinSnap(waypoints, v_max=3, v_avg=0.5, v_start=[0, 0, 0], v_end=[0, 0, 0], loop=True)

        ######  DEMO3 - 03/13/2025
        # waypoints = np.array([[5.73, 0.51, 0.64],  # Heliport leeward side. 
        #                       [5.06, 1.41, 0.80],  # 1st waypoint after heliport
        #                       [4.14, 0.64, 0.83], 
        #                       [4.75, 0.02, 0.81],  # Waypoint between buildings 1st pass. 
        #                       [5.33, -0.84, 0.80], 
        #                       [4.19, -1.22, 1.01], 
        #                       [4.30, -0.35, 1.40], # Second heliport waypoint
        #                       [4.93, 0.52, 1.61],   # Top of 45deg building. 
        #                       [5.96, 1.17, 0.93],    # Approach on landing
        #                       [6.18, 0.47, 0.70],   # Second waypoint for approach on landing
        #                       [5.73, 0.51, 0.64],
        #                       ])

        # self.trajectory = MinSnap(waypoints, v_max=3, v_avg=0.3, v_start=[0, 0, 0], v_end=[0, 0, 0], loop=True)

        # self.trajectory = TwoDLissajous(A=0.25, B=0.25, a=2, b=1, delta=0, height=0.5, yaw_bool=False)
        # self.trajectory = ThreeDCircularTraj(center=np.array([6.0, 0.0, 1.0]), radius=np.array([1.0, 1.0, 0]), freq=np.array([0.25, 0.25, 0.0]))
        # self.trajectory = ThreeDRampedCircularTraj(center=np.array([4.67, 0.0, 1.15]), radius=np.array([0, 1.21, 0]), startfreq=np.array([0.0, 0.05, 0.0]), endfreq=np.array([0.0,0.2,0.0]), freq_vel=0.01)
        # self.trajectory = SpeedSweepTraj(speeds=[1.0, 2.0, 3.0, 4.0], dmax=10.33+3.4, amax=2.0, init_pos=[-3.4, 3.6, 2.75], unit_vec=[1, 0, 0], iter_per_speed=2)

        # self.trajectory = HoverTraj(x=0, y=3.0, z=0.4, yaw_traj=None)
        self.trajectory = ThreeDCircularTraj(center=np.array([0, 3, 0.5]), radius=np.array([1, 0, 0]), freq=np.array([0.2, 0.2, 0.2]))

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
        self.flat_output = self.trajectory.update(self.t)
        self.control = self.controller.update(self.t, self.state, self.flat_output)

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

        # Update windshaper
        if self.windshaper_controller is not None:
            self.windshaper_controller.update(self.t)


        if not self.only_windshape:
            self.update_control()

            # Extract relevant commands from controller
            thrust_des_newtons = self.control['cmd_thrust']
            q_des = self.control['cmd_q']
            f_des = self.control['cmd_acc']*self.mass
            w_des = np.zeros((3))  # Ignore the feedforward body rates from the controller, which seems to add instability (?)

            cmd = SO3Command()
            cmd.header.stamp = rospy.Time.now()
            cmd.aux.enable_motors = True
            cmd.aux.angle_corrections = [0, 0]
            cmd.force.x = f_des[0]
            cmd.force.y = f_des[1]
            cmd.force.z = f_des[2]
            cmd.orientation.x = q_des[0]
            cmd.orientation.y = q_des[1]
            cmd.orientation.z = q_des[2]
            cmd.orientation.w = q_des[3]
            cmd.angular_velocity.x = w_des[0]
            cmd.angular_velocity.y = w_des[1]
            cmd.angular_velocity.z = w_des[2]
            cmd.kR = [0, 0, 1]                  # Hardcoded yaw rate gains that seem to work well for the CFBL

            # Extended SO3 command message.
            so3_cmd_extended = ExtendedSO3Command()
            so3_cmd_extended.header.stamp = rospy.Time.now()
            
            so3_cmd_extended.force.x = 0
            so3_cmd_extended.force.y = 0
            so3_cmd_extended.force.z = thrust_des_newtons
            so3_cmd_extended.orientation.x, so3_cmd_extended.orientation.y, so3_cmd_extended.orientation.z, so3_cmd_extended.orientation.w = q_des
            so3_cmd_extended.angular_velocity.x, so3_cmd_extended.angular_velocity.y, so3_cmd_extended.angular_velocity.z = np.zeros((3,))
            so3_cmd_extended.position.x, so3_cmd_extended.position.y, so3_cmd_extended.position.z = self.flat_output['x']
            so3_cmd_extended.velocity.x, so3_cmd_extended.velocity.y, so3_cmd_extended.velocity.z = self.flat_output['x_dot']
            so3_cmd_extended.acceleration.x, so3_cmd_extended.acceleration.y, so3_cmd_extended.acceleration.z = self.flat_output['x_ddot']
            so3_cmd_extended.jerk.x, so3_cmd_extended.jerk.y, so3_cmd_extended.jerk.z = self.flat_output['x_dddot']
            so3_cmd_extended.snap.x, so3_cmd_extended.snap.y, so3_cmd_extended.snap.z = self.flat_output['x_ddddot']
            so3_cmd_extended.yaw = self.flat_output['yaw']
            so3_cmd_extended.yaw_dot = self.flat_output['yaw_dot']

            # Publish
            self.cmd_pub.publish(cmd)
            self.so3_cmd_extended_pub.publish(so3_cmd_extended)

        # Windshaper message
        if self.windshaper_controller is not None:
            ws_msg = WindShaper()
            ws_msg.header.stamp = rospy.Time.now()
            ws_msg.power_level = int(self.windshaper_controller.power_level)
            ws_msg.fan_power = self.windshaper_controller.fan_power
            ws_msg.fan_rpm = self.windshaper_controller.fan_rpm
            self.windshaper_pub.publish(ws_msg)

    def visualize_traj_des(self, event):
        '''
        Visualize the desired trajectory as points.
        '''

        marker_array = MarkerArray()

        # Loop through each time step and create a marker for each point.
        for j in range(self.controller.x_des.shape[0]):  # Iterate over time steps
            point = self.controller.x_des[j, :]
            
            marker = Marker()
            marker.header.frame_id = "mocap"  # Change to your desired frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "trajectories"
            marker.id = j  # Unique ID for each point
            marker.type = Marker.SPHERE  # Use sphere to represent a point
            marker.action = Marker.ADD
            
            # Set the position of the sphere
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = point[2]

            if j == 0:
                scale = 0.08
            else:
                scale = 0.04
            
            # Set scale and color for the sphere
            marker.scale.x = scale  # Diameter of the sphere
            marker.scale.y = scale
            marker.scale.z = scale
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # Alpha

            # Set orientation of marker.
            marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)

            # Add the marker to the array
            marker_array.markers.append(marker)

        # Publish the marker array
        self.traj_des_markers.publish(marker_array)

        return


    def visualize_mppi_trajectories(self, event):
        '''
        Visualize the MPPI trajectories using a MarkerArray. 
        '''

        marker_array = MarkerArray()
        for i in range(self.controller.x_traj.shape[0]):  # Iterate over each trajectory
            marker = Marker()
            marker.header.frame_id = "mocap"  # Change to your desired frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "trajectories"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            # Populate points from the trajectory
            for j in range(self.controller.x_traj.shape[1]):  # Iterate over time steps
                point = self.controller.x_traj[i, j]
                marker.points.append(self.create_point(point))

            # Set scale and color for the line
            marker.scale.x = 0.01  # Width of the line
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0  # Blue color for the trajectory
            marker.color.a = 1.0  # Alpha

            # Set orientation of marker.
            marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
            
            marker_array.markers.append(marker)

        self.mppi_traj_markers.publish(marker_array)

        return
    
    def create_point(self, position):
        """Create a Point message from a position array."""

        point = Point()
        point.x = position[0]
        point.y = position[1]
        point.z = position[2]
        return point

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
        
    def windshaper_off(self):

        try:
            self.windshaper_controller.__shutdown__()
        except:
            pass 

    def shutdown(self):

        rospy.loginfo("Shutting down motors")
        self.motors_off()
        if self.windshaper_controller is not None:
            rospy.loginfo("Shutting down Windshaper")
            self.windshaper_off()

if __name__ == "__main__":

    rospy.init_node('control_crazyflie')

    cc = ControlCrazyflie()
    rospy.on_shutdown(cc.shutdown)

    while not rospy.is_shutdown() and not cc.failed and cc.trajectory is None:
        rospy.sleep(1)
    if cc.trajectory:  # Once the trajectory is planned, start the control loop!
        cc.transition_to_null_tracker()
        r = rospy.Rate(120) 
        while not rospy.is_shutdown() and not cc.student_in_control:
            r.sleep()
        while not rospy.is_shutdown() and cc.student_in_control:
            cc.run_control()
            r.sleep()
        rospy.spin()