#!/usr/bin/python3

# ROS imports
import rospy
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from rotorpy_hardware.msg import WindEstimate
from crazyflie_driver.msg import GenericLogData
from kr_mav_msgs.msg import SO3Command

# General Python imports
from scipy.spatial.transform import Rotation
import numpy as np
import sys
import os
from copy import deepcopy

# Relative imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.wind_estimators.wind_ukf import WindUKFV2

# RotorPy imports 
from rotorpy.vehicles.crazyfliebrushless_params import quad_params

def pwm_to_motorspeeds(pwm, pwm_to_rpm_coeffs=[0.51212, -1719], output='rad/s'):
    # Converts motor pwm to motor speeds (rpm) using predefined mapping. 

    if isinstance(pwm, list):
        pwm = np.array(pwm)

    rpm = np.polyval(pwm_to_rpm_coeffs, pwm)

    if output == 'rpm':
        return rpm
    elif output == 'rad/s':
        return rpm*0.10472
    else:
        raise ValueError("Motor speed mapping: must specify either 'rpm' or 'rad/s'. ")

def pwm_to_motorspeeds_batcompensated(pwm, vbat, coeffs=[4.3034, 0.759, 10_000], output='rad/s'):
    # Converts motor pwm signal to motor speeds using a predefined mapping. Improved with battery compensation. 
    # coeffs:
    # --- 0: kv rating 
    # --- 1: voltage offset. 
    # --- 2: ESC deadband.
    # Coefficients were identified using thrust stand testing at various different battery supply voltages. git status

    pwm = np.clip(pwm, coeffs[2], np.inf)
    rpm = coeffs[0]*(vbat + coeffs[1])*(pwm - coeffs[2])**(2/3)

    if output == 'rpm':
        return rpm
    elif output == 'rad/s':
        return rpm*0.10472
    else:
        raise ValueError("Motor speed mapping: must specify either 'rpm' or 'rad/s'. ") 

class WindEstimatorNode:
    """
    The wind estimator node takes measurements from the IMU and motion capture and produces estimates of the wind vector. 
    """
    def __init__(self):

        self.filter_initialized = False

        self.state = {}
        self.imu_measurement = {}
        self.motorspeeds = None
        self.cmd_thrust = None
        self.vbat = None

        self.mav_name = rospy.get_namespace()
        self.mass = rospy.get_param(self.mav_name+"/mass")

        self.quad_params = deepcopy(quad_params)
        self.quad_params['mass'] = self.mass 

        # UKF tuning params.
        self.Q = np.diag(np.concatenate([1.0*np.ones(4), 0.5*np.ones(3), 0.5*np.ones(3), 0.5*np.ones(3), 0.1*np.ones(3), 1.45*np.ones(1)]))
        self.R = np.diag(np.concatenate([0.75*np.ones(4), 0.010*np.ones(3), 0.5*np.ones(3), 0.01*np.ones(3), 0.1*np.sqrt(100/2)*(0.38**2)*np.ones(3), 0.5*np.ones(1)]))
        self.P0 = np.diag(np.concatenate([2.5*np.ones(4), 0.5*np.ones(13)]))
        self.dt_filter = 1/100
        self.alpha = 0.1 
        self.beta = 2.0
        self.kappa = -1

        self.accel_bias = np.array([0, 0.18, 0])

        # Pubs and Subs
        self.filter_pub_rate = 1/self.dt_filter  # Hz, nominally 100
        self.filter_pub = rospy.Publisher('wind_estimate', WindEstimate, queue_size=1)
        self.filter_pub_timer = rospy.Timer(rospy.Duration(1.0 / self.filter_pub_rate), self.run)

        self.wind_vector_marker_pub = rospy.Publisher('wind_vector_marker', Marker, queue_size=10)
        
        rospy.Subscriber("odom", Odometry, self.mocap_callback, queue_size=1)
        rospy.Subscriber("imu", Imu, self.imu_callback, queue_size=1)
        rospy.Subscriber("so3_cmd", SO3Command, self.so3_callback, queue_size=1)
        rospy.Subscriber("motor_pwms", GenericLogData, self.motorpwm_callback, queue_size=1)
        rospy.Subscriber("ina_voltage", GenericLogData, self.vbat_callback, queue_size=1)

        return 
    
    def kalman_init(self):
        """ 
        Initialize the Kalman filter. Run only once, when the full state is filled out from various ROS topics. 
        """

        # Extract measurement from the current state of the UAV
        orientation = Rotation.from_quat(deepcopy(self.state['q']))
        euler_angles = orientation.as_euler('zyx', degrees=False)  # Get Euler angles from current orientation
        body_speed = (orientation.as_matrix()).T@self.state['v']
        
        # xhat = [m1, m2, m3, m4, psi, theta, phi, p, q, r, vx, vy, vz, windx, windy, windz]
        xhat0 = np.concatenate([self.motorspeeds, euler_angles, self.imu_measurement['gyro'], body_speed, 1e-5*np.ones((3,)), self.quad_params['k_eta']/1e-8*np.ones(1)])

        self.wind_filter = WindUKFV2(self.quad_params, xhat0=xhat0, Q=self.Q, R=self.R, P0=self.P0, dt=self.dt_filter, alpha=self.alpha, beta=self.beta, kappa=self.kappa)
        self.filter_initialized = True

        return 
    
    def kalman_predict(self):
        """
        Run the predict step of the Kalman filter.
        """

        self.wind_filter.filter.predict()

        return 
    
    def kalman_update(self):
        """
        Run the update step of the Kalman filter.
        """

        # Extract measurement from the current state of the UAV
        orientation = Rotation.from_quat(deepcopy(self.state['q']))
        euler_angles = orientation.as_euler('zyx', degrees=False)  # Get Euler angles from current orientation
        body_speed = (orientation.as_matrix()).T@self.state['v']

        zk = np.array([self.motorspeeds[0]/1e3,                 # m1     (rpm)
                       self.motorspeeds[1]/1e3,                 # m2     (rpm)
                       self.motorspeeds[2]/1e3,                 # m3     (rpm)
                       self.motorspeeds[3]/1e3,                 # m4     (rpm)
                       euler_angles[0],                     # phi    (rad)
                       euler_angles[1],                     # theta  (rad)
                       euler_angles[2],                     # psi    (rad)
                       self.imu_measurement['gyro'][0],     # p      (rad/s)
                       self.imu_measurement['gyro'][1],     # q      (rad/s)
                       self.imu_measurement['gyro'][2],     # r      (rad/s)
                       body_speed[0],                       # vx     (m/s)
                       body_speed[1],                       # vy     (m/s)
                       body_speed[2],                       # vz     (m/s)
                       self.imu_measurement['accel'][0],    # body x acceleration  (m/s/s)
                       self.imu_measurement['accel'][1],    # body y acceleration  (m/s/s)
                       self.imu_measurement['accel'][2],    # body z acceleration  (m/s/s)
                       self.cmd_thrust,                     # thrust force (N)
                       ])
        
        # Run update method.
        self.wind_filter.filter.update(zk)

        return 
    
    def mocap_callback(self, msg):
        """
        Callback for motion capture data for sensing the state of the quadrotor. 
            x, position, m
            v, linear velocity, m/s
            q, quaternion [i,j,k,w]
            w, angular velocity, rad/s
        """
        
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

        return

    def imu_callback(self, msg):
        """ 
        Callback for the inertial measurement unit. 
        """

        # Update body rates (angular velocities)
        self.imu_measurement['gyro'] = np.array([msg.angular_velocity.x,
                                                msg.angular_velocity.y,
                                                msg.angular_velocity.z])
        
        # Update linear accelerations (could be used for body speed prediction if required)
        self.imu_measurement['accel'] = np.array([msg.linear_acceleration.x,
                                                msg.linear_acceleration.y,
                                                msg.linear_acceleration.z]) + self.accel_bias

        return 
    
    def so3_callback(self, msg):
        """
        Callback for the SO3Command, which contains the desired thrust vector. 
        """

        R = Rotation.from_quat(np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])).as_matrix()
        b3 = R @ np.array([0, 0, 1])
        self.cmd_thrust = np.dot(np.array([msg.force.x, msg.force.y, msg.force.z]), b3)

        return 
    
    def motorpwm_callback(self, msg):
        """ 
        Callback for the motor speed commands which we treat as motor speed measurements
        """

        alpha = 0.01

        # self.motorspeeds = pwm_to_motorspeeds(msg.values, output='rad/s')
        if self.vbat is not None:
            cmd_motorspeeds = pwm_to_motorspeeds_batcompensated(np.array(msg.values), self.vbat, output='rad/s')
            if self.motorspeeds is None:
                self.motorspeeds = 0.0
            self.motorspeeds = (1-alpha)*self.motorspeeds + (alpha)*cmd_motorspeeds
        else:
            self.motorspeeds = None

        return

    def vbat_callback(self, msg):
        """ 
        Callback for the battery voltage measurement coming from the ina power sensor. 
        """ 

        self.vbat = msg.values[0] 

        return 
    
    def publish_filter_estimate(self):
        """ 
        Create and publish an estimate of the wind vector from the Kalman filter.
        """

        msg = WindEstimate()

        # Header
        msg.header.stamp = rospy.Time.now() 
        msg.header.frame_id = self.mav_name.split('/')[1]

        # xhat = [m1, m2, m3, m4, psi, theta, phi, p, q, r, vx, vy, vz, windx, windy, windz]

        # Motor speeds
        msg.m1 = self.wind_filter.filter.x[0]
        msg.m2 = self.wind_filter.filter.x[1]
        msg.m3 = self.wind_filter.filter.x[2]
        msg.m4 = self.wind_filter.filter.x[3]

        # Euler angles
        msg.roll = self.wind_filter.filter.x[4]
        msg.pitch = self.wind_filter.filter.x[5]
        msg.yaw = self.wind_filter.filter.x[6]

        # Body rates
        msg.roll_rate = self.wind_filter.filter.x[7]
        msg.pitch_rate = self.wind_filter.filter.x[8]
        msg.yaw_rate = self.wind_filter.filter.x[9]

        # Velocities
        msg.ground_vx = self.wind_filter.filter.x[10]
        msg.ground_vy = self.wind_filter.filter.x[11]
        msg.ground_vz = self.wind_filter.filter.x[12]

        # Wind velocity estimate
        msg.wind_vx = self.wind_filter.filter.x[13]
        msg.wind_vy = self.wind_filter.filter.x[14]
        msg.wind_vz = self.wind_filter.filter.x[15]

        # Normalized thrust coefficient
        msg.thrust_coeff_norm = self.wind_filter.filter.x[16]

        # Covariance matrix (flattened into a 1D array)
        msg.covariance = self.wind_filter.filter.P.flatten().tolist()

        self.filter_pub.publish(msg)

        return 
    
    def publish_wind_vector(self):
        """
        Publish the wind vector for visualization in RViz. 
        """

        # Marker creation
        marker = Marker()
        marker.header.frame_id = self.mav_name.split('/')[1]
        marker.header.stamp = rospy.Time.now()
        marker.ns = "wind_vector"
        marker.id = 0
        marker.type = Marker.ARROW 
        marker.action = Marker.ADD
        marker.scale.x = 0.05   # Arrow head size
        marker.scale.y = 0.1  # Arrow shaft diameter
        marker.scale.z = 0.15  # Arrow shaft diameter
        marker.color.a = 1.0   # Alpha (transparency)
        marker.color.r = 1.0   # Red component
        marker.color.g = 0.0   # Green component
        marker.color.b = 1.0   # Blue component (blue color for wind vector)

        # Set the starting point of the arrow (base of the arrow)
        marker.points = [Point(0, 0, 0)]  # Start at the origin of the robot's TF frame

        marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)

        # Set the wind vector direction and magnitude
        wind_vx = self.wind_filter.filter.x[13]
        wind_vy = self.wind_filter.filter.x[14]
        wind_vz = self.wind_filter.filter.x[15]

        # Set the endpoint of the arrow (the direction and magnitude of the wind)
        marker.points.append(Point(wind_vx, wind_vy, wind_vz))

        # Publish the marker
        self.wind_vector_marker_pub.publish(marker)

        return
    
    def run(self, event):

        if not self.filter_initialized:
            # Check if the filter can be initialized yet. 
            if self.state and self.imu_measurement and self.motorspeeds is not None and self.cmd_thrust is not None and self.vbat is not None:
                self.kalman_init()
            return 
        else:
            # Run the Kalman predict and publish. 
            self.kalman_predict()
            self.kalman_update()
            self.publish_filter_estimate()
            self.publish_wind_vector()

            return
    
    def shutdown(self):
        rospy.loginfo("Shutting down wind estimator.")
        return 
    
if __name__ == "__main__":

    rospy.init_node('wind_estimator')

    estimator_node = WindEstimatorNode()
    rospy.on_shutdown(estimator_node.shutdown)

    rospy.spin()