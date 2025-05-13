#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <cmath>

class OdometryPublisher
{
public:
    OdometryPublisher()
    {
        // ROS node handle
        ros::NodeHandle nh;

        // Subscriber for /robot/pose (PoseStamped)
        pose_sub_ = nh.subscribe("pose", 10, &OdometryPublisher::poseCallback, this);

        // Publisher for /odom (Odometry)
        odom_pub_ = nh.advertise<nav_msgs::Odometry>("odom", 10);

        // Initialize previous pose and timestamp
        prev_pose_ = geometry_msgs::Pose();
        prev_time_ = ros::Time::now();
    }

private:
    // Callback function for PoseStamped messages
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        // Get current time and pose
        ros::Time current_time = msg->header.stamp;
        geometry_msgs::Pose current_pose = msg->pose;

        // Create Odometry message
        nav_msgs::Odometry odom_msg;
        odom_msg.header = msg->header;
        odom_msg.pose.pose = current_pose;

        // If we have a previous pose, compute the twist (velocity)
        if (prev_time_ != ros::Time(0))
        {
            // Compute the time difference
            double delta_time = (current_time - prev_time_).toSec();

            if (delta_time > 0)
            {
                // Compute position differences (dx, dy, dz)
                double dx = current_pose.position.x - prev_pose_.position.x;
                double dy = current_pose.position.y - prev_pose_.position.y;
                double dz = current_pose.position.z - prev_pose_.position.z;

                // Linear velocity (m/s)
                odom_msg.twist.twist.linear.x = dx / delta_time;
                odom_msg.twist.twist.linear.y = dy / delta_time;
                odom_msg.twist.twist.linear.z = dz / delta_time;

                // Compute orientation difference (using yaw angle)
                tf::Quaternion current_quat(
                    current_pose.orientation.x,
                    current_pose.orientation.y,
                    current_pose.orientation.z,
                    current_pose.orientation.w);
                tf::Quaternion prev_quat(
                    prev_pose_.orientation.x,
                    prev_pose_.orientation.y,
                    prev_pose_.orientation.z,
                    prev_pose_.orientation.w);

                // Convert quaternions to Euler angles (roll, pitch, yaw)
                double roll_curr, pitch_curr, yaw_curr;
                double roll_prev, pitch_prev, yaw_prev;
                tf::Matrix3x3(current_quat).getRPY(roll_curr, pitch_curr, yaw_curr);
                tf::Matrix3x3(prev_quat).getRPY(roll_prev, pitch_prev, yaw_prev);

                // Angular velocity (rad/s) in the z direction (yaw)
                odom_msg.twist.twist.angular.z = (yaw_curr - yaw_prev) / delta_time;
            }
        }

        // Publish the Odometry message
        odom_pub_.publish(odom_msg);

        // Update previous pose and time for the next iteration
        prev_pose_ = current_pose;
        prev_time_ = current_time;
    }

    ros::Subscriber pose_sub_;   // Subscriber for /robot/pose
    ros::Publisher odom_pub_;    // Publisher for /odom
    geometry_msgs::Pose prev_pose_; // To store previous pose
    ros::Time prev_time_;           // To store previous time
};

int main(int argc, char** argv)
{
    // Initialize ROS
    ros::init(argc, argv, "odom_publisher");

    // Create an OdometryPublisher object
    OdometryPublisher odom_publisher;

    // Spin to keep the node alive and process callbacks
    ros::spin();

    return 0;
}
