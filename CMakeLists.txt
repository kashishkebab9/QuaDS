cmake_minimum_required(VERSION 2.8.3)
project(rotorpy_hardware)

find_package(catkin REQUIRED
roscpp
rospy
std_msgs
geometry_msgs
message_generation
)

add_message_files(
  FILES
  ExtendedSO3Command.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
  geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES crazyflie_meam620
 CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)

include_directories(
# include
${catkin_INCLUDE_DIRS}
)