#!/bin/bash

# Define the workspace directory
ROS_WS=~/rotorpy_ws

# Ensure the workspace exists
if [ ! -d "$ROS_WS" ]; then
  echo "Workspace directory $ROS_WS does not exist. Please create it first."
  exit 1
fi

xhost +

# Run the Docker container
sudo docker run -it --rm \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="$ROS_WS":/workspace \
  --volume=/tmp/.X11-unix:/tmp/.X11-unix \
  --volume=/dev/bus:/dev/bus \
  --workdir="/workspace" \
  --name ros-noetic-rotorpy_hardware \
  --network host \
  --privileged \
  rotorpy_hardware  # Name of your Docker image