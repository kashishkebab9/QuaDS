FROM osrf/ros:noetic-desktop

# Set environment variables
ENV ROS_DOMAIN_ID=1

# Create necessary directories for the user
RUN mkdir -p /run/user/1000 && \
    chmod 0700 /run/user/1000

# Install dependencies
RUN apt-get update && apt-get install -y \
    vim \
    python3-pip \ 
    python3-venv \
    iproute2 \
    libusb-1.0-0-dev \
    iputils-ping \
    tmux

# Install additional ROS dependencies (if needed)
RUN apt-get install -y \
    ros-noetic-catkin \ 
    python3-catkin-tools \
    ros-noetic-tf2-eigen \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-roscpp \
    ros-noetic-std-msgs

# Install Python dependencies
RUN pip install numpy==1.24.0 matplotlib rotorpy

# Setup workspace directory (not strictly needed if mounting the workspace from the host)
RUN mkdir -p /workspace

# Add ROS sourcing to bashrc so new terminals also have ROS access. 
RUN echo 'source /opt/ros/noetic/setup.bash' >> /root/.bashrc && \
    echo 'if [ -f "/workspace/devel/setup.bash" ]; then' >> /root/.bashrc && \
    echo '  source /workspace/devel/setup.bash' >> /root/.bashrc && \
    echo 'fi' >> /root/.bashrc

# Clear each new terminal. 
RUN echo 'clear' >> /root/.bashrc

# Add rotorpy aliases to bashrc. 
RUN echo 'alias rotorpy_gui="roslaunch rotorpy_hardware main.launch"' >> /root/.bashrc && \
    echo 'alias rotorpy_run="roslaunch rotorpy_hardware run.launch"' >> /root/.bashrc && \
    echo 'alias rotorpy_gui_lidar="roslaunch rotorpy_hardware main_withlidar.launch"' >> /root/.bashrc

# # Set default cmake args
# RUN catkin config -DCMAKE_BUILD_TYPE=Release

# Default command to run when the container starts
CMD ["bash"]
