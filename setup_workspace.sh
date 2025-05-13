echo "WARNING this will delete ~/rotorpy_ws if it already exists. Proceed with caution."
read -p "Press enter to continue"

sudo rm -rf ~/rotorpy_ws

mkdir ~/rotorpy_ws
cd ~/rotorpy_ws
mkdir ~/rotorpy_ws/src/
cd ~/rotorpy_ws/src/
git clone https://github.com/KumarRobotics/kr_mav_control
cd ~/rotorpy_ws/kr_mav_control	
git submodule update --init --recursive	
cd ~/rotorpy_ws/src/
git clone https://github.com/whoenig/crazyflie_ros.git
cd ~/rotorpy_ws/src/crazyflie_ros	
git submodule update --init --recursive	
cd ~/rotorpy_ws/src/	
git clone https://github.com/KumarRobotics/multi_mav_manager.git
cd ~/rotorpy_ws/src/multi_mav_manager	
git submodule update --init --recursive	
cd ~/rotorpy_ws/src/
git clone https://github.com/spencerfolk/mocap_lidar_ros.git
cd ~/rotorpy_ws/src/mocap_lidar_ros
git submodule update --init --recursive
cd ~/rotorpy_ws/src/
git clone https://github.com/spencerfolk/rotorpy_hardware.git