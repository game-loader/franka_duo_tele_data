#!/usr/bin/env bash
# Relaunch the ZED head camera on host 172.16.0.50. Kills only its own previous launch/container PIDs.
set +u
for P in $(ps -eo pid,args | awk '/zed_camera\.launch\.py/ && !/awk/ {print $1}') $(ps -eo pid,args | awk '/component_container_isolated/ && /head_camera/ && !/awk/ {print $1}'); do kill "$P" 2>/dev/null; done
sleep 3
cd ~/ros2_ws && source install/setup.bash 2>/dev/null; source ~/tmr_env.sh 2>/dev/null
mkdir -p ~/log
nohup setsid ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedm namespace:=head_camera publish_tf:=false serial_number:=17064700 > ~/log/zed5.log 2>&1 < /dev/null &
sleep 30
grep -i 'error\|opened\|Camera Model\|died' ~/log/zed5.log | tail -4 | cut -c1-160
echo '=== hz'; timeout 6 ros2 topic hz /head_camera/zed/rgb/color/rect/image 2>/dev/null | grep average | head -1
echo '=== encoding/size'; timeout 5 ros2 topic echo --once /head_camera/zed/rgb/color/rect/image --field encoding 2>/dev/null | head -1; timeout 5 ros2 topic echo --once /head_camera/zed/rgb/color/rect/image --field height 2>/dev/null | head -1
