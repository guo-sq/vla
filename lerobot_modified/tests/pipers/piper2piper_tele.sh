
#!/bin/bash


# sudo chmod 777 /dev/ttyACM0
# sudo chmod 777 /dev/video2

sudo chmod 777 /dev/video2
sudo chmod 777 /dev/video3
sudo chmod 777 /dev/video4
sudo chmod 777 /dev/video5
sudo chmod 777 /dev/video6


# lerobot-find-port
bash ../piper_sdk/piper_sdk/find_all_can_port.sh
# bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can0 1000000 1-8:1.0
# bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can1 1000000 1-5:1.0

bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can2 1000000 1-3:1.0

# follower left-right
bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can0 1000000 1-10:1.0


bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can1 1000000 1-4:1.0
bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can3 1000000 1-8:1.0

# 0-3 ok
# 0-2 no
# 1-2 no




# python -m lerobot.teleoperate --robot.type=piper \
#     --robot.port=can0 \
#     --robot.id=my_awesome_follower_arm_piper_can0 \
#     --robot.cameras="{head: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
#     --teleop.type=piper_leader \
#     --teleop.port=can2 \
#     --teleop.id=my_awesome_leader_arm_piper_can1 \
#     --display_data=true



python -m lerobot.teleoperate --robot.type=piper \
    --robot.port=can3 \
    --robot.id=my_awesome_follower_arm_piper_can0 \
    --robot.cameras="{head: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=piper_leader \
    --teleop.port=can1 \
    --teleop.id=my_awesome_leader_arm_piper_can1 \
    --display_data=true