#!/bin/bash

sudo chmod 777 /dev/video12
sudo chmod 777 /dev/video4
sudo chmod 777 /dev/video10


# leader left-right
bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can0 1000000 1-8:1.0 
bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can2 1000000 1-10:1.0

# follower left-right
# follower left-right
bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can1 1000000 1-5:1.0
bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can3 1000000 1-4:1.0

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
#--sub_episode_time_s=[0,3,15,20] 

python -m lerobot.record --robot.type=bi_piper_follower \
    --robot.left_arm_port=can1 \
    --robot.right_arm_port=can3 \
    --robot.id=my_awesome_bi_piper_follower_arm3 \
    --robot.cameras="{head: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --teleop.type=bi_piper_leader \
    --teleop.left_arm_port=can0 \
    --teleop.right_arm_port=can2 \
    --teleop.id=my_awesome_bi_piper_leader_arm3 \
    --display_data=true \
    --dataset.repo_id=1128_bi/record.xiangqi.bipiper.v1216.1 \
    --dataset.num_episodes=36 \
    --dataset.episode_time_s=25 \
    --dataset.reset_time_s=4 \
    --dataset.single_task="将绿色点标记的棋子移动到蓝色点标记的位置。" \
    --sub_episode_time_s=[0,3,15,20] 



