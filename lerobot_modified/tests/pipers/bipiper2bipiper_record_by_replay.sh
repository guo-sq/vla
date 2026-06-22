#!/bin/bash


sudo chmod 777 /dev/video12
sudo chmod 777 /dev/video4
sudo chmod 777 /dev/video6
# sudo chmod 777 /dev/video5
# sudo chmod 777 /dev/video6

# leader left-right
bash /home/wujie1/work/piper_sdk/piper_sdk/can_activate.sh can2 1000000 1-6:1.0    #左主
bash /home/wujie1/work/piper_sdk/piper_sdk/can_activate.sh can1 1000000 1-5:1.0    #右主


# follower left-right
# follower left-right
bash /home/wujie1/work/piper_sdk/piper_sdk/can_activate.sh can3 1000000 1-8:1.0    #左从
bash /home/wujie1/work/piper_sdk/piper_sdk/can_activate.sh can0 1000000 1-4:1.0    #右从




#--sub_episode_time_s=[0,3,15,20,25,30,35,40] 
python -m lerobot.record_replay --robot.type=bi_piper_follower \
    --robot.left_arm_port=can3 \
    --robot.right_arm_port=can0 \
    --robot.id=my_awesome_bi_piper_follower_arm3 \
    --robot.cameras="{head: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30},left_wrist: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30},right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --policy_host="127.0.0.1" \
    --policy_port=8000 \
    --display_data=true \
    --dataset.repo_id=1210_bi/replay.record.clothes.bipiper.v1215.1 \
    --dataset.num_episodes=11 \
    --dataset.episode_time_s=110 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="fold clothes。" \
    