#!/bin/bash

sudo chmod 777 /dev/video2
sudo chmod 777 /dev/video3
sudo chmod 777 /dev/video4
sudo chmod 777 /dev/video5
sudo chmod 777 /dev/video6

# leader left-right
bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can0 1000000 1-3:1.0
bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can1 1000000 1-4:1.0

# follower left-right
# follower left-right
bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can3 1000000 1-10:1.0
bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can2 1000000 1-8:1.0

for episode_id in {29..39}; do
  echo "Replaying episode $episode_id"
  python -m lerobot.replay --robot.type=bi_piper_follower \
      --robot.left_arm_port=can3 \
      --robot.right_arm_port=can2 \
      --robot.id=my_awesome_bi_piper_follower_arm3 \
      --robot.cameras="{head: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}}" \
      --dataset.repo_id=1114_bi/record.fold_towel.bipiper.v1114.8 \
      --dataset.episode=$episode_id 
done



