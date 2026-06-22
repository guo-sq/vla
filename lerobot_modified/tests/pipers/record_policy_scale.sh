#!/bin/bash


/home/wujie1/miniconda3/envs/pilebot/bin/python -m lerobot.record_policy_scale --robot.type=bi_piper_follower \
    --robot.left_arm_port=can3   \
    --robot.right_arm_port=can0   \
    --robot.cameras="{head: {type: opencv, index_or_path: 13, width: 640, height: 480, fps: 30},left_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30},right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}"  \
    --robot.id=my_awesome_bi_piper_follower_arm3  \
    --teleop.type=bi_piper_leader  \
    --teleop.left_arm_port=can2   \
    --teleop.right_arm_port=can1   \
    --teleop.id=my_awesome_bi_piper_leader_arm3  \
    --dataset.repo_id=1210_bi/record.clothes.bipiper.v0121.policy.scale.1  \
    --dataset.num_episodes=30   \
    --dataset.episode_time_s=100   \
    --dataset.reset_time_s=10   \
    --dataset.single_task="fold clothes。"  \
    --policy_host=localhost  \
    --policy_port=8001   \
    --play_sounds=true 