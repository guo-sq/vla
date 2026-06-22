
export HF_ENDPOINT=https://hf-mirror.com

export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
HUGGINGFACE_TOKEN=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
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential

HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER

export WANDB_API_KEY=77a6f87ffb15b774adba94a59a7d0687344ff8da
wandb login
export WANDB_MODE=offline
export WANDB_RUN_ID=test_5090_lerobot


conda install ffmpeg=7.1.1 -c conda-forge
sudo apt-get install cmake build-essential python-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config



pip install wandb
sudo apt-get install -y libglfw3 libglfw3-dev libgl1-mesa-glx

# cameras: hikivision camera
/dev/tty.usbmodem11106


# when connected to hardware
sudo chmod  777 /dev/ttyACM0
sudo chmod  777 /dev/ttyACM1
sudo chmod  777 /dev/ttyACM2
sudo chmod  777 /dev/ttyACM3
sudo chmod  777 /dev/ttyACM4
sudo chmod  777 /dev/ttyACM5

sudo chmod  777 /dev/video0
sudo chmod  777 /dev/video1
sudo chmod  777 /dev/video2
sudo chmod  777 /dev/video3
sudo chmod  777 /dev/video4
sudo chmod  777 /dev/video5
sudo chmod  777 /dev/video6
sudo chmod  777 /dev/video7
sudo chmod  777 /dev/video8

lerobot-find-port
lerobot-find-cameras opencv
lerobot-find-cameras realsense

# SO-100
lerobot-setup-motors \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM2  # <- paste here the port found at previous step
lerobot-setup-motors \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM3  # <- paste here the port found at previous step

lerobot-teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM2 \
    --robot.id=my_awesome_follower_arm_100 \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM3 \
    --teleop.id=my_awesome_leader_arm_100


# /home/xuwenda/.cache/huggingface/lerobot/calibration/robots/so100_follower/my_awesome_follower_arm.json
# /home/xuwenda/.cache/huggingface/lerobot/calibration/robots/so100_leader/my_awesome_leader_arm.json
lerobot-calibrate \
    --robot.type=piper \
    --robot.port=can1 \
    --robot.id=my_awesome_follower_arm_piper_can1 \
    --robot.cameras="{head: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    

lerobot-calibrate \
    --teleop.type=piper_leader \
    --teleop.port=can1 \
    --teleop.id=my_awesome_leader_arm_piper_can1 \
    # --robot.cameras="{head: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    

lerobot-teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM2 \
    --robot.id=my_awesome_follower_arm_100 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm

# lerobot-calibrate \
#     --robot.type=so100_leader \
#     --robot.port=/dev/ttyACM2 \
#     --robot.id=my_awesome_leader_arm_100

# --------------------------------------------------------#
# SO-101
# --------------------------------------------------------#
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1  # <- paste here the port found at previous step

lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0  # <- paste here the port found at previous step

lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm # <- Give the robot a unique name

# /Users/salvame/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_awesome_follower_arm.json

# lerobot-calibrate \
#     --robot.type=so101_leader \
#     --robot.port=/dev/ttyACM0 \
#     --robot.id=my_awesome_leader_arm

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30, fourcc: 'MJPG'}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true

lerobot-teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30, fourcc: 'MJPG'}, wrist: {type: intelrealsense, serial_number_or_name: "230422272451", width: 640, height: 480, fps: 30, color_mode: RGB, use_depth: true}, side: {type: intelrealsense, serial_number_or_name: "406122072203", width: 640, height: 480, fps: 30, color_mode: RGB, use_depth: true}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true


# ------------------------------------- #
# teleoperate for arxx5
# ------------------------------------- #
sudo -S slcand -o -f -s8 /dev/arxcan1 can1 && sudo ifconfig can1 up
sudo -S slcand -o -f -s8 /dev/arxcan3 can3 && sudo ifconfig can3 up


# lerobot-calibrate \
#     --robot.type=so101_follower \
#     --robot.port=/dev/ttyACM1 \
#     --robot.id=my_awesome_follower_arm

lerobot-find-cameras opencv

rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record-test-arxx5

lerobot-record \
    --robot.type=arxx5 \
    --robot.cameras="{front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, right: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, right: {type: opencv, index_or_path: 22, width: 640, height: 480, fps: 30}}" \
    --robot.id=arxx5 \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record-test-arxx5 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the yellow toy duck and put it in the yellow paper bag."

lerobot-replay \
    --robot.type=arxx5 \
    --robot.id=arxx5 \
    --robot.cameras="{front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, right: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left: {type: opencv, index_or_path: 22, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record-test-arxx5 \
    --dataset.episode=0 # choose the episode you want to replay



# ------------------------------------- #
# teleoperate for arxx5_bimanual
# ------------------------------------- #
lerobot-find-cameras opencv
sudo -S slcand -o -f -s8 /dev/arxcan1 can1 && sudo ifconfig can1 up
sudo -S slcand -o -f -s8 /dev/arxcan3 can3 && sudo ifconfig can3 up

rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_toy_duck

lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 22, width: 640, height: 480, fps: 30}}" \
    --robot.id=arxx5_bimanual \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_toy_duck \
    --dataset.num_episodes=30 \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="pick up the yellow toy duck and put it into write paper bag by right arm"


lerobot-replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_toy_duck \
    --dataset.episode=6


lerobot-replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/datasets.left_duck.right_duck \
    --dataset.episode=6

# ------------------------------------- #
# teleoperate for piper
# ------------------------------------- #
bash can_activate.sh can0 1000000

python -m lerobot.teleoperate --robot.type=piper \
    --robot.port=can0 \
    --robot.id=my_awesome_follower_arm_piper \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true


rm -rf /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/record.industrial.s3.v0
lerobot-record \
    --robot.type=piper \
    --robot.port=can0 \
    --robot.id=my_awesome_follower_arm_piper \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record.industrial.s3.v0 \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the grey magnet tube and attach it to another tube on the white holder."


export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
HUGGINGFACE_TOKEN=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
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential

HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER


rm -rf /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/record-test
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM3 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30, fourcc: 'MJPG'}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM2 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.num_episodes=5 \
    --dataset.single_task="Grab the white cube"

lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.episode=0 # choose the episode you want to replay

https://wandb.ai/authorize?ref=models

export WANDB_API_KEY=77a6f87ffb15b774adba94a59a7d0687344ff8da
wandb login
# export WANDB_MODE=offline
export WANDB_MODE=online
export WANDB_RUN_ID=wuexkizn

# hil-il
python -m lerobot.scripts.rl.gym_manipulator --config_path lerobot_example_config_files/env_config_gym_hil_il.keyboard.json
python -m lerobot.scripts.rl.gym_manipulator --config_path lerobot_example_config_files/env_config_gym_hil_il.json

# hil-serl
# mode=null
python -m lerobot.scripts.rl.gym_manipulator --config_path lerobot_example_config_files/gym_hil_env.json

# train RL in simulation: run the commands at the same time
python -m lerobot.scripts.rl.actor --config_path lerobot_example_config_files/train_gym_hil_env.json
python -m lerobot.scripts.rl.learner --config_path lerobot_example_config_files/train_gym_hil_env.json

######################################################################
# diffusion pusht: https://huggingface.co/lerobot/diffusion_pusht
######################################################################
python -m lerobot.scripts.visualize_dataset \
    --repo-id lerobot/pusht \
    --episode-index 0

lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda

# DONE in local pc: diffusion policy is too large for mac: model size only 200M， 
python src/lerobot/scripts/train.py \
    --output_dir=outputs/train/diffusion_pusht \
    --policy.type=diffusion \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/pusht \
    --seed=100000 \
    --env.type=pusht \
    --batch_size=128 \
    --steps=200000 \
    --eval_freq=25000 \
    --save_freq=25000 \
    --job_name=diffusion_pusht_training \
    --wandb.enable=true

######################################################################
# smol: based on https://huggingface.co/lerobot/smolvla_base
######################################################################
# INFO 2025-08-31 00:05:29 ts/train.py:163 num_learnable_params=99880992 (100M)
# INFO 2025-08-31 00:05:29 ts/train.py:164 num_total_params=450046212 (450M)
# finetune smolvla_base on svla_so101_pickplace

# train from scratch
python lerobot/scripts/train.py \
  --policy.type=smolvla \
  --dataset.repo_id=lerobot/svla_so100_stacking \
  --batch_size=64 \
  --steps=200000

# use smolvla_base finetune on self collected dataset
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/mydataset \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_trainieeng_finetune \
  --policy.device=cuda \
  --wandb.enable=true

# train base: 2G
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --policy.push_to_hub=false \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_trainieeng \
  --policy.device=cuda \
  --wandb.enable=true

lerobot-eval \
  --policy.path=lerobot/smolvla_base \
    --env.type=aloha \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda


######################################################################
# act: https://huggingface.co/lerobot/act_so101_pickplace
######################################################################
python src/lerobot/scripts/train.py \
    --output_dir=outputs/train/act.arxx5_bimanual \
    --policy.type=act \
    --policy.push_to_hub=false \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_toy_duck \
    --seed=100000 \
    --batch_size=32 \
    --steps=200000 \
    --save_freq=10000 \
    --job_name=act.arxx5_bimanual \
    --wandb.enable=false

python src/lerobot/scripts/train.py \
    --output_dir=outputs/train/act.arxx5_bimanual.chunk_size_100.20w \
    --policy.type=act \
    --policy.chunk_size=100 \
    --policy.n_action_steps=1 \
    --policy.temporal_ensemble_coeff=0.01 \
    --policy.push_to_hub=false \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_toy_duck \
    --seed=0 \
    --batch_size=32 \
    --steps=200000 \
    --save_freq=5000 \
    --job_name=act.arxx5_bimanual.chunk_size_100.20w \
    --wandb.enable=true


lerobot-replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_toy_duck \
    --dataset.episode=8

python -m lerobot.scripts.visualize_dataset \
    --repo-id heyuan1993/record.arxx5_bimanual.right_arm_grab_toy_duck \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_toy_duck \
    --episode-index 1

python -m lerobot.datasets.v21.convert_dataset_v20_to_v21 \
    --repo-id heyuan1993/datasets.act.right.1002.duck.800steps.left_no_lift \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/datasets.act.right.1002.duck.800steps.left_no_lift \

python -m lerobot.scripts.visualize_dataset \
    --repo-id heyuan1993/datasets.act.right.1002.duck.800steps.left_no_lift \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/datasets.act.right.1002.duck.800steps.left_no_lift \
    --episode-index 1

lerobot-replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/datasets_gr00t/datasets.act.right.1002.duck.800steps.left_no_lift \
    --dataset.episode=9


rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/eval_arxx5_bimanual.right_arm_grab_toy_duck.20w
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --policy.path=outputs/train/act.arxx5_bimanual.chunk_size_100.20w/checkpoints/last/pretrained_model \
    --display_data=true \
    --dataset.repo_id=heyuan1993/eval_arxx5_bimanual.right_arm_grab_toy_duck.20w \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="pick up the yellow toy duck and put it into write paper bag by right arm"




# ---------------------------------------------------------------------------------------- #
# act: right_arm_grab_messy_duck_long_30s
# ---------------------------------------------------------------------------------------- #
python src/lerobot/scripts/train.py \
    --output_dir=outputs/train/record.arxx5_bimanual.right_arm_grab_messy_duck_long_30s.1007.chunk_size_100.20w \
    --policy.type=act \
    --policy.chunk_size=100 \
    --policy.n_action_steps=1 \
    --policy.temporal_ensemble_coeff=0.01 \
    --policy.push_to_hub=false \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.duck_long_30s.1007 \
    --seed=0 \
    --batch_size=64 \
    --steps=300000 \
    --save_freq=5000 \
    --job_name=record.arxx5_bimanual.right_arm_grab_messy_duck_long_30s.1007.chunk_size_100.20w \
    --wandb.enable=true


lerobot-replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.duck_long_30s.1007 \
    --dataset.episode=10


python -m lerobot.scripts.visualize_dataset \
    --repo-id heyuan1993/record.arxx5_bimanual.duck_long_30s.1007 \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.duck_long_30s.1007 \
    --episode-index 2


rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/eval_arxx5_bimanual.right_arm_grab_messy_duck_long_30s.20w
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --policy.path=outputs/train/record.arxx5_bimanual.right_arm_grab_messy_duck_long_30s.1007.chunk_size_100.20w/checkpoints/150000/pretrained_model \
    --display_data=true \
    --dataset.repo_id=heyuan1993/eval_arxx5_bimanual.right_arm_grab_messy_duck_long_30s.20w \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the toy ducks and pub them into white paper box."



# ---------------------------------------------------------------------------------------- #
# act: right_arm_grab_messy_duck_short_12s
# ---------------------------------------------------------------------------------------- #
python src/lerobot/scripts/train.py \
    --output_dir=outputs/train/record.arxx5_bimanual.right_arm_grab_messy_duck_short_12s.1007.chunk_size_100.20w \
    --policy.type=act \
    --policy.chunk_size=100 \
    --policy.n_action_steps=1 \
    --policy.temporal_ensemble_coeff=0.01 \
    --policy.push_to_hub=false \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.duck_short_12s.1007 \
    --seed=0 \
    --batch_size=64 \
    --steps=300000 \
    --save_freq=5000 \
    --job_name=record.arxx5_bimanual.right_arm_grab_messy_duck_short_12s.1007.chunk_size_100.20w \
    --wandb.enable=true

# resume
python src/lerobot/scripts/train.py \
	--config_path=outputs/train/record.arxx5_bimanual.right_arm_grab_messy_duck_short_12s.1007.chunk_size_100.20w/checkpoints/010000/pretrained_model/train_config.json \
    --resume=true \
    --policy.checkpoint_path=outputs/train/record.arxx5_bimanual.right_arm_grab_messy_duck_short_12s.1007.chunk_size_100.20w/checkpoints/010000 \
    --output_dir=outputs/train/record.arxx5_bimanual.right_arm_grab_messy_duck_short_12s.1007.chunk_size_100.20w.resume \
    --policy.type=act \
    --policy.chunk_size=100 \
    --policy.n_action_steps=1 \
    --policy.temporal_ensemble_coeff=0.01 \
    --policy.push_to_hub=false \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.duck_short_12s.1007 \
    --seed=0 \
    --batch_size=64 \
    --steps=300000 \
    --save_freq=5000 \
    --job_name=record.arxx5_bimanual.right_arm_grab_messy_duck_short_12s.1007.chunk_size_100.20w.resume \
    --wandb.enable=true

rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/eval_arxx5_bimanual.right_arm_grab_messy_duck_short_12s.20w
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --policy.path=outputs/train/record.arxx5_bimanual.right_arm_grab_messy_duck_short_12s.1007.chunk_size_100.20w/checkpoints/020000/pretrained_model \
    --display_data=true \
    --dataset.repo_id=heyuan1993/eval_arxx5_bimanual.right_arm_grab_messy_duck_short_12s.20w \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=12 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the toy duck and pub it into white paper box."

python -m lerobot.scripts.visualize_dataset \
    --repo-id heyuan1993/record.arxx5_bimanual.duck_short_12s.1007 \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.duck_short_12s.1007 \
    --episode-index 2


lerobot-replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/ecord.arxx5_bimanual.duck_short_12s.1007 \
    --dataset.episode=10

# ---------------------------------------------------------------------------------------- #
# act: right_arm_grab_magetic_tube
# ---------------------------------------------------------------------------------------- #

rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/eval_arxx5_bimanual.right_arm_grab_metal_tube.20w
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --policy.path=outputs/train/act.arxx5_bimanual.right_arm_grab_magetic_tube.chunk_size_100.20w/checkpoints/100000/pretrained_model \
    --display_data=true \
    --dataset.repo_id=heyuan1993/eval_arxx5_bimanual.right_arm_grab_metal_tube.20w \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=12 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the grey metal tube and put it into white paper box."


# ---------------------------------------------------------------------------------------- #
# act: left_right_arm_grab_magetic_tube
# ---------------------------------------------------------------------------------------- #

rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/eval_arxx5_bimanual.right_arm_grab_and_attach_magetic_tube.20w
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}}" \
    --policy.path=outputs/train/act.arxx5_bimanual.right_arm_grab_and_attach_magetic_tube.chunk_size_100.20w/checkpoints/090000/pretrained_model \
    --display_data=true \
    --dataset.repo_id=heyuan1993/eval_arxx5_bimanual.right_arm_grab_and_attach_magetic_tube.20w \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the grey metal tube and attach it into white shelf."

lerobot-replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_and_attach_magetic_tube.1007 \
    --dataset.episode=15

python -m lerobot.scripts.visualize_dataset \
    --repo-id heyuan1993/record.arxx5_bimanual.right_arm_grab_and_attach_magetic_tube.1007 \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_and_attach_magetic_tube.1007 \
    --episode-index 15


# ---------------------------------------------------------------------------------------- #
# act: record.arxx5_bimanual.right_arm_grab_duck.1008.fix
# ---------------------------------------------------------------------------------------- #

python src/lerobot/scripts/train.py \
    --output_dir=outputs/train/record.arxx5_bimanual.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.chunk_size_100.20w \
    --policy.type=act \
    --policy.chunk_size=100 \
    --policy.n_action_steps=1 \
    --policy.temporal_ensemble_coeff=0.01 \
    --policy.push_to_hub=false \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix \
    --seed=0 \
    --batch_size=32 \
    --steps=200000 \
    --save_freq=5000 \
    --job_name=record.arxx5_bimanual.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.chunk_size_100.20w \
    --wandb.enable=false

python -m lerobot.scripts.visualize_dataset \
    --repo-id heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix \
    --episode-index 15

lerobot-replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix \
    --dataset.episode=20

rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/eval_arxx5_bimanual.right_arm_grab_duck.1008.fix.chunk_size_100.20w
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --policy.path=outputs/train/record.arxx5_bimanual.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.chunk_size_100.20w/checkpoints/100000/pretrained_model \
    --policy.chunk_size=100 \
    --policy.n_action_steps=50 \
    --display_data=true \
    --dataset.repo_id=heyuan1993/eval_arxx5_bimanual.right_arm_grab_duck.1008.fix.chunk_size_100.20w \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the toy duck and pub it into white paper box."


# ---------------------------------------------------------------------------------------- #
# act: record.arxx5_bimanual.right_arm_grab_duck.1008.fix.chunk_size_30
# ---------------------------------------------------------------------------------------- #

python src/lerobot/scripts/train.py \
    --output_dir=outputs/train/record.arxx5_bimanual.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.chunk_size_30.20w \
    --policy.type=act \
    --policy.chunk_size=30 \
    --policy.n_action_steps=1 \
    --policy.temporal_ensemble_coeff=0.01 \
    --policy.push_to_hub=false \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix \
    --seed=0 \
    --batch_size=32 \
    --steps=200000 \
    --save_freq=5000 \
    --job_name=record.arxx5_bimanual.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.chunk_size_30.20w \
    --wandb.enable=false

rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/eval_arxx5_bimanual.right_arm_grab_duck.1008.fix.chunk_size_30.20w
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --policy.path=outputs/train/record.arxx5_bimanual.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.chunk_size_30.20w/checkpoints/110000/pretrained_model \
    --policy.chunk_size=30 \
    --display_data=true \
    --dataset.repo_id=heyuan1993/eval_arxx5_bimanual.right_arm_grab_duck.1008.fix.chunk_size_30.20w \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=3 \
    --dataset.single_task="Grab the toy duck and pub it into white paper box."
# ---------------------------------------------------------------------------------------- #
# svla_so101_pickplace
# ---------------------------------------------------------------------------------------- #

# train act on svla_so101_pickplace: res18, 52M, 1 step takes 10min on mac
lerobot-train \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --policy.type=act \
  --batch_size=32 \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_policy

lerobot-train \
    --policy.type=act \
    --policy.push_to_hub=false \
    --dataset.repo_id=danaaubakirova/koch_test \
    --output_dir=outputs/train/act_koch_test \
    --job_name=act_koch_test \
    --batch_size=16 \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.repo_id=${HF_USER}/act_koch_test

######################################################################
# pi0: https://huggingface.co/lerobot/pi0: 14G
######################################################################
# load pretrained pi0: missing keys
lerobot-train \
    --policy.path=lerobot/pi0 \
    --policy.push_to_hub=false \
    --dataset.repo_id=danaaubakirova/koch_test \
    --output_dir=outputs/train/pi0_koch_test \
    --job_name=pi0_koch_test \
    --batch_size=32 \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.repo_id=${HF_USER}/pi0_koch_test

python -c "
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
try:
    policy = PI0Policy.from_pretrained('lerobot/pi0')
    print('Model loaded successfully')
except Exception as e:
    print(f'Model loading failed: {e}')
"


######################################################################
# pi0: https://huggingface.co/lerobot/pi0-fast: 11G
######################################################################
# https://huggingface.co/blog/pi0

lerobot-train \
    --policy.path=lerobot/pi0fast_base \
    --policy.push_to_hub=false \
    --dataset.repo_id=danaaubakirova/koch_test \
    --job_name=pi0_fast_koch_test \
    --batch_size=1 \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.repo_id=${HF_USER}/pi0_fast_koch_test


######################################################################
# https://huggingface.co/lerobot/vqbet_pusht
######################################################################

######################################################################
# https://huggingface.co/lerobot/act_aloha_sim_insertion_human
# https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human
######################################################################

# train act on self collected dataset
lerobot-train \
  --dataset.repo_id=${HF_USER}/so101_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_policy

######################################################################
# Dataset Downloads
######################################################################

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, up: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --steps=1000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_train.pickplace \
  --policy.device=cuda \
  --wandb.enable=true

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, up: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
  --dataset.single_task="Grasp a lego block and put it in the bin." \
  --dataset.repo_id=${HF_USER}/eval_smolvla_test \
  --dataset.episode_time_s=20 \
  --dataset.num_episodes=2 \
  --policy.path=outputs/train/my_smolvla/checkpoints/last/pretrained_model \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm

# record dataset: grab white cube and put it in the blue box
rm -rf /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/record.grab_white_cube.v2
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, up: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record.grab_white_cube.v2 \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the white cube and put it in the blue box"

# finetune dataset on smolvla
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/record.grab_white_cube.v2 \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --steps=5000 \
  --output_dir=outputs/train/my_smolvla.grab_white_cube.v2 \
  --job_name=my_smolvla.grab_white_cube.v2 \
  --policy.device=cuda \
  --wandb.enable=true

rm -rf /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/eval_smolvla_test.grab_white_cube
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, up: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
  --dataset.single_task="Grab the white cube and put it in the blue box" \
  --dataset.repo_id=${HF_USER}/eval_smolvla_test.grab_white_cube \
  --display_data=true \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=5 \
  --dataset.num_episodes=10 \
  --policy.path=outputs/train/my_smolvla.grab_white_cube.v2/checkpoints/last/pretrained_model \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm


# record dataset: grab white cube and put it in the blue box
rm -rf /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/record.grab_white_cube.v3
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record.grab_white_cube.v3 \
    --dataset.num_episodes=30 \
    --dataset.episode_time_s=12 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the grey tube and put it in the blue box"


# finetune dataset on smolvla
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/record.grab_white_cube.v3 \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --steps=2000 \
  --output_dir=outputs/train/my_smolvla.grab_white_cube.v3 \
  --job_name=my_smolvla.grab_white_cube.v3 \
  --policy.device=cuda \
  --wandb.enable=true

rm -rf /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/eval_smolvla_test.grab_white_cube2
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
  --dataset.single_task="Grab the grey tube and put it in the blue box" \
  --dataset.repo_id=${HF_USER}/eval_smolvla_test.grab_white_cube2 \
  --display_data=true \
  --dataset.episode_time_s=12 \
  --dataset.reset_time_s=5 \
  --dataset.num_episodes=10 \
  --policy.path=outputs/train/my_smolvla.grab_white_cube.v3/checkpoints/last/pretrained_model \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm


# clear desktop
# record dataset: grab white cube and put it in the blue box
rm -rf /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/record.so101.table_cleanup.v0
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record.so101.table_cleanup.v0 \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab pens and place into pen holder."

rm -rf /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/record.so101.grab_nail.v0
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record.so101.grab_nail.v0 \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab nails and place into pen holder."


rm -rf /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/record.so101.grab_toy_duck.v0
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record.so101.grab_toy_duck.v0 \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab toy ducks and place into pen holder."

rm -rf /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/record.so101.grab_toy_duck.v1
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record.so101.grab_toy_duck.v1 \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab toy ducks and place into pen holder in order."



# --------------------------------------------------
# hil-serl
# --------------------------------------------------
python -m lerobot.scripts.find_joint_limits \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm


python -m lerobot.scripts.find_joint_limits \
    --robot.type=so101_follower_end_effector \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_end_effector_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm


Max ee position [0.1377, 0.0151, -0.0133]
Min ee position [0.1362, 0.0148, -0.0144]
Max joint pos position [-9.1868, -100.2198, 97.6264, 76.0, 6.4615, 4.807]
Min joint pos position [-9.2747, -105.7582, 96.9231, 72.3956, 6.4615, 3.3852]

lerobot-teleoperate \
    --robot.type=so101_follower_end_effector \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_end_effector_arm \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true

# step2: collect dataset
rm -rf /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/pick_lift_cube
python -m lerobot.scripts.rl.gym_manipulator --config_path lerobot_example_config_files/env_config_so101.json

rm -rf /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/pick_lift_cube.v2
python -m lerobot.scripts.rl.gym_manipulator --config_path lerobot_example_config_files/env_config_so101.record.json


python -m lerobot.scripts.rl.crop_dataset_roi --repo-id /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/pick_lift_cube

# step3: train and eval a reward classifier
rm -rf outputs/train/reward_classifier_pick_lift_cube
lerobot-train --config_path lerobot_example_config_files/reward_classifier_train_config.json \
  --dataset.root=/home/xuwenda/.cache/huggingface/lerobot/heyuan1993/pick_lift_cube.v2/ \
  --policy.push_to_hub=false \
  --policy.repo_id=${HF_USER}/reward_classifier_pick_lift_cube \
  --output_dir=outputs/train/reward_classifier_pick_lift_cube \
  --job_name=reward_classifier_pick_lift_cube \
  --policy.device=cuda \
  --wandb.enable=true

rm -rf /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/pick_lift_cube.eval
python -m lerobot.scripts.rl.gym_manipulator --config_path lerobot_example_config_files/env_config_so101.json


# step4: 
# Training with Learner
python -m lerobot.scripts.rl.learner --config_path lerobot_example_config_files/train_config_hilserl_so101.json
# Training with Actor

python -m lerobot.scripts.rl.actor --config_path lerobot_example_config_files/train_config_hilserl_so101.json


#-------------------------------------------
# lerobot record arxx5: right_arm_grab_magetic_tube
#-------------------------------------------
lerobot-find-cameras opencv

rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_magetic_tube.1007
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_magetic_tube.1007 \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=12 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the grey metal tube and put it into white paper box."


lerobot-replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_magetic_tube.1007 \
    --dataset.episode=0

python -m lerobot.scripts.visualize_dataset \
    --repo-id heyuan1993/record.arxx5_bimanual.right_arm_grab_toy_duck \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_toy_duck \
    --episode-index 1


#-------------------------------------------
# lerobot record arxx5: right_arm_grab_and_attach_magetic_tube
#-------------------------------------------
rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_and_attach_magetic_tube.1007
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_and_attach_magetic_tube.1007 \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=3 \
    --dataset.single_task="Grab the grey metal tube and attach it on white shelf."


#-------------------------------------------
# lerobot record arxx5: right_arm_grab_messy_duck
#-------------------------------------------
rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_messy_duck.1007
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_messy_duck.1007 \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=3 \
    --dataset.single_task="Grab the toy duck and pub it into white paper box."


lerobot-replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_messy_duck.1007 \
    --dataset.episode=0


#-------------------------------------------
# lerobot record arxx5: right_arm_grab_messy_duck_long_30s
#-------------------------------------------
rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_messy_duck_long_30s.1007
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_messy_duck_long_30s.1007 \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=3 \
    --dataset.single_task="Grab the toy ducks and pub them into white paper box."

lerobot-replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_messy_duck_long_30s.1007 \
    --dataset.episode=0


python -m lerobot.scripts.visualize_dataset \
    --repo-id heyuan1993/record.arxx5_bimanual.right_arm_grab_toy_duck \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_toy_duck \
    --episode-index 1

python -m lerobot.scripts.visualize_dataset \
    --repo-id heyuan1993/record.arxx5_bimanual.duck_long_30s.1007 \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.duck_long_30s.1007 \
    --episode-index 1


#-------------------------------------------
# lerobot record arxx5: right arm grab toy duck
#-------------------------------------------
rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=3 \
    --dataset.single_task="Grab the toy duck and pub it into white paper box."


python scripts/eval_policy.py --plot --trajs=20 \
   --embodiment_tag new_embodiment \
   --model_path ./work_dirs/arxone_tricam_split.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.1w/checkpoint-4000  \
   --data_config arxone_tricam_split \
   --dataset-path ~/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix/ \
   --video_backend torchvision_av \
   --modality_keys right_arm right_gripper \
   --plot_state \
   --denoising_steps=8 \
   --save_plot_path ./work_dirs/arxone_tricam_split.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.1w/ckpt.4k.denoise8


python scripts/eval_policy.py --plot --trajs=20 --rtc --execution_horizon=8 --inference_latency_steps=4 \
   --embodiment_tag new_embodiment \
   --model_path ./work_dirs/arxone_tricam_split.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.1w/checkpoint-4000  \
   --data_config arxone_tricam_split \
   --dataset-path ~/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix/ \
   --video_backend torchvision_av \
   --modality_keys right_arm right_gripper \
   --plot_state \
   --denoising_steps=8 \
   --save_plot_path ./work_dirs/arxone_tricam_split.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.1w/ckpt.4k.denoise8.rtc.eh8

python scripts/eval_policy.py --plot --trajs=20 --rtc --execution_horizon=10 --inference_latency_steps=4 \
   --embodiment_tag new_embodiment \
   --model_path ./work_dirs/arxone_tricam_split.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.1w/checkpoint-4000  \
   --data_config arxone_tricam_split \
   --dataset-path ~/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix/ \
   --video_backend torchvision_av \
   --modality_keys right_arm right_gripper \
   --plot_state \
   --denoising_steps=8 \
   --save_plot_path ./work_dirs/arxone_tricam_split.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.1w/ckpt.4k.denoise8.rtc.eh10

python scripts/eval_policy.py --plot --trajs=20 --rtc --execution_horizon=8 --inference_latency_steps=4 \
   --embodiment_tag new_embodiment \
   --model_path ./work_dirs/arxone_tricam_split.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.1w/checkpoint-4000  \
   --data_config arxone_tricam_split \
   --dataset-path ~/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix/ \
   --video_backend torchvision_av \
   --modality_keys right_arm right_gripper \
   --plot_state \
   --denoising_steps=4 \
   --save_plot_path ./work_dirs/arxone_tricam_split.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.1w/ckpt.4k.denoise4.rtc.eh8

python scripts/eval_policy.py --plot --trajs=20 --rtc --execution_horizon=12 --inference_latency_steps=4 \
   --embodiment_tag new_embodiment \
   --model_path ./work_dirs/arxone_tricam_split.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.1w/checkpoint-4000  \
   --data_config arxone_tricam_split \
   --dataset-path ~/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix/ \
   --video_backend torchvision_av \
   --modality_keys right_arm right_gripper \
   --plot_state \
   --denoising_steps=12 \
   --save_plot_path ./work_dirs/arxone_tricam_split.record.arxx5_bimanual.right_arm_grab_duck.1008.fix.1w/ckpt.4k.denoise12.rtc.eh12


#-------------------------------------------
# lerobot record 1022: right_arm_grab_duck_slow_25s.60fps
#-------------------------------------------

rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck.fix_position.1022
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 60}, right_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 60}, left_wrist: {type: opencv, index_or_path: 28, width: 640, height: 480, fps: 60}}" \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck.fix_position.1022 \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=25 \
    --dataset.reset_time_s=3 \
    --dataset.single_task="Grab the toy duck and put it into white paper box."

python -m lerobot.scripts.visualize_dataset \
    --repo-id heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck.fix_position.1022 \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck.fix_position.1022 \
    --episode-index 2

python -m lerobot.replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 60}, right_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 60}, left_wrist: {type: opencv, index_or_path: 28, width: 640, height: 480, fps: 60}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck.fix_position.1022 \
    --dataset.episode=3

#-------------------------------------------
# lerobot record 1023: right_arm_grab_duck_slow_25s.diff_head.60fps
#-------------------------------------------

rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_head.60fps
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 60}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 60}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 60}}" \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_head.60fps \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=25 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the toy duck and put it into brown paper box."

python -m lerobot.scripts.visualize_dataset \
    --repo-id heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_head.60fps \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_head.60fps \
    --episode-index 1

python -m lerobot.replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 60}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 60}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 60}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_head.60fps \
    --dataset.episode=20

#-------------------------------------------
# lerobot record 1024: right_arm_grab_duck_slow_25s.diff_pos.1024
#-------------------------------------------

rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_pos.1024
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_pos.1024 \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=25 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the toy duck and put it into brown paper box."

python -m lerobot.scripts.visualize_dataset \
    --repo-id heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_pos.1024 \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_pos.1024 \
    --episode-index 1

python -m lerobot.replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_pos.1024 \
    --dataset.episode=20

#-------------------------------------------
# lerobot replay and record 1027: right_arm_grab_duck_slow_25s.diff_head.60fps
#-------------------------------------------

rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_head.replay.1027
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_head.replay.1027 \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=25 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the toy duck and put it into brown paper box." \
    --dataset_replay.repo_id=heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_head.60fps \
    --dataset_replay.num_episodes=50 \
    --dataset_replay.episode_time_s=25 \
    --dataset_replay.reset_time_s=5 \
    --dataset_replay.single_task="Grab the toy duck and put it into brown paper box."

python -m lerobot.scripts.visualize_dataset \
    --repo-id record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_head.replay.1027 \
    --root  /home/anyverse/.cache/huggingface/lerobot/record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_head.replay.1027 \
    --episode-index 1

python -m lerobot.replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=record.arxx5_bimanual.format.right_arm_grab_duck_slow_25s.diff_head.replay.1027 \
    --dataset.episode=20


# #-------------------------------------------
# # lerobot replay and record 1028: right_arm_grab_duck_normal.1008_fix
# #-------------------------------------------

# rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix.replay.1028
# lerobot-record \
#     --robot.type=arxx5_bimanual \
#     --robot.id=arxx5_bimanual \
#     --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
#     --display_data=true \
#     --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix.replay.1028 \
#     --dataset.num_episodes=50 \
#     --dataset.episode_time_s=10 \
#     --dataset.reset_time_s=5 \
#     --dataset.single_task="Grab the toy duck and pub it into white paper box." \
#     --dataset_replay.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix \
#     --dataset_replay.num_episodes=50 \
#     --dataset_replay.episode_time_s=10 \
#     --dataset_replay.reset_time_s=5 \
#     --dataset_replay.single_task="Grab the toy duck and pub it into white paper box."



# rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_magetic_tube.1007.replay.1028
# lerobot-record \
#     --robot.type=arxx5_bimanual \
#     --robot.id=arxx5_bimanual \
#     --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
#     --display_data=true \
#     --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_magetic_tube.1007.replay.1028 \
#     --dataset.num_episodes=50 \
#     --dataset.episode_time_s=12 \
#     --dataset.reset_time_s=5 \
#     --dataset.single_task="Grab the grey metal tube and put it into white paper box." \
#     --dataset_replay.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_magetic_tube.1007 \
#     --dataset_replay.num_episodes=50 \
#     --dataset_replay.episode_time_s=12 \
#     --dataset_replay.reset_time_s=5 \
#     --dataset_replay.single_task="Grab the grey metal tube and put it into white paper box."



rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_magetic_tube.15s.1028
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_magetic_tube.15s.1028 \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the grey metal tube by right arm and put it into white paper box."


rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.fold_towel.30s.1029
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.fold_towel.30s.1029 \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="fold towel and place it on the table using both arms."

python -m lerobot.replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.fold_towel.30s.1029 \
    --dataset.episode=0

python -m lerobot.scripts.visualize_dataset \
    --repo-id rrecord.arxx5_bimanual.fold_towel.30s.1029 \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.fold_towel.30s.1029 \
    --episode-index 25

rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_grab_left_attach_tube.30s.1030
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_grab_left_attach_tube.30s.1030 \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=35 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the grey metal tube by right arm, transfer it to left arm, and attach it on white shelf."

python -m lerobot.replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_grab_left_attach_tube.30s.1030 \
    --dataset.episode=20

python -m lerobot.scripts.visualize_dataset \
    --repo-id record.arxx5_bimanual.right_grab_left_attach_tube.30s.1030 \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_grab_left_attach_tube.30s.1030 \
    --episode-index 20


rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.fold_towel.30s.1031
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.fold_towel.30s.1031 \
    --dataset.num_episodes=25 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="fold towel and place it on the table using both arms."

python -m lerobot.replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.fold_towel.30s.1031 \
    --dataset.episode=20

python -m lerobot.scripts.visualize_dataset \
    --repo-id rrecord.arxx5_bimanual.fold_towel.30s.1031 \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.fold_towel.30s.1031 \
    --episode-index 0


rm -rf /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.fold_towel.30s.1101
lerobot-record \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.fold_towel.30s.1101 \
    --dataset.num_episodes=25 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="fold towel and place it on the table using both arms."

python -m lerobot.replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.fold_towel.30s.1101 \
    --dataset.episode=20

# ------------------------------------- #
# teleoperate for piper master and slave
# ------------------------------------- #
bash can_activate.sh can0 1000000

python -m lerobot.teleoperate --robot.type=piper \
    --robot.port=can0 \
    --robot.id=my_awesome_follower_arm_piper \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true


rm -rf /home/xuwenda/.cache/huggingface/lerobot/heyuan1993/record.industrial.s3.v0
lerobot-record \
    --robot.type=piper \
    --robot.port=can0 \
    --robot.id=my_awesome_follower_arm_piper \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record.industrial.s3.v0 \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Grab the grey magnet tube and attach it to another tube on the white holder."

# ---------------------------------------------------------- #
# bipiper pour-water recording with YAML-driven subtasks
#   Requires:
#     - tts_record_cfgs/bipiper_record.yaml (contains POUR_WATER block)
#     - tts_record_cfgs/pour_water_subtask_annotations.jsonl
#   Effect:
#     - per-episode random Chinese/English prompt (TTS announced)
#     - per-frame subtask_index column in parquet
#     - subtask_annotations.jsonl copied into dataset/annotations/
# ---------------------------------------------------------- #
bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can1 1000000 1-5:1.0  # leader left
bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can0 1000000 1-4:1.0  # leader right
bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can2 1000000 1-8:1.0  # follower left
bash /home/heyuan/work/piper_sdk/piper_sdk/can_activate.sh can3 1000000 1-10:1.0 # follower right

python -m lerobot.recording.record \
    --robot.type=bi_piper_follower \
    --mode=record \
    --robot.left_arm_port=can2 \
    --robot.right_arm_port=can3 \
    --robot.id=my_awesome_bi_piper_follower_arm3 \
    --robot.cameras="{head: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, third_view: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}}" \
    --teleop.type=bi_piper_leader \
    --teleop.left_arm_port=can1 \
    --teleop.right_arm_port=can0 \
    --teleop.id=my_awesome_bi_piper_leader_arm3 \
    --display_data=false \
    --dataset.repo_id=${HF_USER}/record.pourwater.bipiper.test \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=55 \
    --dataset.reset_time_s=2 \
    --dataset.single_task="pick water and pour into the container" \
    --subtask_config_path=tts_record_cfgs/bipiper_record.yaml \
    --record_task="POUR_WATER"
