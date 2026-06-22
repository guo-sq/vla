
source $HOME/.local/bin/env
source .venv/bin/activate

# Check Python version
python --version
# Should show Python 3.11.x

# Verify OpenPi installation
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import openpi
    print('✓ OpenPi imported successfully')
except ImportError as e:
    print(f'✗ OpenPi import failed: {e}')
"

# automatic download dataset

# Method 1: Using huggingface-cli
hf download openvla/modified_libero_rlds --local-dir openvla/modified_libero_rlds

# Method 2: Using huggingface_hub library
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='openvla/modified_libero_rlds',
    repo_type='dataset',
    local_dir='/media/xuwenda/4456BE5656BE4886/heyuan/datasets/libero/data',
    local_dir_use_symlinks=False
)
"

pip install --upgrade fsspec aiohttp gcsfs

gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_fast_droid ./checkpoints/
gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_aloha_sim ./checkpoints/
gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_base ./checkpoints/
gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_fast_base ./checkpoints/
gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_fast_libero ./checkpoints/
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base ./checkpoints/
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero ./checkpoints/
gsutil -m cp -r gs://big_vision/paligemma_tokenizer.model ./
uv run gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid ./openpi-assets/checkpoints/


python3 openpi/examples/test_code.py


uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /media/xuwenda/4456BE5656BE4886/heyuan/datasets/libero/data


export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

export HF_ENDPOINT=https://hf-mirror.com
HUGGINGFACE_TOKEN=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
git config --global credential.helper store

# hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
uv run huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential

HF_USER=$(uv run huggingface-cli whoami | head -n 1)
echo $HF_USER

export WANDB_API_KEY=77a6f87ffb15b774adba94a59a7d0687344ff8da
uv run wandb login
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_finetune_right_arm_grab_tube.data_1028.exp1029

hf download CogACT/CogACT-Base --local-dir pretrain/CogACT-Base
hf download openvla/openvla-7b-prismatic  --local-dir pretrain/openvla-7b-prismatic


uv run huggingface-cli download \
  --repo-type dataset \
  OpenGalaxea/Galaxea-Open-World-Dataset \
  --local-dir /mnt/workspace/shared/datasets/OpenGalaxea/Galaxea-Open-World-Dataset

uv run huggingface-cli download \
  --repo-type dataset \
  InternRobotics/InternData-A1 \
  --local-dir /mnt/workspace/shared/datasets/InternRobotics/InternData-A1


uv run huggingface-cli download \
  --repo-type dataset \
  physical-intelligence/aloha_pen_uncap_diverse \
  --local-dir /mnt/workspace/shared/datasets/physical-intelligence/aloha_pen_uncap_diverse

uv run huggingface-cli download \
  --repo-type dataset \
  physical-intelligence/libero \
  --local-dir /mnt/workspace/shared/datasets/physical-intelligence/libero


export HF_ENDPOINT=https://hf-mirror.com
hf download NousResearch/Llama-2-7b-hf 




#-----gwl
#go to .venv environment
source .venv/bin/activate

# norm stat
uv run scripts/compute_norm_stats.py --config-name pi05_gr00t_lerobot_finetune

#train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_gr00t_lerobot_finetune --exp-name=pi05_gr00t_lerobot_finetune_exp1008 --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi_gr00t_lerobot_finetune --exp-name=pi05_gr00t_lerobot_finetune_exp1008 --overwrite


# test pi0_libero
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_finetune_right_arm_grab_tube.data_1028.exp_1029
export WANDB_RUN_NAME=pi05_libero_finetune_right_arm_grab_tube.data_1028.exp_1029
uv run scripts/compute_norm_stats.py --config-name pi05_libero_finetune_right_arm_grab_tube
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_finetune_right_arm_grab_tube --exp-name=${WANDB_RUN_NAME} --overwrite

export WANDB_MODE=online
export WANDB_RUN_ID=pi0_aloha_sim_right_grab_left_attach_tube_30s.data_1030.exp_1030
export WANDB_RUN_NAME=pi0_aloha_sim_right_grab_left_attach_tube_30s.data_1030.exp_1030
uv run scripts/compute_norm_stats.py --config-name pi0_aloha_sim_right_grab_left_attach_tube_30s
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_sim_right_grab_left_attach_tube_30s --exp-name=${WANDB_RUN_NAME} --overwrite

uv run scripts/compute_norm_stats.py --config-name pi0_aloha_ft_fold_towel_1029

# uv run scripts/compute_norm_stats.py --config-name pi05_aloha_ft_fold_towel_1029
uv run scripts/compute_norm_stats.py --config-name pi05_base_ft_fold_towel_1029
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_base_ft_fold_towel_1029  --exp-name=pi05_base_ft_fold_towel_1029_exp1030

# --------------------------------------- # 
# train pi0_aloha_mobile_cabinet on pi0_base
# --------------------------------------- # 
export WANDB_MODE=online
export WANDB_RUN_ID=pi0_aloha_mobile_cabinet
export WANDB_RUN_NAME=pi0_aloha_mobile_cabinet.exp.1019
uv run scripts/compute_norm_stats.py --config-name pi0_aloha_mobile_cabinet
CUDA_VISIBLE_DIVICE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_mobile_cabinet --exp-name=${WANDB_RUN_NAME} --overwrite

# terminal-1: serve policy
export WANDB_RUN_NAME=pi0_aloha_mobile_cabinet.exp.1019
POLICY_CONFIG=pi0_aloha_mobile_cabinet
POLICY_CKPT=${WANDB_RUN_NAME}/19999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

export WANDB_RUN_NAME=pi0_aloha_mobile_cabinet.exp.1019
POLICY_CONFIG=pi0_aloha_mobile_cabinet
POLICY_CKPT=${WANDB_RUN_NAME}/19999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/root/.cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet/
uv run examples/simple_client/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_
# --------------------------------------- # 
# pi0: pi0_aloha_pen_uncap from pi0-base
# --------------------------------------- #
# train pi0_aloha_pen_uncap on pi0_base
export WANDB_MODE=online
export WANDB_RUN_ID=pi0_aloha_pen_uncap
export WANDB_RUN_NAME=pi0_aloha_pen_uncap.exp.1019
uv run scripts/compute_norm_stats.py --config-name pi0_aloha_pen_uncap
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_pen_uncap --exp-name=${WANDB_RUN_NAME} --overwrite

# terminal-1: serve policy
export WANDB_RUN_NAME=pi0_aloha_pen_uncap.exp.1019
POLICY_CONFIG=pi0_aloha_pen_uncap
POLICY_CKPT=${WANDB_RUN_NAME}/19999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

# terminal-2: run dataset environment
export WANDB_RUN_NAME=pi0_aloha_pen_uncap.exp.1019
POLICY_CONFIG=pi0_aloha_pen_uncap
POLICY_CKPT=${WANDB_RUN_NAME}/19999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/root/.cache/huggingface/lerobot/physical-intelligence/aloha_pen_uncap_diverse/
uv run examples/simple_client/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_
# --------------------------------------- # 
# pi0 eval
# --------------------------------------- #
# terminal-1: serve policy
POLICY_CONFIG=pi0_aloha_sim_gr00t_lerobot_finetune
POLICY_CKPT=pi0_aloha_sim_gr00t_lerobot_finetune_1016/29999
POLICY_PATH=/mnt/workspace/gwl/playground/openpi_modified/checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

# terminal-2: run dataset environment
POLICY_PATH=/mnt/workspace/gwl/playground/openpi_modlified/checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/root/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix
uv run examples/simple_client/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_



# --------------------------------------- #
# pi0: train multi datasets
# --------------------------------------- #
# train pi0_aloha_ft_all_anyv on pi0_base
export WANDB_MODE=online
export WANDB_RUN_ID=pi0_aloha_ft_all_anyv
export WANDB_RUN_NAME=pi0_aloha_ft_all_anyv.exp.1031
uv run scripts/compute_norm_stats_multi_datasets.py --config-name pi0_aloha_ft_all_anyv
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_ft_all_anyv --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi0: train fold towel
# --------------------------------------- #
# train pi0_aloha_ft_fold_towel_fix_gripper_1104 on pi0_aloha
export WANDB_MODE=online
export WANDB_RUN_ID=pi0_aloha_ft_fold_towel_fix_gripper_1104
export WANDB_RUN_NAME=pi0_aloha_ft_fold_towel_fix_gripper_1104.exp.1104
# uv run scripts/compute_norm_stats.py --config-name pi0_aloha_ft_fold_towel_fix_gripper_1104
uv run scripts/compute_norm_stats_multi_datasets.py --config-name pi0_aloha_ft_fold_towel_fix_gripper_1104
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_ft_fold_towel_fix_gripper_1104 --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi0: train fold towel
# --------------------------------------- #
# train pi0_aloha_ft_fold_towel_fix_gripper_1105 on pi0_aloha
export WANDB_MODE=online
export WANDB_RUN_ID=pi0_aloha_ft_fold_towel_fix_gripper_1105
export WANDB_RUN_NAME=pi0_aloha_ft_fold_towel_fix_gripper_1105.exp.1105
# uv run scripts/compute_norm_stats.py --config-name pi0_aloha_ft_fold_towel_fix_gripper_1105
uv run scripts/compute_norm_stats_multi_datasets.py --config-name pi0_aloha_ft_fold_towel_fix_gripper_1105
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_ft_fold_towel_fix_gripper_1105 --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi0: train fold towel with error recovery
# --------------------------------------- #
# train pi0_aloha_ft_fold_towel_fix_gripper_recovery on pi0_aloha
export WANDB_MODE=online
export WANDB_RUN_ID=pi0_aloha_ft_fold_towel_fix_gripper_recovery
export WANDB_RUN_NAME=pi0_aloha_ft_fold_towel_fix_gripper_recovery.exp.1106
uv run scripts/compute_norm_stats_multi_datasets.py --config-name pi0_aloha_ft_fold_towel_fix_gripper_recovery
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_ft_fold_towel_fix_gripper_recovery --exp-name=${WANDB_RUN_NAME} --overwrite


# --------------------------------------- #
# pi0: train fold cloth
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi0_aloha_ft_fold_t_shirt
export WANDB_RUN_NAME=pi0_aloha_ft_fold_t_shirt.exp.1107
uv run scripts/compute_norm_stats_multi_datasets.py --config-name pi0_aloha_ft_fold_t_shirt
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_ft_fold_t_shirt --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi0: train fold cloth: fast
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi0_aloha_ft_fold_t_shirt_fast
export WANDB_RUN_NAME=pi0_aloha_ft_fold_t_shirt_fast.exp.1108
uv run scripts/compute_norm_stats_multi_datasets.py --config-name pi0_aloha_ft_fold_t_shirt_fast
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_ft_fold_t_shirt_fast --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi0: pi0_aloha_ft_fold_towel_fix_gripper_random
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi0_aloha_ft_fold_towel_fix_gripper_random
export WANDB_RUN_NAME=pi0_aloha_ft_fold_towel_fix_gripper_random.exp.1108
uv run scripts/compute_norm_stats_multi_datasets.py --config-name pi0_aloha_ft_fold_towel_fix_gripper_random
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_ft_fold_towel_fix_gripper_random --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi0: fold t shirt in two methods
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi0_aloha_ft_fold_t_shirt_mix
export WANDB_RUN_NAME=pi0_aloha_ft_fold_t_shirt_mix.exp.1109
uv run scripts/compute_norm_stats_multi_datasets.py --config-name pi0_aloha_ft_fold_t_shirt_mix
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_ft_fold_t_shirt_mix --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi0: fold t shirt in two methods
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi0_aloha_ft_fold_towel_fix_gripper_mix
export WANDB_RUN_NAME=pi0_aloha_ft_fold_towel_fix_gripper_mix.exp.1109
uv run scripts/compute_norm_stats_multi_datasets.py --config-name pi0_aloha_ft_fold_towel_fix_gripper_mix
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_ft_fold_towel_fix_gripper_mix --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi0: fold all
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi0_aloha_ft_fold_all
export WANDB_RUN_NAME=pi0_aloha_ft_fold_all.exp.1109
uv run scripts/compute_norm_stats_multi_datasets.py --config-name pi0_aloha_ft_fold_all
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_ft_fold_all --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi0: fold towel random all
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi0_aloha_ft_fold_towel_random_selected
export WANDB_RUN_NAME=pi0_aloha_ft_fold_towel_random_selected.exp.1112
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi0_aloha_ft_fold_towel_random_selected
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_ft_fold_towel_random_selected --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_fold_towel_random_selected
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_fold_towel_random_selected.rerun
export WANDB_RUN_NAME=pi05_libero_ft_fold_towel_random_selected.exp.1115
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_fold_towel_random_selected
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_fold_towel_random_selected --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_static_sort
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_static_sort
export WANDB_RUN_NAME=pi05_libero_ft_static_sort.exp.1115
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_static_sort
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_static_sort --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_static_sort_mix_multi_item
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_static_sort_mix_multi_item
export WANDB_RUN_NAME=pi05_libero_ft_static_sort_mix_multi_item.exp.1117
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_static_sort_mix_multi_item
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_static_sort_mix_multi_item --exp-name=${WANDB_RUN_NAME} --overwrite


# --------------------------------------- #
# pi05: pi05_libero_ft_static_sort_mix_multi_item.random
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_static_sort_mix_multi_item_random
export WANDB_RUN_NAME=pi05_libero_ft_static_sort_mix_multi_item_random.exp.1120
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_static_sort_mix_multi_item_random
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_static_sort_mix_multi_item_random --exp-name=${WANDB_RUN_NAME} --overwrite


# --------------------------------------- #
# pi05: pi05_libero_ft_fold_t_shirt_fast_format
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_fold_t_shirt_fast_format
export WANDB_RUN_NAME=pi05_libero_ft_fold_t_shirt_fast_format.exp.1117
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_fold_t_shirt_fast_format
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_fold_t_shirt_fast_format --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_fold_t_shirt_fast_format.mix
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_fold_t_shirt_fast_format.mix
export WANDB_RUN_NAME=pi05_libero_ft_fold_t_shirt_fast_format.mix.exp.1120
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_fold_t_shirt_fast_format.mix
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_fold_t_shirt_fast_format.mix --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_grab_attach_tube
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_grab_attach_tube
export WANDB_RUN_NAME=pi05_libero_ft_grab_attach_tube.exp.1120
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_grab_attach_tube
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_grab_attach_tube --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_grab_attach_tube_reorg_mix_approach
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_grab_attach_tube_reorg_mix_approach
export WANDB_RUN_NAME=pi05_libero_ft_grab_attach_tube_reorg_mix_approach.exp.1122
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_grab_attach_tube_reorg_mix_approach
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_grab_attach_tube_reorg_mix_approach --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab
export WANDB_RUN_NAME=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab.exp.1124
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj
export WANDB_RUN_NAME=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj.exp.1125
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right
export WANDB_RUN_NAME=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right.exp.1126
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_1127
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_1127
export WANDB_RUN_NAME=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_1127.exp.1127
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_1127
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_1127 --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_1129_wpretrain
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_1129_wpretrain
export WANDB_RUN_NAME=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_1129_wpretrain.exp.1128
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_1129_wpretrain
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_1129_wpretrain --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_split_lr
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_split_lr
export WANDB_RUN_NAME=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_split_lr.exp.1202
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_split_lr
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_split_lr --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_split_lr_brief_prompt
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_split_lr_brief_prompt
export WANDB_RUN_NAME=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_split_lr_brief_prompt.exp.1203
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_split_lr_brief_prompt
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_right_split_lr_brief_prompt --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_mix_nov
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_mix_nov
export WANDB_RUN_NAME=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_mix_nov.exp.1203
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_mix_nov
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_mix_nov --exp-name=${WANDB_RUN_NAME} --overwrite


# --------------------------------------- #
# pi05: pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1204
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1204
export WANDB_RUN_NAME=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1204.exp.1204
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1204
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1204 --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207
export WANDB_RUN_NAME=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207.exp.1207
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207 --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_base_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207_delta_action
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_base_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207_delta_action
export WANDB_RUN_NAME=pi05_base_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207_delta_action.exp.1207
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_base_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207_delta_action
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_base_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207_delta_action --exp-name=${WANDB_RUN_NAME} --overwrite

# --------------------------------------- #
# pi05: pi05_base_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207_delta_action
# --------------------------------------- #
export WANDB_MODE=online
export WANDB_RUN_ID=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207_episode_prompt
export WANDB_RUN_NAME=pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207_episode_prompt.exp.1207
uv run scripts/compute_norm_stats_multi_datasetsv3.py --config-name pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207_episode_prompt
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ft_grab_attach_tube_reorg_mix_approach_grab_more_full_traj_brief_prompt_dec_1207_episode_prompt --exp-name=${WANDB_RUN_NAME} --overwrite
