
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


python3 openpi/examples/test_code.py


uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /media/xuwenda/4456BE5656BE4886/heyuan/datasets/libero/data


export HF_ENDPOINT=https://hf-mirror.com

export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

export HF_ENDPOINT=https://hf-mirror.com
HUGGINGFACE_TOKEN=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
# uv run hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
uv run huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential

HF_USER=$(uv run huggingface-cli whoami | head -n 1)
echo $HF_USER

export WANDB_API_KEY=77a6f87ffb15b774adba94a59a7d0687344ff8da
uv run wandb login
export WANDB_MODE=offline
export WANDB_RUN_ID=wuexkizn

hf download CogACT/CogACT-Base --local-dir pretrain/CogACT-Base
hf download openvla/openvla-7b-prismatic  --local-dir pretrain/openvla-7b-prismatic

export HF_ENDPOINT=https://hf-mirror.com
hf download NousResearch/Llama-2-7b-hf 




#-----gwl
#go to .venv environment
source .venv/bin/activate

# norm stat
uv run scripts/compute_norm_stats.py --config-name pi05_gr00t_lerobot_finetune

#train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_gr00t_lerobot_finetune --exp-name=pi05_gr00t_lerobot_finetune_exp1008 --overwrite

# openloop eval

# --------------------------------------- # 
# pi0.5
# --------------------------------------- #
# terminal-1: serve policy
POLICY_PATH=checkpoints/pi05_gr00t_lerobot_finetune/pi05_gr00t_lerobot_finetune_exp1008_t1013_liberomodel/2000
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_gr00t_lerobot_finetune \
  --policy.dir=${POLICY_PATH}
# terminal-2: run dataset environment
uv run examples/arxx5_bimanual/main_eval_gr00t_lerobot_policy.py --env=LIBERO

# --------------------------------------- # 
# real robot test: pi0-fast
# --------------------------------------- #
# terminal-1: serve policy
POLICY_CONFIG=pi0_fast_gr00t_lerobot_finetune
POLICY_CKPT=pi0_fast_gr00t_lerobot_finetune_exp1008_t1013/13700
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}
# terminal-2: run dataset environment
uv run examples/arxx5_bimanual/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
    --save_path=${POLICY_PATH}_

# --------------------------------------- # 
# pi0_aloha_sim_gr00t_lerobot_finetune eval
# --------------------------------------- #
# terminal-1: serve policy
POLICY_CONFIG=pi0_aloha_sim_gr00t_lerobot_finetune
POLICY_CKPT=pi0_aloha_sim_gr00t_lerobot_finetune_1016/80000
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

# terminal-2: offboard eval run dataset environment
POLICY_CONFIG=pi0_aloha_sim_gr00t_lerobot_finetune
POLICY_CKPT=pi0_aloha_sim_gr00t_lerobot_finetune_1016/80000
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix
uv run examples/arxx5_bimanual/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_8w_

# terminal-3: onboard eval run dataset environment
python examples/arxx5_bimanual/eval_arxone_gr00t_wise.py \
    --robot.type=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --robot.id=arxx5_bimanual \
    --policy_host="localhost" \
    --lang_instruction="Grab the toy duck and pub it into white paper box."

lerobot-replay \
    --robot.type=arxx5_bimanual \
    --robot.id=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 28, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix \
    --dataset.episode=0

# --------------------------------------- # 
# pi05_libero_gr00t_lerobot_finetune
# --------------------------------------- #
# terminal-1: serve policy
POLICY_CONFIG=pi05_libero_gr00t_lerobot_finetune
POLICY_CKPT=pi05_libero_gr00t_lerobot_ft_base_1016/59999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

# terminal-2: offboard eval run dataset environment
POLICY_CONFIG=pi05_libero_gr00t_lerobot_finetune
POLICY_CKPT=pi05_libero_gr00t_lerobot_ft_base_1016/59999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_duck.1008.fix
uv run examples/arxx5_bimanual/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_6w_

python examples/arxx5_bimanual/eval_arxone_gr00t_wise.py \
    --robot.type=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --robot.id=arxx5_bimanual \
    --policy_host="localhost" \
    --lang_instruction="Grab the toy duck and put it into white paper box."
# --------------------------------------- # 
# pi0_aloha_sim_lerobot_ft_12s_1007
# --------------------------------------- #
# terminal-1: serve policy
POLICY_CONFIG=pi0_aloha_sim_lerobot_ft_12s_1007
POLICY_CKPT=pi0_aloha_sim_lerobot_ft_short_12s_1007_1021/7999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

# terminal-2: offboard eval run dataset environment
POLICY_CONFIG=pi0_aloha_sim_lerobot_ft_12s_1007
POLICY_CKPT=pi0_aloha_sim_lerobot_ft_short_12s_1007_1021/7999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.duck_short_12s.1007
uv run examples/arxx5_bimanual/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_2w_ac10

# terminal-3: onboard eval run dataset environment
python examples/arxx5_bimanual/eval_arxone_gr00t_wise.py \
    --robot.type=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --robot.id=arxx5_bimanual \
    --policy_host="localhost" \
    --lang_instruction="Grab the toy duck and pub it into white paper box."


python -m lerobot.scripts.visualize_dataset \
    --repo-id heyuan1993/record.arxx5_bimanual.duck_short_12s.1007 \
    --root  /home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.duck_short_12s.1007 \
    --episode-index 2


# --------------------------------------- # 
# pi0_aloha_sim_lerobot_ft_30s_1007
# --------------------------------------- #
# terminal-1: serve policy
POLICY_CONFIG=pi0_aloha_sim_lerobot_ft_30s_1007
POLICY_CKPT=pi0_aloha_sim_lerobot_ft_long_30s_1007_1020/99999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

# terminal-2: offboard eval run dataset environment
POLICY_CONFIG=pi0_aloha_sim_lerobot_ft_30s_1007
POLICY_CKPT=pi0_aloha_sim_lerobot_ft_long_30s_1007_1020/99999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.duck_long_30s.1007
uv run examples/arxx5_bimanual/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_10w_ac40


# terminal-3: onboard eval run dataset environment
python examples/arxx5_bimanual/eval_arxone_gr00t_wise.py \
    --robot.type=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --robot.id=arxx5_bimanual \
    --policy_host="localhost" \
    --lang_instruction="Grab the toy ducks and pub them into white paper box."

# --------------------------------------- # 
# pi0_aloha_sim_lerobot_ft_grab_magetic_tube_1007
# --------------------------------------- #
# terminal-1: serve policy
POLICY_CONFIG=pi0_aloha_sim_lerobot_ft_grab_magetic_tube_1007
POLICY_CKPT=pi0_aloha_sim_lerobot_ft_grab_magetic_tube_1007_1021/7999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

# terminal-2: offboard eval run dataset environment
POLICY_CONFIG=pi0_aloha_sim_lerobot_ft_grab_magetic_tube_1007
POLICY_CKPT=pi0_aloha_sim_lerobot_ft_grab_magetic_tube_1007_1021/7999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_magetic_tube.1007
uv run examples/arxx5_bimanual/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_8k_ac40

# terminal-3: onboard eval run dataset environment
python examples/arxx5_bimanual/eval_arxone_gr00t_wise.py \
    --robot.type=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 28, width: 640, height: 480, fps: 30}}" \
    --robot.id=arxx5_bimanual \
    --policy_host="localhost" \
    --lang_instruction="Grab the grey metal tube and put it into white paper box."

# --------------------------------------- # 
# pi0_aloha_sim_lerobot_ft_grab_attach_magetic_tube_1007
# --------------------------------------- #
# terminal-1: serve policy
POLICY_CONFIG=pi0_aloha_sim_lerobot_ft_grab_attach_magetic_tube_1007
POLICY_CKPT=pi0_aloha_sim_lerobot_ft_grab_attach_magetic_tube_1007_1021/59999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

# terminal-2: offboard eval run dataset environment
POLICY_CONFIG=pi0_aloha_sim_lerobot_ft_grab_attach_magetic_tube_1007
POLICY_CKPT=pi0_aloha_sim_lerobot_ft_grab_attach_magetic_tube_1007_1021/59999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_and_attach_magetic_tube.1007
uv run examples/arxx5_bimanual/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_6w_ac40

# terminal-3: onboard eval run dataset environment
python examples/arxx5_bimanual/eval_arxone_gr00t_wise.py \
    --robot.type=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --robot.id=arxx5_bimanual \
    --policy_host="localhost" \
    --lang_instruction="Grab the grey metal tube and attach it on white shelf."

# --------------------------------------- # 
# pi0_aloha_sim_lerobot_ft_1022
# --------------------------------------- #
source .venv/bin/activate
lerobot-find-cameras opencv

# terminal-1: serve policy
POLICY_CONFIG=pi0_aloha_sim_lerobot_ft_1022
POLICY_CKPT=pi0_aloha_sim_lerobot_ft_1022_1022/149999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

# terminal-2: offboard eval run dataset environment
POLICY_CONFIG=pi0_aloha_sim_lerobot_ft_1022
POLICY_CKPT=pi0_aloha_sim_lerobot_ft_1022_1022/149999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck.fix_position.1022
uv run examples/arxx5_bimanual/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_150k_ac40

# terminal-3: onboard eval run dataset environment
python examples/arxx5_bimanual/eval_arxone_gr00t_wise.py \
    --robot.type=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --robot.id=arxx5_bimanual \
    --policy_host="localhost" \
    --lang_instruction="Grab the toy duck and put it into white paper box."



# --------------------------------------- # 
# pi05_libero_gr00t_lerobot_ft_1022
# --------------------------------------- #
# terminal-1: serve policy
POLICY_CONFIG=pi05_libero_gr00t_lerobot_ft_1022
POLICY_CKPT=pi05_libero_gr00t_lerobot_ft_1022_1022/120000
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

# terminal-2: offboard eval run dataset environment
POLICY_CONFIG=pi05_libero_gr00t_lerobot_ft_1022
POLICY_CKPT=pi05_libero_gr00t_lerobot_ft_1022_1022/120000
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.format.right_arm_grab_duck.fix_position.1022
uv run examples/arxx5_bimanual/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_21w_ac40

# terminal-3: onboard eval run dataset environment
python examples/arxx5_bimanual/eval_arxone_gr00t_wise.py \
    --robot.type=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --robot.id=arxx5_bimanual \
    --policy_host="localhost" \
    --lang_instruction="Grab the toy duck and put it into white paper box."


# --------------------------------------- # 
# pi05_libero_finetune_right_arm_grab_tube
# --------------------------------------- #
# terminal-1: serve policy
POLICY_CONFIG=pi05_libero_finetune_right_arm_grab_tube
POLICY_CKPT=pi05_libero_finetune_right_arm_grab_tube.data_1028.exp_1029/59999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

# terminal-2: offboard eval run dataset environment
POLICY_CONFIG=pi05_libero_finetune_right_arm_grab_tube
POLICY_CKPT=pi05_libero_finetune_right_arm_grab_tube.data_1028.exp_1029/59999
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_arm_grab_magetic_tube.15s.1028
uv run examples/arxx5_bimanual/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_6w_ac40

# terminal-3: onboard eval run dataset environment
python examples/arxx5_bimanual/eval_arxone_gr00t_wise.py \
    --robot.type=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --robot.id=arxx5_bimanual \
    --policy_host="localhost" \
    --lang_instruction="Grab the grey metal tube by right arm and put it into white paper box."


# --------------------------------------- # 
# pi0_aloha_sim_right_grab_left_attach_tube_30s
# --------------------------------------- #
# terminal-1: serve policy
POLICY_CONFIG=pi0_aloha_sim_right_grab_left_attach_tube_30s
POLICY_CKPT=pi0_aloha_sim_right_grab_left_attach_tube_30s.data_1030.exp_1030/30000
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

# terminal-2: offboard eval run dataset environment
POLICY_CONFIG=pi0_aloha_sim_right_grab_left_attach_tube_30s
POLICY_CKPT=pi0_aloha_sim_right_grab_left_attach_tube_30s.data_1030.exp_1030/30000
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.right_grab_left_attach_tube.30s.1030
uv run examples/arxx5_bimanual/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_3w_ac40

# terminal-3: onboard eval run dataset environment
python examples/arxx5_bimanual/eval_arxone_gr00t_wise.py \
    --robot.type=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --robot.id=arxx5_bimanual \
    --policy_host="localhost" \
    --lang_instruction="Grab the grey metal tube by right arm, transfer it to left arm, and attach it on white shelf."


# --------------------------------------- # 
# pi0_aloha_ft_fold_towel_1029
# --------------------------------------- #
# terminal-1: serve policy
POLICY_CONFIG=pi0_aloha_ft_fold_towel_1029
POLICY_CKPT=pi0_aloha_ft_fold_towel_1029_exp1030.fix_dataset/30000
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

# terminal-2: offboard eval run dataset environment
POLICY_CONFIG=pi0_aloha_ft_fold_towel_1029
POLICY_CKPT=pi0_aloha_ft_fold_towel_1029_exp1030.fix_dataset/30000
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.fold_towel.30s.1029
uv run examples/arxx5_bimanual/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_3w_ac40

# terminal-3: onboard eval run dataset environment
python examples/arxx5_bimanual/eval_arxone_gr00t_wise.py \
    --robot.type=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --robot.id=arxx5_bimanual \
    --policy_host="localhost" \
    --lang_instruction="fold towel and place it on the table using both arms."


# --------------------------------------- # 
# pi05_aloha_ft_fold_towel_1029
# --------------------------------------- #
# terminal-1: serve policy
POLICY_CONFIG=pi05_aloha_ft_fold_towel_1029
POLICY_CKPT=pi05_libero_ft_fold_towel_1029_exp1031/30000
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=${POLICY_CONFIG} \
  --policy.dir=${POLICY_PATH}

# terminal-2: offboard eval run dataset environment
POLICY_CONFIG=pi05_libero_ft_fold_towel_1029
POLICY_CKPT=pi05_libero_ft_fold_towel_1029_exp1031/30000
POLICY_PATH=checkpoints/${POLICY_CONFIG}/${POLICY_CKPT}
DATASET_ROOT=/home/anyverse/.cache/huggingface/lerobot/heyuan1993/record.arxx5_bimanual.fold_towel.30s.1029
uv run examples/arxx5_bimanual/main_eval_gr00t_lerobot_policy.py --env=LIBERO \
  --dataset_root=${DATASET_ROOT} \
  --save_path=${POLICY_PATH}_3w_ac40

# terminal-3: onboard eval run dataset environment
python examples/arxx5_bimanual/eval_arxone_gr00t_wise.py \
    --robot.type=arxx5_bimanual \
    --robot.cameras="{head: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --robot.id=arxx5_bimanual \
    --policy_host="localhost" \
    --lang_instruction="fold towel and place it on the table using both arms."

