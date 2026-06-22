import os

REPO_ID = []

ROOT_DIR = "/mnt/"

split = "train"
sub_path = "oss_data/x-humanoid-robomind/RoboMIND/benchmark1_0_lerobot/h5_agilex_3rgb"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]

sub_path = "oss_data/x-humanoid-robomind/RoboMIND/benchmark1_0_lerobot/h5_franka_1rgb"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
# parquet number超出chunk-size
tasks = [task for task in tasks if task not in ["bread_in_basket", "bread_on_table"]]
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]

sub_path = "oss_data/x-humanoid-robomind/RoboMIND/benchmark1_0_lerobot/h5_franka_3rgb"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]

# sub_path = "oss_data/x-humanoid-robomind/RoboMIND/benchmark1_0_lerobot/h5_simulation"
# tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
# REPO_ID += [os.path.join(sub_path, task) for task in tasks]

sub_path = (
    "oss_data/x-humanoid-robomind/RoboMIND/benchmark1_0_lerobot/h5_tienkung_gello_1rgb"
)
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]

sub_path = (
    "oss_data/x-humanoid-robomind/RoboMIND/benchmark1_0_lerobot/h5_tienkung_xsens_1rgb"
)
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]


REPO_ID = [
    repo_id for repo_id in REPO_ID if os.path.exists(os.path.join(ROOT_DIR, repo_id))
]
print("-----------------------------")
print(REPO_ID)
