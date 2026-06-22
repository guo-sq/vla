import os

REPO_ID = []

ROOT_DIR = "/mnt/"

sub_path = "oss_data/RoboChallenge/train_data_lerobot/franka_single_arm"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks]

REPO_ID = [repo_id for repo_id in REPO_ID if os.path.exists(os.path.join(ROOT_DIR, repo_id, "meta/info.json"))]
