import os

REPO_ID = []

ROOT_DIR = "/mnt/"
split = "success_episodes"

sub_path = "oss_data/X-Humanoid/RoboMIND2.0-Ark_lerobot/ark"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]

REPO_ID = [repo_id for repo_id in REPO_ID if os.path.exists(os.path.join(ROOT_DIR, repo_id, "meta/info.json"))]
print("-----------------------------")
print(REPO_ID)
