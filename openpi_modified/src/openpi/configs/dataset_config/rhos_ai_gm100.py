import os


ROOT_DIR = "/mnt/"
sub_path = "oss_data/rhos-ai/gm100-cobotmagic-lerobot/"

REPO_ID = []

tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]

REPO_ID = [
    repo_id for repo_id in REPO_ID if os.path.isdir(os.path.join(ROOT_DIR, repo_id))
]
print("-----------------------------")
print(REPO_ID)
