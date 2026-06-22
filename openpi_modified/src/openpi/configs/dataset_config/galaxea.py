import os

REPO_ID = []

ROOT_DIR = "/mnt/"

sub_path = "oss_data/OpenGalaxea/Galaxea-Open-World-Dataset/lerobot_unzip"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [
    os.path.join(sub_path, task)
    for task in tasks
    if task != ".cache"
    and task not in ["Boil_The_Water_20250714_006"]  # 该数据时间戳突变为0
]

REPO_ID = [
    repo_id for repo_id in REPO_ID if os.path.isdir(os.path.join(ROOT_DIR, repo_id))
]
print("-----------------------------")
print(REPO_ID)
