import os

REPO_ID = []
ROOT_DIR = "/mnt/"  # 根目录保持/mnt不变
sub_path = "oss_data/anyverse_pour_water_record"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
black_list = [
    "place_data",
    "with_cap_data",
    "without_left_camera",
    "cup_contact",
]
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task not in black_list]

# Pour water in shanghai
sub_path = "oss_data/anyverse_pour_water_record_sh"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task not in black_list]
print(REPO_ID)
