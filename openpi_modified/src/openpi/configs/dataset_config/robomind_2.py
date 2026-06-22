import os

REPO_ID = []

ROOT_DIR = "/mnt/"
split = "success_episodes"

# sub_path = "oss_data/X-Humanoid/RoboMIND2.0-Agilex_lerobot/agilex"
# tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
# REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]


# sub_path = "oss_data/X-Humanoid/RoboMIND2.0-Agilex-mobile/agilex_mobile"
# tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
# REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]

sub_path = "oss_data/X-Humanoid/RoboMIND2.0-Ark_lerobot/ark"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]

sub_path = "oss_data/X-Humanoid/RoboMIND2.0-Ark-mobile_lerobot/ark_mobile"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]


# sync
sub_path = "oss_data/X-Humanoid/RoboMIND2.0-UR5_lerobot/ur"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]

sub_path = "oss_data/X-Humanoid/RoboMIND2.0-UR5-Dex_lerobot/ur_dex"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]

sub_path = "oss_data/X-Humanoid/RoboMIND2.0-Franka-Part-1_lerobot/franka"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]


sub_path = "oss_data/X-Humanoid/RoboMIND2.0-Franka-Part-2_lerobot/franka"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]


sub_path = "oss_data/X-Humanoid/RoboMIND2.0-Franka-Part-3_lerobot/franka"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]


sub_path = "oss_data/X-Humanoid/RoboMIND2.0-Franka-Part-4_lerobot/franka"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]


sub_path = "oss_data/X-Humanoid/RoboMIND2.0-Franka-Part-5_lerobot/franka"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]

sub_path = "oss_data/X-Humanoid/RoboMIND2.0-Tienkung_lerobot/tienkung"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [
    os.path.join(sub_path, task, split)
    for task in tasks
    if task
    not in [
        "flip_black_knob_to_close_circuit",  # 26
        "flip_yellow_button_and_stand_it_with_both_arms",  # 14
        "insert_blue_part_into_gray_base",  # 14
        "insert_purple_and_pink_pens_in_holder",  # 14
        "open_door_with_both_hands",  # 26
        "open_oven",  # 14
        "organize_books_on_shelf",  # NONE
        "place_button_on_rack_with_both_arms",  # 14
        "pour_wine_from_measuring_cup_to_shaker",
    ]
]


sub_path = "oss_data/X-Humanoid/RoboMIND2.0-Tianyi_lerobot/tienyi"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]

sub_path = "oss_data/X-Humanoid/RoboMIND2.0-Tianyi-mobile_lerobot/tienyi_mobile"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task, split) for task in tasks]


REPO_ID = [repo_id for repo_id in REPO_ID if os.path.exists(os.path.join(ROOT_DIR, repo_id, "meta/info.json"))]
print("-----------------------------")
print(REPO_ID)
