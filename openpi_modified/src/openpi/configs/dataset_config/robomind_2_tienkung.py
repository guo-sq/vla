import os

REPO_ID = []

ROOT_DIR = "/mnt/"
split = "success_episodes"

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


REPO_ID = [repo_id for repo_id in REPO_ID if os.path.exists(os.path.join(ROOT_DIR, repo_id, "meta/info.json"))]
print("-----------------------------")
print(REPO_ID)
