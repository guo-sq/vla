import os

REPO_ID = []

ROOT_DIR = "/mnt/"

sub_path = "oss_data/RoboChallenge/train_data_lerobot/aloha_bimanual"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [
    os.path.join(sub_path, task)
    for task in tasks
    if task
    not in [
        "scan_QR_code",
        "put_pen_into_pencil_case",
        "sweep_the_rubbish",
        "stick_tape_to_box",
        "stack_bowls",
        "turn_on_faucet",
        "pour_fries_into_plate",
    ]
]

REPO_ID = [repo_id for repo_id in REPO_ID if os.path.exists(os.path.join(ROOT_DIR, repo_id, "meta/info.json"))]
