import os

ROOT_DIR = "/mnt/"
sub_path = "oss_data/IPEC-COMMUNITY"

# """
CHECK_REPO_ID = [
    # "austin_buds_dataset_lerobot",
    # "austin_sailor_dataset_lerobot",
    # "austin_sirius_dataset_lerobot",
    # "bc_z_lerobot",
    # "berkeley_autolab_ur5_lerobot",
    # "berkeley_cable_routing_lerobot",
    # "berkeley_fanuc_manipulation_lerobot",
    # "berkeley_mvp_lerobot",
    # "berkeley_rpt_lerobot",
    # "bridge_orig_lerobot",
    # "cmu_play_fusion_lerobot",
    # "cmu_stretch_lerobot",
    # "dlr_edan_shared_control_lerobot",
    # "dobbe_lerobot",
    # "fmb_dataset_lerobot",
    # "fractal20220817_data_lerobot",
    # "furniture_bench_dataset_lerobot",
    # "iamlab_cmu_pickup_insert_lerobot",
    # "jaco_play_lerobot",
    # "kuka_lerobot",
    "language_table_lerobot",
    # "nyu_door_opening_surprising_effectiveness_lerobot",
    # "nyu_franka_play_dataset_lerobot",
    # "roboturk_lerobot",
    # "stanford_hydra_dataset_lerobot",
    # "taco_play_lerobot",
    # "toto_lerobot",
    # "ucsd_kitchen_dataset_lerobot",
    # "utaustin_mutex_lerobot",
    # "viola_lerobot",
]

# tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
# REPO_ID = [os.path.join(sub_path, task) for task in tasks if task in CHECK_REPO_ID]
# """


REPO_ID = [
    os.path.join(sub_path, repo_id)
    for repo_id in os.listdir(os.path.join(ROOT_DIR, sub_path))
    if os.path.isdir(os.path.join(ROOT_DIR, sub_path, repo_id))
    and repo_id in CHECK_REPO_ID
]
print("-----------------------------")
print(REPO_ID)
