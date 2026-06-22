import os

ROOT_DIR = "/mnt"
# ROOT_DIR = "/mnt/"
REPO_ID = [
    "oss_data/IPEC-COMMUNITY/bridge_orig_lerobot",
]

REPO_ID = [
    repo_id for repo_id in REPO_ID if os.path.isdir(os.path.join(ROOT_DIR, repo_id))
]
print("-----------------------------")
print(REPO_ID)

