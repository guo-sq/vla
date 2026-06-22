import os

ROOT_DIR = "/mnt/workspace/heyuan/openpi_modified/datasets"
REPO_ID = [
    "IPEC-COMMUNITY/kuka_lerobot",
]

REPO_ID = [
    repo_id for repo_id in REPO_ID if os.path.isdir(os.path.join(ROOT_DIR, repo_id))
]
print("-----------------------------")
print(REPO_ID)
