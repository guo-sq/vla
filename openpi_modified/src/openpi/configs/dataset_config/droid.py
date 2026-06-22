import os

ROOT_DIR = "/mnt/"
REPO_ID = [
    "workspace/heyuan/openpi_modified/datasets/cadene/droid_1.0.1",
]

REPO_ID = [
    repo_id for repo_id in REPO_ID if os.path.isdir(os.path.join(ROOT_DIR, repo_id))
]
print("-----------------------------")
print(REPO_ID)