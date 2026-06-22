
import os

REPO_ID = []

base_path = "/mnt/oss_data/"

# ----------------------------------------------------------------------- #
task = "articulation_tasks"
sub_path = f"InternRobotics/InternData-A1/sim/{task}/franka"
tasks = sorted(os.listdir(os.path.join(base_path, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]

sub_path = f"InternRobotics/InternData-A1/sim/{task}/lift2"
tasks = sorted(os.listdir(os.path.join(base_path, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]

sub_path = f"InternRobotics/InternData-A1/sim/{task}/split_aloha"
tasks = sorted(os.listdir(os.path.join(base_path, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]

# ----------------------------------------------------------------------- #
task = "basic_tasks"
sub_path = f"InternRobotics/InternData-A1/sim/{task}/franka"
tasks = sorted(os.listdir(os.path.join(base_path, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]

sub_path = f"InternRobotics/InternData-A1/sim/{task}/genie1"
tasks = sorted(os.listdir(os.path.join(base_path, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]

sub_path = f"InternRobotics/InternData-A1/sim/{task}/lift2"
tasks = sorted(os.listdir(os.path.join(base_path, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]

sub_path = f"InternRobotics/InternData-A1/sim/{task}/split_aloha"
tasks = sorted(os.listdir(os.path.join(base_path, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]

# ----------------------------------------------------------------------- #
task = "long_horizon_tasks"
sub_path = f"InternRobotics/InternData-A1/sim/{task}/franka"
tasks = sorted(os.listdir(os.path.join(base_path, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]

sub_path = f"InternRobotics/InternData-A1/sim/{task}/genie1"
tasks = sorted(os.listdir(os.path.join(base_path, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]

sub_path = f"InternRobotics/InternData-A1/sim/{task}/lift2"
tasks = sorted(os.listdir(os.path.join(base_path, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]

sub_path = f"InternRobotics/InternData-A1/sim/{task}/split_aloha"
tasks = sorted(os.listdir(os.path.join(base_path, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]

# ----------------------------------------------------------------------- #
task = "pick_and_place_tasks"
sub_path = f"InternRobotics/InternData-A1/sim/{task}/franka"
tasks = sorted(os.listdir(os.path.join(base_path, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]

sub_path = f"InternRobotics/InternData-A1/sim/{task}/lift2"
tasks = sorted(os.listdir(os.path.join(base_path, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]

sub_path = f"InternRobotics/InternData-A1/sim/{task}/split_aloha"
tasks = sorted(os.listdir(os.path.join(base_path, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task != ".cache"]


REPO_ID = [repo_id for repo_id in REPO_ID if os.path.isdir(os.path.join(base_path, repo_id))]
print("-----------------------------")
print(REPO_ID)
