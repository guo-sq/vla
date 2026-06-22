from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from datasets import load_dataset


# 1. 配置本地数据集路径
# 替换为你的本地数据集根目录（包含数据文件的文件夹）
# dataset_root = "/media/xuwenda/4456BE5656BE4886/heyuan/playground/Isaac-GR00T/demo_data/record.arxx5_bimanual.right_arm_grab_toy_duck"
dataset_root = "/home/anyverse/playground/up_to_data/openpi/data/record.arxx5_bimanual.right_arm_grab_duck.1008.fix"
dataset = LeRobotDataset(
    repo_id="my_local_robot_dataset1010",  # 自定义名称，必须非空且符合格式
    root=dataset_root,
    force_cache_sync=False,
)

meta = LeRobotDatasetMetadata(
    repo_id="my_local_robot_dataset1010",  # 自定义名称，必须非空且符合格式
    root=dataset_root,
    force_cache_sync=False,
)
print(dataset[0].keys())
# print(dataset[0])
for key in dataset[0].keys():
    value = dataset[0][key]
    if type(value) == type("123"):
        print(f"{key} : {value}")
    else:
        print(f"key:{key} value:{value.shape}")


print("-----meta-----")
print(meta)

print(f"data len:{len(dataset)}")
# print(dataset[0])
