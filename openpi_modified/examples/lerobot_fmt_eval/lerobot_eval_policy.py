from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from datasets import load_dataset


# 1. 配置本地数据集路径
# 替换为你的本地数据集根目录（包含数据文件的文件夹）
dataset_root = "/media/xuwenda/4456BE5656BE4886/heyuan/playground/Isaac-GR00T/demo_data/record.arxx5_bimanual.right_arm_grab_toy_duck"

# 2. 配置元数据（关键步骤）
# repo_id：给本地数据集起一个唯一标识（符合HF规范，不用真实存在于线上）
# root：本地数据集存储路径
# force_cache_sync：设为False避免同步线上数据
# meta = LeRobotDatasetMetadata(
#     repo_id="my_local_robot_dataset1010",  # 自定义名称，必须非空且符合格式
#     root=dataset_root,
#     force_cache_sync=False  # 禁止同步线上数据，仅使用本地文件
# )

# print(meta)
# # 2. 读取具体的Parquet文件（以第一个episode为例）
# episode_index = 0
# parquet_file = meta.get_data_file_path(episode_index)

# print("Parquet 文件路径:", parquet_file)

# # 使用datasets加载parquet文件，
# hf_dataset = load_dataset("parquet", data_files=str(dataset_root / parquet_file), split="train")
# #hf_dataset.set_format("torch")

# # 检查数据字段
# print("字段列表:", hf_dataset.column_names)

# # 打印数据示例 , 时刻的参数
# print(hf_dataset[0])

# 输出全部数据
# for sample in hf_dataset:
#     print(sample)

dataset = LeRobotDataset(
    repo_id="my_local_robot_dataset1010",  # 自定义名称，必须非空且符合格式
    root=dataset_root,
    force_cache_sync=False,
)
print(dataset[0].keys())
