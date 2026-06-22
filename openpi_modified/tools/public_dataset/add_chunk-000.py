"""
所有数据集的data下面应该是chunk-000目录,如果是parquet,就创建一个chunk-000目录,然后把parquet移过去
"""

import os
import shutil
from pathlib import Path

repo_list = [
    "/mnt/workspace/shared/datasets/lerobot/aloha_mobile_cabinet",
    "/mnt/workspace/shared/datasets/lerobot/aloha_mobile_chair",
    "/mnt/workspace/shared/datasets/lerobot/aloha_mobile_elevator",
    "/mnt/workspace/shared/datasets/lerobot/aloha_mobile_shrimp",
    "/mnt/workspace/shared/datasets/lerobot/aloha_mobile_wash_pan",
    "/mnt/workspace/shared/datasets/lerobot/aloha_mobile_wipe_wine",
    "/mnt/workspace/shared/datasets/lerobot/aloha_sim_transfer_cube_human",
    "/mnt/workspace/shared/datasets/lerobot/aloha_static_candy",
    "/mnt/workspace/shared/datasets/lerobot/aloha_static_coffee",
    "/mnt/workspace/shared/datasets/lerobot/aloha_static_cups_open",
    "/mnt/workspace/shared/datasets/lerobot/aloha_static_pingpong_test",
    "/mnt/workspace/shared/datasets/lerobot/aloha_static_pro_pencil",
    "/mnt/workspace/shared/datasets/lerobot/aloha_static_screw_driver",
    "/mnt/workspace/shared/datasets/lerobot/aloha_static_tape",
    "/mnt/workspace/shared/datasets/lerobot/aloha_static_towel",
    "/mnt/workspace/shared/datasets/lerobot/aloha_static_vinh_cup",
    "/mnt/workspace/shared/datasets/lerobot/aloha_static_vinh_cup_left",
    "/mnt/workspace/shared/datasets/lerobot/aloha_static_ziploc_slide",
    # "/mnt/workspace/shared/datasets/lerobot/aloha_static_fork_pick_up",
    # "/mnt/workspace/shared/datasets/lerobot/aloha_static_batter",
    # "/mnt/workspace/shared/datasets/lerobot/aloha_static_thread_velcro",
    # "/mnt/workspace/shared/datasets/lerobot/aloha_static_fork_pick_up",
    # "/mnt/workspace/shared/datasets/lerobot/aloha_static_battery",
]


def move_to_chunk_000():
    for repo_path in repo_list:
        data_path = os.path.join(repo_path, "data")

        # 检查数据目录是否存在
        if not os.path.exists(data_path):
            print(f"数据目录不存在: {data_path}")
            continue

        # 获取当前目录下的所有parquet文件
        parquet_files = [f for f in os.listdir(data_path) if f.endswith(".parquet")]

        if not parquet_files:
            print(f"在 {data_path} 中未找到parquet文件")
            continue

        # 创建chunk-000目录
        chunk_dir = os.path.join(data_path, "chunk-000")
        os.makedirs(chunk_dir, exist_ok=True)

        # 移动parquet文件到chunk-000目录
        for parquet_file in parquet_files:
            src_path = os.path.join(data_path, parquet_file)
            dst_path = os.path.join(chunk_dir, parquet_file)

            # 如果目标文件已存在，询问是否覆盖
            if os.path.exists(dst_path):
                response = input(f"文件 {dst_path} 已存在，是否覆盖? (y/n): ")
                if response.lower() != "y":
                    print(f"跳过文件: {parquet_file}")
                    continue

            shutil.move(src_path, dst_path)
            print(f"移动文件: {parquet_file} -> chunk-000/")

        print(f"处理完成: {repo_path}")


def change_parquet_name():
    """
    change_parquet_name,使其按照顺序重命名为episode_{episode_index:06d}.parquet格式
    """
    for repo_path in repo_list:
        data_path = os.path.join(repo_path, "data")
        chunk_dir = os.path.join(data_path, "chunk-000")
        parquet_files = [f for f in os.listdir(chunk_dir) if f.endswith(".parquet")]
        parquet_files = sorted(parquet_files)
        for i, parquet_file in enumerate(parquet_files):
            if not parquet_file.startswith("episode_"):
                new_name = f"episode_{i:06d}.parquet"
                os.rename(
                    os.path.join(chunk_dir, parquet_file),
                    os.path.join(chunk_dir, new_name),
                )
                print(f"重命名文件: {parquet_file} -> {new_name}")
                print(f"处理完成: {repo_path}")


if __name__ == "__main__":
    # move_to_chunk_000()
    change_parquet_name()
