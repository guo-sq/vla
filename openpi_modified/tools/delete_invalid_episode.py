import os
import json
import glob
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datasets import load_dataset

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetProcessor:
    """处理数据集的类，用于移除无效的episode并更新相关文件"""

    def __init__(
        self, root: str, repo_id: str, invalid_episode_list: Optional[List[int]] = None
    ):
        """
        初始化数据集处理器

        Args:
            root: 数据集根目录
            repo_id: 仓库ID，用于构建新路径
            invalid_episode_list: 无效的episode索引列表
        """
        self.repo_id = repo_id
        self.new_repo_id = f"new_{repo_id}"
        self.root = Path(root)
        self.invalid_episode_list = invalid_episode_list or []

        # 初始化路径
        self._init_paths()

        # 统计无效长度
        self.invalid_length = 0
        self.frame_num_in_info_jsonl = 0
        self.valid_length = 0
        self.actual_episode_count = 0

        # 检查文件是否存在
        self._check_files_exist()

    def _init_paths(self) -> None:
        """初始化所有文件路径"""
        # 注释文件（可选）
        annotation_dir = self.root / "annotations"
        self.has_annotations = annotation_dir.exists()
        self.frame_sub_task_state_path = annotation_dir / "frame_sub_task_state.jsonl"
        self.frame_sub_task_path = annotation_dir / "frame_sub_task.jsonl"
        self.subtask_annotations_path = annotation_dir / "subtask_annotations.jsonl"

        # Parquet文件
        parquet_dir = self.root / "data" / "chunk-000"
        self.parquet_paths = list(parquet_dir.glob("*"))

        # 元数据文件
        meta_dir = self.root / "meta"
        self.episodes_stats_path = meta_dir / "episodes_stats.jsonl"
        self.episode_path = meta_dir / "episodes.jsonl"
        self.info_path = meta_dir / "info.json"
        self.task_jsonl_path = meta_dir / "tasks.jsonl"

        # 视频文件
        self.video_paths = {}
        video_dir = self.root / "videos" / "chunk-000"
        for video_folder in video_dir.iterdir():
            if video_folder.is_dir():
                mp4_files = list(video_folder.glob("*.mp4"))
                if mp4_files:
                    self.video_paths[video_folder.name] = mp4_files

    def _check_files_exist(self) -> None:
        """检查所有必需的文件是否存在"""
        required_paths = [
            (self.episodes_stats_path, "episodes_stats.jsonl"),
            (self.episode_path, "episodes.jsonl"),
            (self.info_path, "info.json"),
            (self.task_jsonl_path, "tasks.jsonl"),
        ]

        optional_paths = []

        if self.has_annotations:
            optional_paths.extend([
                (self.frame_sub_task_state_path, "frame_sub_task_state.jsonl"),
                (self.frame_sub_task_path, "frame_sub_task.jsonl"),
                (self.subtask_annotations_path, "subtask_annotations.jsonl"),
            ])

        for path, name in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"{name} 不存在: {path}")

        for path, name in optional_paths:
            if not path.exists():
                logger.warning(f"可选文件不存在，将跳过: {name} ({path})")

        # 检查视频文件
        for folder_name, video_files in self.video_paths.items():
            if not video_files:
                logger.warning(f"文件夹 {folder_name} 中没有找到MP4文件")

        # 检查parquet文件
        if not self.parquet_paths:
            logger.warning("未找到parquet文件")

    def _update_episode_file(
        self, input_file: Path, record_length: bool = False
    ) -> Path:
        """
        更新episode相关的JSONL文件，移除无效的episode并重新索引

        Args:
            input_file: 输入文件路径
            record_length: 是否记录无效episode的长度

        Returns:
            新文件路径
        """
        modified_lines = []

        with open(input_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    episode_idx = data.get("episode_index", idx)

                    # 跳过无效的episode
                    if episode_idx in self.invalid_episode_list:
                        if record_length:
                            self.invalid_length += data.get("length", 0)
                        continue

                    modified_lines.append(data)
                except json.JSONDecodeError as e:
                    logger.error(f"第 {idx+1} 行JSON解析错误: {e}")

        # 重新索引episode
        for i, data in enumerate(modified_lines):
            data["episode_index"] = i

        # 创建新路径并保存
        new_path = self._get_new_path(input_file)
        new_path.parent.mkdir(parents=True, exist_ok=True)

        with open(new_path, "w", encoding="utf-8") as f:
            for data in modified_lines:
                f.write(json.dumps(data) + "\n")

        logger.info(f"已更新文件: {new_path}")
        return new_path

    def _get_new_path(self, original_path: Path) -> Path:
        """获取新文件路径"""
        # 将路径中的repo_id替换为new_repo_id
        path_str = str(original_path)
        new_path_str = path_str.replace(self.repo_id, self.new_repo_id)
        return Path(new_path_str)

    def _copy_file_with_new_name(
        self, original_path: Path, episode_num: int, new_episode_num: int
    ) -> Path:
        """
        复制文件并更新文件名中的episode编号

        Args:
            original_path: 原始文件路径
            episode_num: 原始episode编号
            new_episode_num: 新的episode编号

        Returns:
            新文件路径
        """
        new_path = self._get_new_path(original_path)

        # 替换文件名中的episode编号
        old_num_str = str(episode_num).zfill(6)
        new_num_str = str(new_episode_num).zfill(6)

        new_path_str = str(new_path).replace(old_num_str, new_num_str)
        new_path = Path(new_path_str)

        # 创建目录并复制文件
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_path, new_path)

        return new_path

    def update_frame_sub_task_state(self) -> None:
        """更新frame_sub_task_state.jsonl文件"""
        logger.info("更新 frame_sub_task_state.jsonl...")
        self._update_episode_file(self.frame_sub_task_state_path)

    def update_frame_sub_task(self) -> None:
        """更新frame_sub_task.jsonl文件"""
        logger.info("更新 frame_sub_task.jsonl...")
        self._update_episode_file(self.frame_sub_task_path)

    def update_subtask_annotations(self) -> None:
        """复制subtask_annotations.jsonl文件"""
        logger.info("复制 subtask_annotations.jsonl...")
        new_path = self._get_new_path(self.subtask_annotations_path)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.subtask_annotations_path, new_path)

    def update_parquet_files(self) -> None:
        """更新parquet文件"""
        logger.info("更新 parquet 文件...")

        # 过滤有效的parquet文件
        valid_parquet_files = []
        for parquet_file in self.parquet_paths:
            try:
                # 从文件名中提取episode编号
                file_name = parquet_file.stem
                parts = file_name.split("_")
                episode_num = int(parts[-1]) if parts[-1].isdigit() else -1

                if episode_num not in self.invalid_episode_list:
                    valid_parquet_files.append((parquet_file, episode_num))
            except (ValueError, IndexError):
                logger.warning(f"无法从文件名解析episode编号: {parquet_file.name}")
                valid_parquet_files.append((parquet_file, -1))

        # 按episode编号排序
        valid_parquet_files.sort(key=lambda x: x[1])

        # 复制并重命名文件
        frame_offset = 0
        for new_idx, (parquet_file, episode_num) in enumerate(valid_parquet_files):
            ds = load_dataset("parquet", data_files=str(parquet_file), split="train")
            df = ds.to_pandas()
            df["episode_index"] = new_idx
            df["index"] = range(frame_offset, frame_offset + len(df))
            frame_offset += len(df)
            new_path = self._get_new_path(parquet_file)

            # 替换文件名中的episode编号
            old_num_str = str(episode_num).zfill(6)
            new_num_str = str(new_idx).zfill(6)

            new_path_str = str(new_path).replace(old_num_str, new_num_str)
            new_path = Path(new_path_str)

            # 创建目录并复制文件
            new_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(new_path_str, engine="pyarrow", index=False)

        self.actual_episode_count = len(valid_parquet_files)

    def update_episodes_stats(self) -> None:
        """更新episodes_stats.jsonl文件"""
        logger.info("更新 episodes_stats.jsonl...")
        self._update_episode_file(self.episodes_stats_path)

    def update_episodes(self) -> None:
        """更新episodes.jsonl文件"""
        logger.info("更新 episodes.jsonl...")
        self._update_episode_file(self.episode_path, record_length=True)

    def update_info(self) -> None:
        """更新info.json文件，基于实际输出文件计算而非原值减法"""
        logger.info("更新 info.json...")

        with open(self.info_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 基于实际输出的 parquet 文件计算
        new_path = self._get_new_path(self.info_path)
        new_parquet_dir = new_path.parent.parent / "data" / "chunk-000"
        actual_parquet_files = sorted(new_parquet_dir.glob("episode_*.parquet"))

        data["total_episodes"] = len(actual_parquet_files)

        # 计算总帧数：读取所有新 parquet 文件的行数
        import pandas as pd
        total_frames = 0
        for pf in actual_parquet_files:
            df = pd.read_parquet(pf, columns=["index"])
            total_frames += len(df)
        data["total_frames"] = total_frames

        # 更新训练分割
        if "splits" in data and "train" in data["splits"]:
            data["splits"]["train"] = f"0:{data['total_episodes']}"

        # 保存到新路径
        new_path.parent.mkdir(parents=True, exist_ok=True)

        with open(new_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"已更新info.json: {new_path}")

    def update_tasks(self) -> None:
        """复制tasks.jsonl文件"""
        logger.info("复制 tasks.jsonl...")
        new_path = self._get_new_path(self.task_jsonl_path)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.task_jsonl_path, new_path)

    def update_videos(self) -> None:
        """更新视频文件"""
        logger.info("更新视频文件...")

        for folder_name, video_files in self.video_paths.items():
            # 过滤有效的视频文件
            valid_videos = []
            for video_file in video_files:
                try:
                    # 从文件名中提取episode编号
                    file_name = video_file.stem
                    parts = file_name.split("_")
                    episode_num = int(parts[-1]) if parts[-1].isdigit() else -1

                    if episode_num not in self.invalid_episode_list:
                        valid_videos.append((video_file, episode_num))
                except (ValueError, IndexError):
                    logger.warning(f"无法从文件名解析episode编号: {video_file.name}")
                    valid_videos.append((video_file, -1))

            # 按episode编号排序
            valid_videos.sort(key=lambda x: x[1])

            # 复制并重命名文件
            for new_idx, (video_file, episode_num) in enumerate(valid_videos):
                new_file = self._copy_file_with_new_name(
                    video_file, episode_num, new_idx
                )
                logger.debug(f"复制视频: {video_file.name} -> {new_file.name}")

    def process_all(self) -> None:
        """处理所有文件"""
        logger.info(f"开始处理数据集: {self.root}")
        logger.info(f"无效episode列表: {self.invalid_episode_list}")

        try:
            if self.has_annotations:
                self.update_frame_sub_task_state()
                self.update_frame_sub_task()
                self.update_subtask_annotations()
            else:
                logger.info("跳过 annotations 更新（目录不存在）")
            self.update_parquet_files()
            self.update_episodes_stats()
            self.update_episodes()
            self.update_info()
            self.update_tasks()
            self.update_videos()

            logger.info(
                f"处理完成! 新数据集保存在: {self.root.parent / self.new_repo_id}"
            )
        except Exception as e:
            logger.error(f"处理过程中发生错误: {e}")
            raise

    def get_all_frame_num(self):
        with open(self.info_path, "r") as f:
            # json文件，直接load
            data = json.load(f)
        self.frame_num_in_info_jsonl = data["total_frames"]

    def get_all_episode_num(self):
        with open(self.episode_path, "r") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                self.valid_length += data["length"]


def main():
    """主函数"""
    # 定义要处理的数据集和无效的episode列表
    repo_dict = {
        "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/close_the_flap/rein21-24/close_the_flap.rein21-24.usb_typec_new.7s.20260401.batch.2": [0],
        "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/close_the_flap/rein21-24/close_the_flap.rein21-24.usb_typec_new.7s.20260401.batch.5": [3],
    }

    for root, invalid_episode_list in repo_dict.items():
        repo_id = Path(root).name
        logger.info(f"处理 {root} (repo_id: {repo_id})")

        try:
            processor = DatasetProcessor(root, repo_id, invalid_episode_list)
            processor.process_all()
        except Exception as e:
            logger.error(f"处理 {root} 时失败: {e}")
            continue


if __name__ == "__main__":
    """
    功能说明:
    1. 删除无效的episode，并更新所有相关文件
    2. 保存到新文件夹下（new_开头），人工检查无问题后，可替代源文件夹
    """
    main()
