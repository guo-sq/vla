#!/usr/bin/env python3
"""
Auto Rename Prompt Tool

用于批量替换指定文件夹中 meta/episodes.jsonl 和 meta/tasks.jsonl 里的语句。
"""

import argparse
import json
from pathlib import Path
from typing import List


def replace_in_jsonl(file_path: Path, old_text: str, new_text: str) -> int:
    """
    在jsonl文件中替换文本

    Args:
        file_path: jsonl文件路径
        old_text: 要替换的旧文本
        new_text: 替换后的新文本

    Returns:
        替换的行数
    """
    if not file_path.exists():
        print(f"  文件不存在: {file_path}")
        return 0

    modified_count = 0
    new_lines = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    line_modified = False

                    # 递归检查并替换字典中的所有字符串值
                    def replace_in_dict(obj):
                        nonlocal line_modified
                        if isinstance(obj, dict):
                            for key, value in obj.items():
                                if isinstance(value, str) and old_text in value:
                                    obj[key] = value.replace(old_text, new_text)
                                    line_modified = True
                                elif isinstance(value, (dict, list)):
                                    replace_in_dict(value)
                        elif isinstance(obj, list):
                            for i, item in enumerate(obj):
                                if isinstance(item, str) and old_text in item:
                                    obj[i] = item.replace(old_text, new_text)
                                    line_modified = True
                                elif isinstance(item, (dict, list)):
                                    replace_in_dict(item)

                    replace_in_dict(data)
                    new_lines.append(json.dumps(data, ensure_ascii=False))
                    if line_modified:
                        modified_count += 1

                except json.JSONDecodeError as e:
                    print(f"  JSON解析错误: {e}")
                    new_lines.append(line)

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line + '\n')

    return modified_count


def process_folder(folder_path: str, old_text: str, new_text: str) -> dict:
    """
    处理单个文件夹

    Args:
        folder_path: 文件夹路径
        old_text: 要替换的旧文本
        new_text: 替换后的新文本

    Returns:
        处理结果统计
    """
    folder = Path(folder_path)
    if not folder.exists():
        return {"error": f"文件夹不存在: {folder_path}"}

    result = {
        "folder": folder_path,
        "episodes_modified": 0,
        "tasks_modified": 0
    }

    # 处理 meta/episodes.jsonl
    episodes_file = folder / "meta" / "episodes.jsonl"
    print(f"\n处理: {episodes_file}")
    result["episodes_modified"] = replace_in_jsonl(episodes_file, old_text, new_text)

    # 处理 meta/tasks.jsonl
    tasks_file = folder / "meta" / "tasks.jsonl"
    print(f"处理: {tasks_file}")
    result["tasks_modified"] = replace_in_jsonl(tasks_file, old_text, new_text)

    return result


def batch_replace(
    folder_list: List[str],
    old_text: str,
    new_text: str,
):
    """
    批量替换多个文件夹中的文本

    Args:
        folder_list: 文件夹路径列表
        old_text: 要替换的旧文本
        new_text: 替换后的新文本
    """
    print(f"旧文本: {old_text}")
    print(f"新文本: {new_text}")
    print(f"待处理文件夹数: {len(folder_list)}")
    print("=" * 80)

    summary = {
        "total_folders": len(folder_list),
        "episodes_total_modified": 0,
        "tasks_total_modified": 0,
        "failed_folders": []
    }

    for i, folder_path in enumerate(folder_list, 1):
        print(f"\n[{i}/{len(folder_list)}] 处理文件夹: {folder_path}")
        result = process_folder(folder_path, old_text, new_text)

        if "error" in result:
            print(f"  错误: {result['error']}")
            summary["failed_folders"].append(result)
        else:
            print(f"  episodes.jsonl 修改行数: {result['episodes_modified']}")
            print(f"  tasks.jsonl 修改行数: {result['tasks_modified']}")
            summary["episodes_total_modified"] += result["episodes_modified"]
            summary["tasks_total_modified"] += result["tasks_modified"]

    print("\n" + "=" * 80)
    print("处理完成！汇总统计:")
    print(f"  总文件夹数: {summary['total_folders']}")
    print(f"  episodes.jsonl 总修改行数: {summary['episodes_total_modified']}")
    print(f"  tasks.jsonl 总修改行数: {summary['tasks_total_modified']}")
    print(f"  失败文件夹数: {len(summary['failed_folders'])}")

    if summary["failed_folders"]:
        print("\n失败的文件夹:")
        for fail in summary["failed_folders"]:
            print(f"  - {fail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批量替换指定文件夹中 meta/episodes.jsonl 和 meta/tasks.jsonl 里的语句"
    )
    parser.add_argument("-r", "--root-path", required=True, help="数据根路径")
    parser.add_argument("-s", "--sub-datasets", required=True, nargs="+", help="子数据集路径（可多个）")
    parser.add_argument("-o", "--old-text", required=True, help="要替换的旧文本")
    parser.add_argument("-n", "--new-text", required=True, help="替换后的新文本")

    args = parser.parse_args()

    root = Path(args.root_path)
    folder_list = [str(root / sub) for sub in args.sub_datasets]

    batch_replace(folder_list, args.old_text, args.new_text)

'''
python tools/public_dataset/auto_rename_prompt.py \                                                                  
    -r /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/lose_the_flap/total_steps \                                            
    -s fold_box_scratch.all.67s.20260305.batch.1111 \                                                                  
    -o "Assemble the box from scratch and put on the desk" \                                                         
    -n "Close the box and put on the desk"
'''

