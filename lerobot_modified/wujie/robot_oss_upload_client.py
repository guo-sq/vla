#!/usr/bin/env python3
# coding=utf-8
#
# Copyright @2025 Wujiedongli Technology Inc. All rights reserved.
# Authors: Yawei Wang <yawei.wang@anyverseintelligence.com>

import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path
from tkinter import (END, Button, DoubleVar, Entry, Frame, Label, Listbox,
                     Scrollbar, StringVar, Tk, filedialog, messagebox, ttk)

from lerobot_data_check import run_checks
from oss_uploader import REMOTE_UPLOAD_DIR, OSSUploader

# 日志配置
logger = logging.getLogger("RobotOSSUploadClient")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

HISTORY_FILE = "upload_history.json"


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


def format_transfer_size(num_bytes):
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024


def move_dataset_to_backup(dataset_path: Path, root_task_path: str):
    """
    Move the dataset (folder containing meta/) to ~/.daq_upload/oss/
    Preserving hierarchy relative to root_task_path.
    """
    backup_base = Path(os.path.expanduser("~/.daq_upload/oss/"))
    root_task_dir = Path(root_task_path)

    # Calculate relative path
    try:
        rel_path = dataset_path.relative_to(root_task_dir)
        # To avoid collisions if users upload same relative path from different roots,
        # we might want to include the root folder name.
        # Requirement: "按待上传的目录层级在~/.daq_upload/oss/ 下面"
        # Interpreted as: ~/.daq_upload/oss/ + rel_path
        # But if root_task_dir is /data/A, and dataset is /data/A/B
        # rel_path is B. Destination: ~/.daq_upload/oss/B

        # If the user uploads /data/C next time, and it has /data/C/B, conflict.
        # So it's better to preserve the root directory name too.
        # However, checking the requirement "按待上传的目录层级". usually means mirror the structure.
        # Let's use: ~/.daq_upload/oss/ + <root_folder_name> + rel_path

        dest_dir = backup_base / root_task_dir.name / rel_path
    except ValueError:
        # Should not happen if dataset_path is inside root_task_path
        dest_dir = backup_base / "misc" / dataset_path.name

    if not dest_dir.parent.exists():
        dest_dir.parent.mkdir(parents=True, exist_ok=True)

    # If dest exists, we might overwrite or skip. Move usually fails if dest exists.
    if dest_dir.exists():
        # Remove old backup to allow new move
        shutil.rmtree(dest_dir)

    try:
        shutil.move(str(dataset_path), str(dest_dir))
        logger.info(f"Moved {dataset_path} to {dest_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to move {dataset_path} to {dest_dir}: {e}")
        return False


class UploadApp:
    def __init__(self, root, args=None):
        self.root = root
        self.args = args
        self.root.title("Anyverse OSS 上传/下载器")
        self.root.geometry("1920x1280")

        # 统一字体设置，使用支持中文的字体 (例如 WenQuanYi Micro Hei 或 Microsoft YaHei)
        ui_font = ("Noto Sans CJK SC", 12)
        input_font = ("Noto Sans CJK SC", 12)
        list_font = ("Noto Sans CJK SC", 12)
        tab_font = ("Noto Sans CJK SC", 12)

        # 默认 bucket
        initial_bucket = "dataset-robot"
        bucket_state = "readonly"
        if self.args and self.args.bucket:
            initial_bucket = self.args.bucket
            bucket_state = "normal"

        self.bucket_var = StringVar(value=initial_bucket)

        # 远程前缀
        self.prefix_var = StringVar(value=REMOTE_UPLOAD_DIR)
        self.download_bucket_var = StringVar(value=initial_bucket)
        self.download_oss_path_var = StringVar()
        self.download_local_path_var = StringVar(value=os.getcwd())
        self.download_progress_var = DoubleVar(value=0.0)

        style = ttk.Style(root)
        style.configure("Large.TNotebook.Tab", font=tab_font, padding=(24, 10))

        self.notebook = ttk.Notebook(root, style="Large.TNotebook")
        self.notebook.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        upload_tab = Frame(self.notebook)
        download_tab = Frame(self.notebook)
        self.notebook.add(upload_tab, text="上传")
        self.notebook.add(download_tab, text="下载")

        # 上传页签输入栏
        top_frame = Frame(upload_tab, padx=10, pady=10)
        top_frame.pack(side="top", fill="x")

        Label(top_frame, text="Bucket:", font=ui_font).pack(side="left")
        Entry(
            top_frame,
            textvariable=self.bucket_var,
            width=15,
            font=input_font,
            state=bucket_state,
        ).pack(side="left", padx=5)

        Label(top_frame, text="目录前缀:", font=ui_font).pack(side="left", padx=5)
        Entry(top_frame, textvariable=self.prefix_var, width=70, font=input_font).pack(
            side="left", padx=5
        )

        Button(
            top_frame, text="添加输入目录", command=self.add_directory, font=ui_font
        ).pack(side="left", padx=5)

        Button(
            top_frame,
            text="删除选中",
            command=self.delete_selected,
            font=ui_font,
            bg="orange",
            fg="black",
        ).pack(side="left", padx=5)

        Button(
            top_frame,
            text="清空列表",
            command=self.clear_all,
            font=ui_font,
            bg="red",
            fg="black",
        ).pack(side="left", padx=5)

        Button(
            top_frame,
            text="开始上传",
            command=self.start_upload,
            bg="green",
            fg="black",
            font=ui_font,
        ).pack(side="left", padx=5)

        # 下载页签入口
        download_frame = Frame(download_tab, padx=10, pady=10)
        download_frame.pack(side="top", fill="x")

        Label(download_frame, text="下载 Bucket:", font=ui_font).pack(side="left")
        Entry(
            download_frame,
            textvariable=self.download_bucket_var,
            width=15,
            font=input_font,
            state="readonly",
        ).pack(side="left", padx=5)

        Label(download_frame, text="OSS文件路径:", font=ui_font).pack(
            side="left", padx=5
        )
        Entry(
            download_frame,
            textvariable=self.download_oss_path_var,
            width=60,
            font=input_font,
        ).pack(side="left", padx=5)

        Label(download_frame, text="本地路径:", font=ui_font).pack(side="left", padx=5)
        Entry(
            download_frame,
            textvariable=self.download_local_path_var,
            width=45,
            font=input_font,
        ).pack(side="left", padx=5)

        Button(
            download_frame,
            text="选择本地目录",
            command=self.select_download_directory,
            font=ui_font,
        ).pack(side="left", padx=5)

        Button(
            download_frame,
            text="开始下载",
            command=self.start_download,
            bg="green",
            fg="black",
            font=ui_font,
        ).pack(side="left", padx=5)

        download_status_frame = Frame(download_tab, padx=10, pady=10)
        download_status_frame.pack(side="top", fill="x")

        self.download_progress_bar = ttk.Progressbar(
            download_status_frame,
            variable=self.download_progress_var,
            maximum=100,
            length=500,
        )
        self.download_progress_bar.pack(side="left", padx=5)

        self.download_status_label = Label(
            download_status_frame,
            text="下载状态: 准备就绪",
            anchor="w",
            font=("Arial", 12),
        )
        self.download_status_label.pack(side="left", fill="x", expand=True, padx=5)

        # 上传任务列表区域
        list_frame = Frame(upload_tab, padx=10, pady=5)
        list_frame.pack(fill="both", expand=True)

        Label(list_frame, text="上传任务:", font=ui_font).pack(anchor="w")

        self.listbox = Listbox(
            list_frame, width=80, height=20, font=list_font, selectmode="extended"
        )
        self.listbox.pack(side="left", fill="both", expand=True)

        scrollbar = Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        # 状态栏
        self.status_label = Label(
            root, text="准备就绪", bd=1, relief="sunken", anchor="w", font=("Arial", 12)
        )
        self.status_label.pack(side="bottom", fill="x")

        # 任务列表 [{ "path": str, "status": "Pending"|"Uploading"|"Done", "progress": str }]
        self.tasks = load_history()
        self.refresh_list()

        self.is_uploading = False
        self.is_downloading = False

    def select_download_directory(self):
        initial_dir = self.download_local_path_var.get() or os.getcwd()
        dir_path = filedialog.askdirectory(
            initialdir=initial_dir, title="选择下载保存目录"
        )
        if dir_path:
            self.download_local_path_var.set(dir_path)

    def update_download_progress(self, percent, text):
        percent = max(0.0, min(100.0, float(percent)))
        self.download_progress_var.set(percent)
        self.download_status_label.config(text=f"下载状态: {text}")
        self.root.update_idletasks()

    def start_download(self):
        if self.is_downloading:
            return

        bucket_name = self.download_bucket_var.get().strip()
        remote_path = self.download_oss_path_var.get().strip()
        local_dir = self.download_local_path_var.get().strip()

        if not bucket_name or not remote_path or not local_dir:
            messagebox.showwarning("提示", "请填写 Bucket、OSS文件路径和本地路径")
            return

        if not os.path.isdir(local_dir):
            messagebox.showerror("错误", f"本地路径不存在或不是目录: {local_dir}")
            return

        self.is_downloading = True
        self.update_download_progress(0, "准备下载...")
        self.status_label.config(text="下载中...")

        thread = threading.Thread(
            target=self.run_download, args=(bucket_name, remote_path, local_dir)
        )
        thread.daemon = True
        thread.start()

    def run_download(self, bucket_name, remote_path, local_dir):
        progress_lock = threading.Lock()
        last_update_time = time.monotonic()
        last_consumed = 0

        def progress_cb(filename, consumed, total, rate):
            nonlocal last_update_time, last_consumed

            with progress_lock:
                now = time.monotonic()
                elapsed = now - last_update_time
                is_finished = total and consumed >= total
                if elapsed < 0.5 and not is_finished:
                    return

                byte_delta = max(0, consumed - last_consumed)
                speed = byte_delta / elapsed if elapsed > 0 else 0
                last_update_time = now
                last_consumed = consumed

            percent = rate if total else 100
            text = (
                f"{filename} | 完成度: {percent}% "
                f"({format_transfer_size(consumed)}/{format_transfer_size(total)}) "
                f"| 速度: {format_transfer_size(speed)}/s"
            )
            self.root.after(0, self.update_download_progress, percent, text)

        try:
            downloader = OSSUploader(bucket_name=bucket_name)
            success = downloader.download_path(
                remote_path,
                local_dir,
                progress_callback=progress_cb,
            )
        except Exception as e:
            err_msg = str(e)
            logger.error(f"下载失败: {err_msg}")
            self.root.after(
                0, lambda msg=err_msg: messagebox.showerror("错误", f"下载失败: {msg}")
            )
            success = False

        self.is_downloading = False
        if success:
            final_text = f"完成: {remote_path} -> {local_dir}"
            self.root.after(0, self.update_download_progress, 100, final_text)
            self.root.after(0, lambda: self.status_label.config(text="下载完成"))
            self.root.after(0, lambda: messagebox.showinfo("完成", "下载任务已完成"))
        else:
            self.root.after(
                0, self.update_download_progress, 0, "下载失败，请检查日志或重试"
            )
            self.root.after(0, lambda: self.status_label.config(text="下载失败"))

    def delete_selected(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("提示", "请先选择要删除的项目")
            return

        # 逆序删除，防止索引错位
        for index in reversed(selection):
            del self.tasks[index]

        save_history(self.tasks)
        self.refresh_list()

    def clear_all(self):
        if not self.tasks:
            return

        if messagebox.askyesno("确认", "确定要清空所有任务吗？"):
            self.tasks = []
            save_history(self.tasks)
            self.refresh_list()

    def add_directory(self):
        dir_path = filedialog.askdirectory(initialdir=os.getcwd(), title="选择上传目录")
        if dir_path:
            # 查重
            for index, task in enumerate(self.tasks):
                if task["path"] == dir_path:
                    if messagebox.askyesno("确认", "目录已在列表中，是否重新上传？"):
                        self.tasks[index]["status"] = "等待中"
                        self.tasks[index]["progress"] = "0%"
                        save_history(self.tasks)
                        self.refresh_list()
                    return

            task = {
                "path": dir_path,
                "status": "等待中",
                "progress": "0%",
                "target_bucket": self.bucket_var.get(),
            }
            self.tasks.append(task)
            save_history(self.tasks)
            self.refresh_list()

    def refresh_list(self):
        self.listbox.delete(0, END)
        for task in self.tasks:
            display_text = (
                f"[{task['status']}] {task['path']}  -- {task.get('progress', '')}"
            )
            self.listbox.insert(END, display_text)

    def update_task_status(self, index, status, progress=None):
        if index >= len(self.tasks):
            return

        self.tasks[index]["status"] = status
        if progress:
            self.tasks[index]["progress"] = progress

        # 优化刷新，只更新单行
        display_text = f"[{self.tasks[index]['status']}] {self.tasks[index]['path']}  -- {self.tasks[index].get('progress', '')}"
        self.listbox.delete(index)
        self.listbox.insert(index, display_text)
        self.root.update_idletasks()

    def start_upload(self):
        if self.is_uploading:
            return

        pending_indices = [
            i for i, t in enumerate(self.tasks) if t["status"] not in ["Done", "完成"]
        ]
        if not pending_indices:
            # Maybe re-upload failed ones or just info
            if messagebox.askyesno("提示", "所有任务已标记为完成。是否重试所有任务？"):
                for task in self.tasks:
                    task["status"] = "等待中"
                self.refresh_list()
                pending_indices = range(len(self.tasks))
            else:
                return

        self.is_uploading = True
        self.status_label.config(text="正在进行数据校验...")

        # Start thread
        thread = threading.Thread(
            target=self.run_checks_and_uploads, args=(pending_indices,)
        )
        thread.daemon = True
        thread.start()

    def run_checks_and_uploads(self, indices):
        passed_indices = []
        failed_dirs = []

        # 1. 执行校验
        for index in indices:
            task = self.tasks[index]
            path = task["path"]

            # Update status
            self.root.after(0, self.update_task_status, index, "校验中...", "0%")

            # 校验
            # run_checks 返回 True 表示通过, False 表示失败
            try:
                is_valid = run_checks(path)
            except Exception as e:
                logger.error(f"校验异常: {e}")
                is_valid = False

            if is_valid:
                passed_indices.append(index)
                self.root.after(
                    0, self.update_task_status, index, "等待上传", "校验通过"
                )
            else:
                failed_dirs.append(path)
                self.root.after(
                    0, self.update_task_status, index, "校验失败", "数据异常"
                )
                # 更新内部状态
                self.tasks[index]["status"] = "校验失败"
                self.tasks[index]["progress"] = "数据异常"

        # 保存状态
        save_history(self.tasks)

        # 2. 弹窗提示失败项 (如果有)
        if failed_dirs:
            msg = "以下目录数据校验未通过，将跳过上传：\n\n" + "\n".join(failed_dirs)
            self.root.after(0, lambda: messagebox.showwarning("数据校验结果", msg))

        # 3. 如果没有通过的，结束
        if not passed_indices:
            self.is_uploading = False
            self.root.after(
                0, lambda: self.status_label.config(text="校验结束，无有效任务")
            )
            return

        # 4. 执行上传 (仅上传校验通过的)
        self.root.after(0, lambda: self.status_label.config(text="上传中..."))
        self.run_uploads(passed_indices)

    def run_uploads(self, indices):
        bucket_name = self.bucket_var.get()
        custom_prefix_base = self.prefix_var.get()
        # 初始化 worker
        try:
            uploader = OSSUploader(bucket_name=bucket_name)
        except Exception as e:
            self.root.after(
                0, lambda: messagebox.showerror("错误", f"初始化上传器失败: {e}")
            )
            self.is_uploading = False
            return

        for index in indices:
            task = self.tasks[index]
            # 如果状态已经是 Done, 跳过 (除非是上面强制重试逻辑覆盖)
            if task["status"] in ["Done", "完成"]:
                continue

            local_dir = task["path"]

            self.root.after(0, self.update_task_status, index, "上传中", "0%")

            def progress_cb(filename, consumed, total, rate):
                # 限制更新频率，或者这里简单更新 status label
                # 若要更新列表里的进度，注意线程安全和刷新频率
                pass

            local_dir_name = os.path.basename(local_dir)
            current_remote_prefix = os.path.join(custom_prefix_base, local_dir_name)

            success = uploader.upload_directory(
                local_dir,
                remote_prefix=current_remote_prefix,
                progress_callback=progress_cb,
            )

            final_status = "完成" if success else "失败"
            self.root.after(
                0,
                self.update_task_status,
                index,
                final_status,
                "100%" if success else "错误",
            )

            # Save after each task
            self.tasks[index]["status"] = final_status
            save_history(self.tasks)

        self.is_uploading = False
        self.root.after(0, lambda: self.status_label.config(text="所有任务已完成"))
        self.root.after(0, lambda: messagebox.showinfo("完成", "选定的上传任务已完成"))


def run_cli(input_dir, bucket_name):
    print(f"开始 CLI 上传...")
    print(f"输入目录: {input_dir}")
    print(f"目标 Bucket: {bucket_name}")

    if not os.path.exists(input_dir):
        print(f"错误: 目录 {input_dir} 不存在。")
        return

    try:
        uploader = OSSUploader(bucket_name=bucket_name)
        success = uploader.upload_directory(input_dir)
        if success:
            print("上传成功完成。")
        else:
            print("上传失败。")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="机器人 OSS 上传客户端")
    parser.add_argument("--ui", action="store_true", help="启动 GUI 模式")
    parser.add_argument("--input_dir", type=str, help="要上传的本地目录（CLI 必需）")
    parser.add_argument("--bucket", type=str, help="目标 OSS Bucket（CLI 必需）")

    args = parser.parse_args()

    if args.ui:
        try:
            root = Tk()
            # 强制设置Tcl/Tk编码
            try:
                root.tk.call('encoding', 'system', 'utf-8')
                print("  已设置Tk编码为UTF-8")
            except:
                print("  设置Tk编码失败")
            app = UploadApp(root, args)
            root.mainloop()
        except Exception as e:
            logger.error(f"启动 GUI 失败: {e}")
            print("错误: 无法启动 GUI。请确保您有显示环境或使用 CLI 模式。")
    else:
        # Check requirements
        if not args.input_dir or not args.bucket:
            parser.print_help()
            print("\n错误: CLI 模式都需要 --input_dir 和 --bucket 参数。")
            exit(1)

        run_cli(args.input_dir, args.bucket)
