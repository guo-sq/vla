#!/usr/bin/env python3
# ruff: noqa
"""Visualize robot arm joint states alongside video frames for pour-water datasets."""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import urllib.parse
from http.server import ThreadingHTTPServer as HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
import pandas as pd

# ==================== 配置区 ====================
# 数据集路径列表（完整绝对路径）
# 与 --dataset 参数合并使用，自动按路径去重
DATASET_PATHS = [
    # ====== Anyverse dataset ======
    "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/fold_box_from_scratch/total_steps/fold_box_scratch_green.all.105s.20260311.batch.1",
    "/mnt/oss_data/anyverse/bipiper/fold_clothes/record.clothes.bipiper.v0115.1",
    "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/insert_tube/right_grab_and_attach_tube.random_tray_and_hole.approach_recover.50s.20260123.batch.30",
    "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/pack_socks/pack_socks.3_colors.M.continuous_pair_s0s1s2.panjinlong.20260224.batch.1",
    "/mnt/oss_data/anyverse/bipiper/pick_place/anyverse_pickAndplace_record/record.pick.place.withunzipandrestore_container.bipiper.v0319.2",
    "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt/seatbelt.single.back_home.zhangruixng.20260312.batch.9",
    "/mnt/oss_data/anyverse_pour_water_record/record.pourwater.bipiper.0319.3",
    # # ===== Public datasets =======
    # "/mnt/oss_data/X-Humanoid/RoboMIND2.0-Agilex_lerobot/agilex/fold_clothes/success_episodes",  # AGILE_X_BIMANUAL_ROBOMIND_2
    # "/mnt/oss_data/robotics-diffusion-transformer/rdt-ft-data/lerobot_data/airpods_on_second_layer",  # ALOHA
    # "/mnt/oss_data/IPEC-COMMUNITY/austin_buds_dataset_lerobot",  # AUSTIN_BUDS_OPENX_FRANKA
    # "/mnt/oss_data/IPEC-COMMUNITY/austin_sailor_dataset_lerobot",  # AUSTIN_SAILOR_OPENX_FRANKA
    # "/mnt/oss_data/IPEC-COMMUNITY/austin_sirius_dataset_lerobot",  # AUSTIN_SIRIUS_OPENX_FRANKA
    # "/mnt/oss_data/IPEC-COMMUNITY/bc_z_lerobot",  # BC_Z_OPENX_GOOGLE_ROBOT
    # "/mnt/oss_data/IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot",  # BERKELEY_AUTOLAB_OPENX_UR5
    # "/mnt/oss_data/IPEC-COMMUNITY/berkeley_cable_routing_lerobot",  # BERKELEY_CABLE_ROUTING_OPENX_FRANKA
    # "/mnt/oss_data/IPEC-COMMUNITY/berkeley_fanuc_manipulation_lerobot",  # BERKELEY_FANUC_MANIPULATION_OPENX_FANUC_MATE
    # "/mnt/oss_data/IPEC-COMMUNITY/berkeley_mvp_lerobot",  # BERKELEY_MVP_OPENX_XARM
    # "/mnt/oss_data/IPEC-COMMUNITY/berkeley_rpt_lerobot",  # BERKELEY_RPT_OPENX_FRANKA
    # "/mnt/oss_data/IPEC-COMMUNITY/bridge_orig_lerobot",  # BRIDGE_ORIG_OPENX_WINDOWX
    # "/mnt/oss_data/IPEC-COMMUNITY/cmu_play_fusion_lerobot",  # CMU_PLAY_FUSION_OPENX_FRANKA
    # "/mnt/oss_data/IPEC-COMMUNITY/cmu_stretch_lerobot",  # CMU_STRETCH_OPENX_STRETCH
    # "/mnt/oss_data/IPEC-COMMUNITY/dlr_edan_shared_control_lerobot",  # DLR_EDAN_SHARED_CONTROL_OPENX
    # "/mnt/oss_data/IPEC-COMMUNITY/fmb_dataset_lerobot",  # FMB_OPENX_FRANKA
    # "/mnt/oss_data/IPEC-COMMUNITY/fractal20220817_data_lerobot",  # FRACTRAL_OPENX_GOOGLE_ROBOT
    # "/mnt/oss_data/IPEC-COMMUNITY/furniture_bench_dataset_lerobot",  # FURNITURE_BENCH_OPENX_FRANKA
    # "/mnt/oss_data/x-humanoid-robomind/RoboMIND/benchmark1_0_lerobot/h5_agilex_3rgb/10_packplate/train",  # H5_AGILEX_3RGB_ROBOMIND_1
    # "/mnt/oss_data/x-humanoid-robomind/RoboMIND/benchmark1_0_lerobot/h5_franka_3rgb/2024_09_20_close_cabinet/train",  # H5_FRANKA_2RGB_ROBOMIND_1
    # "/mnt/oss_data/x-humanoid-robomind/RoboMIND/benchmark1_0_lerobot/h5_tienkung_gello_1rgb/clean_table_2_241211/train",  # H5_TIENKUNG_GELLO_1RGB_ROBOMIND_1
    # "/mnt/oss_data/x-humanoid-robomind/RoboMIND/benchmark1_0_lerobot/h5_tienkung_xsens_1rgb/battery_insertion_with_pullout/train",  # H5_TIENKUNG_XSENS_1RGB_ROBOMIND_1
    # "/mnt/oss_data/IPEC-COMMUNITY/iamlab_cmu_pickup_insert_lerobot",  # IAMLAB_CMU_PICKUP_INSERT_OPENX_FRANKA
    # "/mnt/oss_data/IPEC-COMMUNITY/jaco_play_lerobot",  # JACO_PLAY_OPENX_JACO
    # "/mnt/oss_data/IPEC-COMMUNITY/language_table_lerobot",  # LANGUAGE_TABLE_OPENX
    # "/mnt/oss_data/IPEC-COMMUNITY/nyu_door_opening_surprising_effectiveness_lerobot",  # NYU_DOOR_OPEN_OPENX_STRETCH
    # "/mnt/oss_data/IPEC-COMMUNITY/nyu_franka_play_dataset_lerobot",  # NYU_FRANKA_PLAY_OPENX
    # "/mnt/oss_data/IPEC-COMMUNITY/roboturk_lerobot",  # ROBOTURK_OPENX_FRANKA
    # "/mnt/oss_data/IPEC-COMMUNITY/stanford_hydra_dataset_lerobot",  # STANFORD_HYDRA_OPENX_FRANKA
    # "/mnt/oss_data/IPEC-COMMUNITY/taco_play_lerobot",  # TACO_PLAY_OPENX_FRANKA
    # "/mnt/oss_data/X-Humanoid/RoboMIND2.0-Tianyi_lerobot/tienyi/close_drawer_under_combined_cabinet/success_episodes",  # TIENYI_ROBOMIND_2
    # "/mnt/oss_data/IPEC-COMMUNITY/toto_lerobot",  # TOTO_OPENX_FRANKA
    # "/mnt/oss_data/IPEC-COMMUNITY/ucsd_kitchen_dataset_lerobot",  # USCD_KITCHEN_OPENX_XARM
    # "/mnt/oss_data/IPEC-COMMUNITY/utaustin_mutex_lerobot",  # UTAUSTIN_MUTEX_OPENX_FRANKA
    # "/mnt/oss_data/IPEC-COMMUNITY/viola_lerobot",  # VIOLA_OPENX_FRANKA
    # "/mnt/oss_data/jesbu1/bridge_v2_lerobot",  # WINDOWX
    # "/mnt/oss_data/InternRobotics/InternData-A1/real/genie1/Pick_a_bag_of_bread_with_the_left_arm__then_handover/set_0",  # a2d
    # "/mnt/oss_data/robocoin/RoboCOIN/Cobot_Magic_cap_the_pen_a",  # agilex_cobot_decoupled_magic
    # "/mnt/oss_data/lerobot/aloha_mobile_cabinet",  # aloha
    # "/mnt/oss_data/robocoin/RoboCOIN/alpha_bot_2_sticker",  # alpha_bot_2
    # "/mnt/oss_data/rhos-ai/gm100-cobotmagic-lerobot/task_00016",  # cobot_magic
    # "/mnt/oss_data/robocoin/RoboCOIN/AIRBOT_MMK2_mobile_car",  # discover_robotics_aitbot_mmk2
    # "/mnt/oss_data/robocoin/RoboCOIN/leju_robot_hotel_services_f",  # leju_robot
    # "/mnt/oss_data/OpenGalaxea/Galaxea-Open-World-Dataset/lerobot_unzip/Adjust_The_Air_Conditioner_Temperature_20250711_006",  # r1lite
    # "/mnt/oss_data/OpenGalaxea/Galaxea-Open-World-Dataset/lerobot_unzip/Put_The_Items_Into_The_Storage_Box_20250929_002_007",  # r1pro
    # "/mnt/oss_data/robocoin/RoboCOIN/AgiBot-g1_box_storage_a",  # ruantong_a2d
    # "/mnt/oss_data/rhos-ai/gm100-cobotmagic-lerobot/task_00001",  # songling_selfcollect
    # "/mnt/oss_data/robocoin/RoboCOIN/Galbot_g1_steamer_storage_baozi_j",  # yinhe
]
# ==================== 配置区结束 ====================


# 页面同时展示三路视频帧：左腕 | 头部 | 右腕（与数据集 videos 下目录名一致）
def default_viewer_cameras(head_camera: str):
    return [
        "observation.images.left_wrist",
        head_camera,
        "observation.images.right_wrist",
    ]


# Global cache for parsed episode data: state + action + (action - state) per frame
episode_cache = {}  # {(dataset_name, episode_name): {"state": ..., "action": ..., "action_state": ...}}

# Cache for threshold stats across repo_ids (datasets). Keyed by (mode, dim_index, low, high).
threshold_stats_cache = {}  # {(mode, dim_index, low, high): {"in_range": int, "total": int}}

# Cache for static stats across repo_ids (datasets).
# Keyed by (fps, joint_vth, gripper_vth, smoothing_half_window).
static_stats_cache = {}  # {(int, float, float, int): {"in_static": int, "total": int}}

PUBLIC_DATASET_MAP = {
    # Common
    "observation.images.cam_left_wrist_rgb": "observation.images.left_wrist",
    "observation.images.cam_right_wrist_rgb": "observation.images.right_wrist",
    # AgiBot-g1(ruantong_a2d)
    # AIRBOT_MMK2(discover_robotics_aitbot_mmk2)
    # Galbot_g1(yinhe)
    # opengalaxea: r1lite
    "observation.images.head_rgb": "observation.images.head",
    "observation.images.head_right_rgb": "observation.images.third_view",
    "observation.images.left_wrist_rgb": "observation.images.left_wrist",
    "observation.images.right_wrist_rgb": "observation.images.right_wrist",
    # Split_aloha(agilex_cobot_decoupled_magic)
    "observation.images.cam_high_rgb": "observation.images.head",
    # TODO(heyuan): why same mapping of cam_high_rgb and cam_high_left_rgb
    "observation.images.cam_high_left_rgb": "observation.images.third_view",  # R1_Lite
    # alpha_bot_2(alpha_bot_2)
    "observation.images.cam_head_rgb": "observation.images.head",
    # Cobot_Magic(agilex_cobot_decoupled_magic)
    "observation.images.cam_front_rgb": "observation.images.head",
    "observation.images.cam_left_wrist_rgb_rgb": "observation.images.left_wrist",
    "observation.images.cam_right_wrist_rgb_rgb": "observation.images.right_wrist",
    # G1edu-u3(unitree_g1)
    # "observation.images.cam_high_rgb": "observation.images.head",
    "observation.images.color_left_wrist": "observation.images.left_wrist",
    "observation.images.color_right_wrist": "observation.images.right_wrist",
    # leju_robot(leju_robot)
    "observation.images.camera_head_rgb": "observation.images.head",
    "observation.images.camera_left_wrist_rgb": "observation.images.left_wrist",
    "observation.images.camera_right_wrist_rgb": "observation.images.right_wrist",
    # ALOHA
    "observation.images.camera_front": "observation.images.head",
    "observation.images.cam_high": "observation.images.head",
    "observation.images.cam_left_wrist": "observation.images.left_wrist",
    "observation.images.cam_right_wrist": "observation.images.right_wrist",
    "observation.images.camera_left_wrist": "observation.images.left_wrist",
    "observation.images.camera_right_wrist": "observation.images.right_wrist",
    # INTERNDATA_Genie1
    "observation.images.hand_right": "observation.images.right_wrist",
    "observation.images.hand_left": "observation.images.left_wrist",
    # GLM
    "observation.images.camera_top": "observation.images.head",
    "observation.images.camera_wrist_left": "observation.images.left_wrist",
    "observation.images.camera_wrist_right": "observation.images.right_wrist",
    # H5_FRANKA_2RGB
    "observation.images.camera_left": "observation.images.left_wrist",
    "observation.images.camera_right": "observation.images.right_wrist",
    # OPENX_EMBODIMENT
    "observation.images.image": "observation.images.head",
    "observation.images.wrist_image": "observation.images.left_wrist",
    "observation.images.hand_image": "observation.images.left_wrist",
    "observation.images.top_image": "observation.images.third_view",
    "observation.images.wrist45_image": "observation.images.left_wrist",
    "observation.images.wrist225_image": "observation.images.right_wrist",
    "observation.images.image_additional_view": "observation.images.third_view",
    "observation.images.agentview_rgb": "observation.images.third_view",
    "observation.images.eye_in_hand_rgb": "observation.images.left_wrist",
    "observation.images.front_rgb": "observation.images.head",
    "observation.images.rgb": "observation.images.head",
    # oss_data/IPEC-COMMUNITY: has 4 camera, 3 third view
    "observation.images.image_0": "observation.images.head",
    "observation.images.image_1": "observation.images.third_view",
    "observation.images.image_2": "observation.images.left_wrist",
    "observation.images.image_3": "observation.images.right_wrist",
    "observation.images.image_side_2": "observation.images.third_view",
    "observation.images.image_side_1": "observation.images.head",
    "observation.images.image_wrist_2": "observation.images.right_wrist",
    "observation.images.image_wrist_1": "observation.images.left_wrist",
    # DROID
    "observation.images.exterior_1_left": "observation.images.head",
    "observation.images.exterior_2_left": "observation.images.third_view",
    "observation.images.rgb_gripper": "observation.images.left_wrist",
    "observation.images.rgb_static": "observation.images.head",
}


def clean_vector_row(row):
    """Convert one parquet cell to list[float|None].

    Parquet cells may be:
    - vector-like (list/tuple/np.ndarray/pyarrow list scalar) → flatten to list
    - scalar (float/int/np scalar) → wrap into length-1 list
    - None/NaN/Inf → None
    """
    if row is None:
        return []
    # Scalar fast path (covers python float/int and numpy scalars via float() below)
    if isinstance(row, (int, float)):
        v = float(row)
        if math.isnan(v) or math.isinf(v):
            return [None]
        return [v]
    # Some array scalars expose .tolist()
    if hasattr(row, "tolist") and not isinstance(row, (list, tuple)):
        try:
            row = row.tolist()
        except Exception:
            pass
    # After tolist(), it still might be a scalar
    if isinstance(row, (int, float)):
        v = float(row)
        if math.isnan(v) or math.isinf(v):
            return [None]
        return [v]
    # Strings/bytes are iterable but not vector cells
    if isinstance(row, (str, bytes, dict)):
        return []

    out = []
    try:
        itr = iter(row)
    except TypeError:
        # Unknown scalar-ish type
        try:
            v = float(row)
            if math.isnan(v) or math.isinf(v):
                return [None]
            return [v]
        except Exception:
            return []
    for val in itr:
        try:
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                out.append(None)
            else:
                fv = float(val)
                if math.isnan(fv) or math.isinf(fv):
                    out.append(None)
                else:
                    out.append(fv)
        except Exception:
            out.append(None)
    return out


def vector_diff_row(a_row, b_row):
    """Element-wise a - b; None if either operand missing."""
    n = min(len(a_row), len(b_row))
    diff = []
    for j in range(n):
        av, bv = a_row[j], b_row[j]
        if av is None or bv is None:
            diff.append(None)
        else:
            diff.append(float(av - bv))
    return diff


def _shape_numel(shape):
    if not isinstance(shape, list) or not shape:
        return None
    n = 1
    for x in shape:
        if not isinstance(x, int) or x <= 0:
            return None
        n *= x
    return n


def feature_keys_containing(feats, needle: str, skip_video: bool = True):
    """All feature keys in info.json whose name contains needle (case-insensitive). Excludes video by default."""
    nl = needle.lower()
    out = []
    for name, spec in (feats or {}).items():
        k = str(name)
        if nl not in k.lower():
            continue
        if skip_video and (spec or {}).get("dtype") == "video":
            continue
        out.append(k)
    out.sort()
    return out


def feature_dimension_names_from_spec(feature_spec):
    """Joint/vector axis labels from a single feature dict in info.json."""
    spec = feature_spec or {}
    sn = spec.get("names")
    if isinstance(sn, list) and sn:
        return [str(x) for x in sn]
    if isinstance(sn, dict):
        motors = sn.get("motors")
        if isinstance(motors, list) and motors:
            return [str(x) for x in motors]
    nm = _shape_numel(spec.get("shape"))
    if nm:
        return [f"dim_{i}" for i in range(nm)]
    return []


def joint_labels_for_feature_keys(feats, keys):
    """Axis labels for UI: one key → use names as-is; multiple keys → prefix with short key to avoid clashes."""
    if not keys:
        return []
    if len(keys) == 1:
        return feature_dimension_names_from_spec(feats.get(keys[0]))
    out = []
    for k in keys:
        short = k.split(".")[-1] if "." in k else k
        for n in feature_dimension_names_from_spec(feats.get(k)):
            out.append(f"{short}:{n}")
    return out


def concat_clean_row(raw_parts):
    """Concatenate several per-row vectors (as in parquet cells) into one flat float list."""
    acc = []
    for raw in raw_parts:
        acc.extend(clean_vector_row(raw))
    return acc


def parquet_column_set(parquet_path):
    """Column names in a parquet file without loading all data."""
    import pyarrow.parquet as pq

    return set(pq.read_schema(parquet_path).names)


def _ensure_default_state_action_keys(out: dict):
    if not out.get("state_parquet_keys"):
        out["state_parquet_keys"] = ["observation.state"]
    if not out.get("action_parquet_keys"):
        out["action_parquet_keys"] = ["action"]


# Prevent duplicated extraction/parsing for same episode
episode_locks = {}  # {(dataset_name, episode_name): threading.Lock()}
# Cache prompts/tasks from meta/episodes.jsonl
tasks_cache = {}  # {dataset_name: {episode_index:int -> tasks:list[str]}}
# Cache meta/info.json per dataset (robot_type, feature names, video keys)
info_json_cache = {}  # {dataset_name: {..., state_parquet_keys, action_parquet_keys}}


def get_episode_lock(ds_name, ep_name):
    key = (ds_name, ep_name)
    lock = episode_locks.get(key)
    if lock is None:
        lock = threading.Lock()
        episode_locks[key] = lock
    return lock


def parse_episode_index(ep_name: str):
    # episode_000001 -> 1
    try:
        if ep_name.startswith("episode_"):
            return int(ep_name.replace("episode_", ""))
        return int(ep_name)
    except Exception:
        return None


def resolve_dataset_entry(param: str):
    """Resolve DATASETS_INFO entry: prefer full path match, then basename (legacy)."""
    if not param:
        return None
    norm = os.path.normpath(param)
    for ds in DATASETS_INFO:
        if os.path.normpath(ds.get("path", "")) == norm:
            return ds
    for ds in DATASETS_INFO:
        if ds.get("name") == param:
            return ds
    return None


def canonical_dataset_id(param: str):
    """Stable id for caches (normalized root path)."""
    ds = resolve_dataset_entry(param)
    return os.path.normpath(ds["path"]) if ds else None


def get_dataset_path_by_name(ds_name: str):
    ds = resolve_dataset_entry(ds_name)
    return ds.get("path") if ds else None


def get_episode_tasks(ds_name: str, ep_name: str):
    """Return tasks list from meta/episodes.jsonl for the given dataset/episode."""
    ep_idx = parse_episode_index(ep_name)
    if ep_idx is None:
        return None

    canon = canonical_dataset_id(ds_name)
    if not canon:
        return None
    cached = tasks_cache.get(canon)
    if cached is not None and ep_idx in cached:
        return cached.get(ep_idx)

    ds_path = get_dataset_path_by_name(ds_name)
    if not ds_path:
        return None
    episodes_jsonl = os.path.join(ds_path, "meta", "episodes.jsonl")
    if not os.path.isfile(episodes_jsonl):
        return None

    mapping = tasks_cache.get(canon)
    if mapping is None:
        mapping = {}
        tasks_cache[canon] = mapping

    try:
        with open(episodes_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                idx = obj.get("episode_index", None)
                t = obj.get("tasks", None)
                if isinstance(idx, int) and isinstance(t, list):
                    mapping[idx] = t
    except Exception:
        return None

    return mapping.get(ep_idx)


def _video_feature_keys_from_features(feats: dict):
    """Collect observation.images.* keys with dtype video (LeRobot info.json)."""
    keys = []
    for fname, spec in (feats or {}).items():
        fk = str(fname)
        if not fk.startswith("observation.images."):
            continue
        if (spec or {}).get("dtype") != "video":
            continue
        keys.append(fk)
    keys.sort()
    return keys


def list_video_keys_for_dataset_root(root_path: str):
    """Read meta/info.json under root_path; return sorted video camera feature keys."""
    info_path = os.path.join(root_path, "meta", "info.json")
    if not os.path.isfile(info_path):
        return []
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        return _video_feature_keys_from_features(info.get("features") or {})
    except Exception:
        return []


def get_dataset_info_meta(ds_name: str):
    """Load meta/info.json: robot_type, video keys, state/action parquet keys and axis names.

    State/action columns are every feature key whose name contains 'state' / 'action' (non-video),
    sorted. If none found, fall back to observation.state and action.
    """
    canon = canonical_dataset_id(ds_name)
    if canon and canon in info_json_cache:
        return info_json_cache[canon]
    out = {
        "robot_type": None,
        "state_names": [],
        "action_names": [],
        "state_parquet_keys": [],
        "action_parquet_keys": [],
        "video_feature_keys": [],
    }
    ds_path = get_dataset_path_by_name(ds_name)
    if not ds_path:
        _ensure_default_state_action_keys(out)
        if canon:
            info_json_cache[canon] = out
        return out
    info_path = os.path.join(ds_path, "meta", "info.json")
    if not os.path.isfile(info_path):
        _ensure_default_state_action_keys(out)
        if canon:
            info_json_cache[canon] = out
        return out
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        out["robot_type"] = info.get("robot_type")
        feats = info.get("features") or {}
        out["video_feature_keys"] = _video_feature_keys_from_features(feats)
        sk = feature_keys_containing(feats, "state")
        ak = feature_keys_containing(feats, "action")
        if not sk:
            sk = ["observation.state"]
        if not ak:
            ak = ["action"]
        out["state_parquet_keys"] = sk
        out["action_parquet_keys"] = ak
        out["state_names"] = joint_labels_for_feature_keys(feats, sk)
        out["action_names"] = joint_labels_for_feature_keys(feats, ak)
    except Exception:
        pass
    _ensure_default_state_action_keys(out)
    if canon:
        info_json_cache[canon] = out
    return out


def resolve_viewer_camera_triplet(ds_name: str):
    """Pick dataset-native video keys for [left, center, right] using PUBLIC_DATASET_MAP + HEAD_CAMERA_NAME."""
    meta = get_dataset_info_meta(ds_name)
    keys = meta.get("video_feature_keys") or []
    left_t = "observation.images.left_wrist"
    center_t = HEAD_CAMERA_NAME
    right_t = "observation.images.right_wrist"
    if not keys:
        return (VIEWER_CAMERAS[0], VIEWER_CAMERAS[1], VIEWER_CAMERAS[2])
    sl, sc, sr = None, None, None
    for k in keys:
        c = PUBLIC_DATASET_MAP.get(k, k)
        if c == left_t and sl is None:
            sl = k
        elif c == center_t and sc is None:
            sc = k
        elif c == right_t and sr is None:
            sr = k
    return (sl, sc, sr)


def camera_label_for_ui(dataset_key):
    """Short label for viewer caption (strip observation.images.)."""
    if not dataset_key:
        return ""
    s = str(dataset_key)
    p = "observation.images."
    if s.startswith(p):
        return s[len(p) :]
    return s


def triplet_labels_and_meta(triplet):
    """Human-readable labels for three slots; empty string if slot unused."""
    return [camera_label_for_ui(triplet[0]), camera_label_for_ui(triplet[1]), camera_label_for_ui(triplet[2])]


def discover_datasets(dataset_path, cameras):
    """Discover datasets from the given path.

    If dataset_path directly contains data/chunk-*/episode_*.parquet, treat as single dataset.
    Otherwise, scan immediate subdirectories for valid datasets.

    Returns a list of dicts: {name, path, episodes: [{name, parquet_path, video_paths}]}
    """
    datasets = []

    # Check if the path itself is a dataset
    parquet_files = sorted(glob.glob(os.path.join(dataset_path, "data", "chunk-*", "episode_*.parquet")))
    if parquet_files:
        video_cams = list_video_keys_for_dataset_root(dataset_path) or list(cameras)
        episodes = []
        for pq in parquet_files:
            ep_name = os.path.splitext(os.path.basename(pq))[0]
            chunk_dir = os.path.basename(os.path.dirname(pq))
            video_paths = {
                cam: os.path.join(dataset_path, "videos", chunk_dir, cam, f"{ep_name}.mp4") for cam in video_cams
            }
            episodes.append(
                {
                    "name": ep_name,
                    "parquet_path": pq,
                    "video_paths": video_paths,
                }
            )
        datasets.append(
            {
                "name": os.path.basename(os.path.normpath(dataset_path)),
                "path": dataset_path,
                "episodes": episodes,
            }
        )
        return datasets

    # Scan immediate subdirectories
    for subdir in sorted(os.listdir(dataset_path)):
        subdir_path = os.path.join(dataset_path, subdir)
        if not os.path.isdir(subdir_path):
            continue
        parquet_files = sorted(glob.glob(os.path.join(subdir_path, "data", "chunk-*", "episode_*.parquet")))
        if not parquet_files:
            continue
        video_cams = list_video_keys_for_dataset_root(subdir_path) or list(cameras)
        episodes = []
        for pq in parquet_files:
            ep_name = os.path.splitext(os.path.basename(pq))[0]
            chunk_dir = os.path.basename(os.path.dirname(pq))
            video_paths = {
                cam: os.path.join(subdir_path, "videos", chunk_dir, cam, f"{ep_name}.mp4") for cam in video_cams
            }
            episodes.append(
                {
                    "name": ep_name,
                    "parquet_path": pq,
                    "video_paths": video_paths,
                }
            )
        datasets.append(
            {
                "name": subdir,
                "path": subdir_path,
                "episodes": episodes,
            }
        )

    return datasets


# Global variable to store datasets info for API
DATASETS_INFO = []
BASE_DIR = ""
# Set in main(): cameras shown in viewer (left | head | right) and head dir name for --camera
VIEWER_CAMERAS = default_viewer_cameras("observation.images.head")
HEAD_CAMERA_NAME = "observation.images.head"


def main():
    global DATASETS_INFO, BASE_DIR, VIEWER_CAMERAS, HEAD_CAMERA_NAME
    parser = argparse.ArgumentParser(description="Visualize pour-water dataset joint states")
    parser.add_argument(
        "--dataset",
        required=False,
        default=None,
        help="Path to dataset directory or root directory containing multiple datasets",
    )
    parser.add_argument(
        "--camera",
        default="observation.images.head",
        help="Center (head) camera directory name; left/right wrist are fixed (default: observation.images.head)",
    )
    parser.add_argument("--extract", action="store_true", help="Extract video frames (skipped by default)")
    parser.add_argument(
        "--extract-episodes", default=None, help="Only extract specific episodes, e.g. 'episode_000000,episode_000001'"
    )
    parser.add_argument("--port", type=int, default=8765, help="HTTP port for viewer/API (default: 8765)")
    args = parser.parse_args()

    viewer_cameras = default_viewer_cameras(args.camera)
    VIEWER_CAMERAS = viewer_cameras
    HEAD_CAMERA_NAME = args.camera
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Collect paths from config and CLI
    all_paths = list(DATASET_PATHS)
    if args.dataset:
        all_paths.append(args.dataset)

    # Deduplicate by normalized path
    seen = set()
    unique_paths = []
    for p in all_paths:
        norm = os.path.normpath(p)
        if norm not in seen:
            seen.add(norm)
            unique_paths.append(p)

    # Validate each path
    valid_paths = []
    for p in unique_paths:
        if not os.path.isdir(p):
            print(f"Warning: {p} is not a valid directory, skipping.", file=sys.stderr)
            continue
        valid_paths.append(p)

    if not valid_paths:
        print(
            "Error: No dataset paths specified. Add paths to DATASET_PATHS in visualize.py or use --dataset.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Discover datasets from all valid paths
    DATASETS_INFO = []
    for path in valid_paths:
        DATASETS_INFO.extend(discover_datasets(path, viewer_cameras))

    if not DATASETS_INFO:
        print(f"Error: No valid datasets found in the specified paths", file=sys.stderr)
        print("Expected directories containing data/chunk-*/episode_*.parquet", file=sys.stderr)
        sys.exit(1)

    total_episodes = sum(len(ds["episodes"]) for ds in DATASETS_INFO)
    print(f"Discovered {len(DATASETS_INFO)} dataset(s), {total_episodes} episode(s)")
    for ds in DATASETS_INFO:
        print(f"  {ds['name']}: {len(ds['episodes'])} episodes")

    # Pre-extract frames for the first episode to speed up first view
    try:
        first_ds = DATASETS_INFO[0]
        if first_ds["episodes"]:
            print("\nPre-extracting frames for first episode (if needed)...")
            _tri = resolve_viewer_camera_triplet(first_ds["name"])
            _active = [c for c in _tri if c]
            _hc = _tri[1] or HEAD_CAMERA_NAME
            # Startup optimization: only extract frame_000000.jpg for each camera.
            # Full extraction can be triggered later by the viewer (or via --extract).
            ep0 = first_ds["episodes"][0]
            migrate_legacy_flat_frames(BASE_DIR, first_ds["name"], ep0["name"], 0, _hc)
            video_paths = ep0.get("video_paths") or {}
            for cam in _active:
                vpath = video_paths.get(cam)
                if not vpath or not os.path.isfile(vpath):
                    continue
                frames_dir = frames_dir_for_camera(BASE_DIR, first_ds["name"], ep0["name"], cam)
                f0 = os.path.join(frames_dir, "frame_000000.jpg")
                if os.path.isfile(f0):
                    continue
                extract_first_frame(vpath, f0)
    except Exception as e:
        print(f"Warning: Pre-extraction failed: {e}", file=sys.stderr)

    # Frame extraction (only when --extract is specified)
    if args.extract:
        episode_filter = set(args.extract_episodes.split(",")) if args.extract_episodes else None
        # Flatten all episodes with dataset name for progress tracking
        all_tasks = []
        for ds in DATASETS_INFO:
            for ep in ds["episodes"]:
                if episode_filter and ep["name"] not in episode_filter:
                    continue
                all_tasks.append((ds["name"], ep))
        print(f"\nExtracting frames: {len(all_tasks)} episode(s) to process")
        import time

        extract_start = time.time()
        for idx, (ds_name, ep) in enumerate(all_tasks, 1):
            _tri = resolve_viewer_camera_triplet(ds_name)
            _active = [c for c in _tri if c]
            _hc = _tri[1] or HEAD_CAMERA_NAME
            process_episode_frames(ds_name, ep, BASE_DIR, idx, len(all_tasks), _active, _hc)
        elapsed = time.time() - extract_start
        print(f"\nFrame extraction complete: {len(all_tasks)} episode(s) in {format_time(elapsed)}")
    else:
        print("Skipping frame extraction (use --extract to extract frames)")

    # Generate lightweight HTML (no embedded data)
    html = generate_html(DATASETS_INFO)
    save_html(html, os.path.join(BASE_DIR, "viewer.html"))

    # Start API server (auto-fallback if port is taken)
    port = int(args.port)
    os.chdir(BASE_DIR)
    server = None
    for try_port in range(port, port + 20):
        try:
            server = HTTPServer(("0.0.0.0", try_port), APIHandler)
            port = try_port
            break
        except OSError as e:
            if getattr(e, "errno", None) == 98:  # Address already in use
                continue
            raise
    if server is None:
        print(f"Error: Cannot bind to ports {args.port}-{args.port+19}", file=sys.stderr)
        sys.exit(2)

    print(f"\nAPI server running at http://localhost:{port}/viewer.html")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


def get_parquet_row_count(parquet_path):
    """Get row count from a parquet file without loading full data."""
    df = pd.read_parquet(parquet_path, columns=[])
    return len(df)


def _moving_average_axis0(arr: np.ndarray, h: int) -> np.ndarray:
    """Match openpi.training.frame_attributes_preprocessors.utils.moving_average(arr, h, axis=0)."""
    a = np.asarray(arr)
    if a.size == 0:
        return a
    h = int(h)
    if h <= 0:
        return a
    n = a.shape[0]
    # cumulative sum along time
    cum = np.cumsum(a, axis=0)
    right = np.clip(np.arange(n) + h, 0, n - 1)
    left = np.clip(np.arange(n) - h, 0, n - 1)
    window_sum = cum[right] - np.where(left[:, None] > 0, cum[left - 1], 0)
    window_size = (right - left + 1).astype(a.dtype)
    return window_sum / window_size[:, None]


def _compute_smoothed_velocities(states: np.ndarray, fps: int, smoothing_half_window: int) -> np.ndarray:
    """Match openpi.training.frame_attributes_preprocessors.utils.compute_smoothed_velocities."""
    dt = 1.0 / float(max(1, int(fps)))
    raw = np.gradient(states, dt, axis=0)
    return _moving_average_axis0(raw, smoothing_half_window)


def _build_velocity_threshold(state_dim: int, joint_vth: float, gripper_vth: float) -> np.ndarray:
    """Match openpi.training.frame_attributes_preprocessors.utils.build_velocity_threshold for 14-DOF.

    For non-14 dims (viewer-only), fall back to per-dim joint_vth.
    """
    if int(state_dim) == 14:
        return np.array(
            [
                *[float(joint_vth)] * 6,
                float(gripper_vth),
                *[float(joint_vth)] * 6,
                float(gripper_vth),
            ],
            dtype=float,
        )
    return np.full((int(state_dim),), float(joint_vth), dtype=float)


def count_parquet_static_frames(parquet_path: str, state_keys: list[str], cfg: dict) -> tuple[int, int]:
    """Return (num_static, total) for one episode parquet."""
    fps = int(cfg.get("fps", 30) or 30)
    joint_vth = float(cfg.get("joint_velocity_threshold", 0.1) or 0.1)
    gripper_vth = float(cfg.get("gripper_velocity_threshold", 0.2) or 0.2)
    smoothing_h = int(cfg.get("smoothing_half_window", 2) or 2)
    if fps <= 0 or smoothing_h < 0 or joint_vth < 0 or gripper_vth < 0:
        raise ValueError("Invalid static cfg")

    df = pd.read_parquet(parquet_path, columns=state_keys)
    total = len(df)
    if total <= 0:
        return 0, 0
    cols = [df[k].tolist() for k in state_keys]
    # concat into (T, D) like training concat_clean_row does, but here states are already vectors.
    rows = []
    for i in range(total):
        parts = []
        for j in range(len(state_keys)):
            v = cols[j][i]
            if isinstance(v, (list, tuple, np.ndarray)):
                parts.extend(list(v))
            else:
                parts.append(v)
        rows.append(parts)
    states = np.asarray(rows, dtype=float)
    if states.ndim != 2:
        states = states.reshape((total, -1))
    th = _build_velocity_threshold(states.shape[1], joint_vth, gripper_vth)
    vel = _compute_smoothed_velocities(states, fps, smoothing_h)
    is_static = np.all(np.abs(vel) < th[None, :], axis=1)
    return int(is_static.sum()), int(total)


def video_frame_count_ffprobe(video_path: str):
    """Best-effort frame count from ffprobe; returns int or None."""
    if not video_path or not os.path.isfile(video_path) or not shutil.which("ffprobe"):
        return None
    try:
        # Prefer nb_read_frames via -count_frames (works even when nb_frames is missing)
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames,nb_frames",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0:
            return None
        # ffprobe may print one or two lines; take the first parseable positive int
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                n = int(line)
                if n > 0:
                    return n
            except Exception:
                continue
        return None
    except Exception:
        return None


def _safe_int(s, default, min_v=None, max_v=None):
    try:
        v = int(s)
    except Exception:
        return default
    if min_v is not None and v < min_v:
        return min_v
    if max_v is not None and v > max_v:
        return max_v
    return v


def _thumb_path(base_dir: str, frames_key: str, ep_name: str, camera: str):
    return os.path.join(base_dir, ".cache", "thumbs", frames_key, ep_name, f"{camera}.jpg")


def _triptych_thumb_path(base_dir: str, frames_key: str, ep_name: str):
    return os.path.join(base_dir, ".cache", "thumbs_triptych", frames_key, f"{ep_name}.jpg")


def ensure_episode_thumbnail(ds_entry: dict, ep_entry: dict, base_dir: str, camera: str):
    """Create cached thumbnail (frame 0) for an episode/camera; returns path or None."""
    frames_key = ds_entry["name"]
    ep_name = ep_entry["name"]
    camera = str(camera or "")
    if not camera:
        return None

    out_path = _thumb_path(base_dir, frames_key, ep_name, camera)
    if os.path.isfile(out_path):
        return out_path
    vpath = (ep_entry.get("video_paths") or {}).get(camera)
    if not vpath or not os.path.isfile(vpath) or not shutil.which("ffmpeg"):
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Generate a small JPEG thumbnail for fast grid preview
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        vpath,
        "-frames:v",
        "1",
        "-vf",
        "scale=320:-1",
        out_path,
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=120)
        if r.returncode != 0:
            try:
                os.remove(out_path)
            except OSError:
                pass
            return None
        return out_path if os.path.isfile(out_path) else None
    except Exception:
        try:
            os.remove(out_path)
        except OSError:
            pass
        return None


def ensure_episode_triptych_thumbnail(ds_entry: dict, ep_entry: dict, base_dir: str):
    """Create cached triptych thumbnail (left|center|right frame0); returns path or None."""
    frames_key = ds_entry["name"]
    ep_name = ep_entry["name"]
    out_path = _triptych_thumb_path(base_dir, frames_key, ep_name)
    if os.path.isfile(out_path):
        return out_path

    # Determine camera triplet (dataset-native keys)
    tri = resolve_viewer_camera_triplet(ds_entry.get("path") or ds_entry.get("name"))
    cams = [tri[0], tri[1], tri[2]]
    # Generate (or reuse) per-camera thumbs first
    thumb_paths = []
    for cam in cams:
        p = ensure_episode_thumbnail(ds_entry, ep_entry, base_dir, cam) if cam else None
        thumb_paths.append(p if p and os.path.isfile(p) else None)

    if not any(thumb_paths):
        return None

    # Stitch using OpenCV (already a dependency)
    imgs = []
    heights = []
    for p in thumb_paths:
        if not p:
            imgs.append(None)
            continue
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            imgs.append(None)
            continue
        imgs.append(im)
        heights.append(im.shape[0])

    h = int(min(heights)) if heights else 180
    if h <= 0:
        h = 180

    normed = []
    for im in imgs:
        if im is None:
            # placeholder (dark)
            normed.append(None)
            continue
        ih, iw = im.shape[:2]
        if ih != h and ih > 0:
            nw = max(1, int(iw * (h / ih)))
            im = cv2.resize(im, (nw, h), interpolation=cv2.INTER_AREA)
        normed.append(im)

    # Fill missing with black panels matching first available width (or 320)
    widths = [im.shape[1] for im in normed if im is not None]
    w0 = int(widths[0]) if widths else 320
    black = (0, 0, 0)
    panels = []
    for im in normed:
        if im is None:
            panels.append((np.zeros((h, w0, 3), dtype=np.uint8)))
        else:
            panels.append(im)
    try:
        trip = cv2.hconcat(panels)
    except Exception:
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = cv2.imwrite(out_path, trip, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return out_path if ok and os.path.isfile(out_path) else None
    out_path = _thumb_path(base_dir, frames_key, ep_name, camera)
    if os.path.isfile(out_path):
        return out_path
    vpath = (ep_entry.get("video_paths") or {}).get(camera)
    if not vpath or not os.path.isfile(vpath) or not shutil.which("ffmpeg"):
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Generate a small JPEG thumbnail for fast grid preview
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        vpath,
        "-frames:v",
        "1",
        "-vf",
        "scale=320:-1",
        out_path,
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=120)
        if r.returncode != 0:
            try:
                os.remove(out_path)
            except OSError:
                pass
            return None
        return out_path if os.path.isfile(out_path) else None
    except Exception:
        try:
            os.remove(out_path)
        except OSError:
            pass
        return None


def count_jpg_files(dir_path):
    """Count .jpg files in a directory."""
    if not os.path.isdir(dir_path):
        return 0
    return len(glob.glob(os.path.join(dir_path, "*.jpg")))


def frames_dir_for_camera(base_dir, dataset_name, ep_name, camera):
    return os.path.join(base_dir, ".cache", "frames", dataset_name, ep_name, camera)


def migrate_legacy_flat_frames(base_dir, dataset_name, ep_name, row_count, head_camera):
    """Older versions saved head frames as frames/<ds>/<ep>/frame_*.jpg; move under head_camera/."""
    episode_root = os.path.join(base_dir, ".cache", "frames", dataset_name, ep_name)
    if not os.path.isdir(episode_root):
        return
    legacy_jpgs = glob.glob(os.path.join(episode_root, "*.jpg"))
    if not legacy_jpgs:
        return
    if len(legacy_jpgs) != row_count or row_count <= 0:
        return
    head_dir = frames_dir_for_camera(base_dir, dataset_name, ep_name, head_camera)
    if count_jpg_files(head_dir) == row_count:
        return
    os.makedirs(head_dir, exist_ok=True)
    for fpath in legacy_jpgs:
        shutil.move(fpath, os.path.join(head_dir, os.path.basename(fpath)))


def format_progress_bar(current, total, width=30):
    """Return a progress bar string like [████████░░░░░░░░] 52%."""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct * 100:5.1f}%"


def format_time(seconds):
    """Format seconds to MM:SS."""
    if seconds < 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# OpenCV 自带的 FFmpeg 常无法解码 AV1（终端刷屏 av1 / Missing Sequence Header）；改用系统 ffmpeg（如 libdav1d）
CODECS_DECODE_WITH_SYSTEM_FFMPEG = frozenset({"av1"})


def ffprobe_video_codec(video_path):
    """Return first video stream codec name, lowercased, or None."""
    if not video_path or not os.path.isfile(video_path) or not shutil.which("ffprobe"):
        return None
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0:
            return None
        return (r.stdout or "").strip().lower() or None
    except Exception:
        return None


def use_system_ffmpeg_for_video(video_path):
    if not shutil.which("ffmpeg"):
        return False
    c = ffprobe_video_codec(video_path)
    return c in CODECS_DECODE_WITH_SYSTEM_FFMPEG if c else False


def _clear_ff_temp_jpegs(output_dir):
    for f in glob.glob(os.path.join(output_dir, "_ff_*.jpg")):
        try:
            os.remove(f)
        except OSError:
            pass


def _finalize_ff_temp_jpegs(output_dir):
    """Rename ffmpeg output _ff_000001.jpg … to frame_000000.jpg …"""
    chunks = sorted(glob.glob(os.path.join(output_dir, "_ff_*.jpg")))
    for i, p in enumerate(chunks):
        dest = os.path.join(output_dir, f"frame_{i:06d}.jpg")
        os.replace(p, dest)
    return len(chunks)


def extract_frames_via_ffmpeg_cli(video_path, output_dir, expected_count, progress_prefix="", progress_cb=None):
    """Decode with system ffmpeg; writes frame_XXXXXX.jpg matching OpenCV naming."""
    import time

    _clear_ff_temp_jpegs(output_dir)
    tmp_pat = os.path.join(output_dir, "_ff_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vsync",
        "0",
    ]
    if expected_count > 0:
        cmd.extend(["-vframes", str(expected_count)])
    cmd.append(tmp_pat)

    t0 = time.time()
    if progress_cb:
        proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        last_report_t = 0.0
        while proc.poll() is None:
            time.sleep(0.25)
            now = time.time()
            n = len(glob.glob(os.path.join(output_dir, "_ff_*.jpg")))
            if now - last_report_t >= 0.45 or (expected_count > 0 and n >= expected_count):
                last_report_t = now
                elapsed = now - t0
                fps = n / elapsed if elapsed > 0 else 0.0
                eta = (expected_count - n) / fps if fps > 0 and expected_count > 0 else None
                progress_cb(
                    {
                        "current": n,
                        "total": expected_count,
                        "fps": fps,
                        "eta_sec": eta,
                    }
                )
        rc = proc.returncode
    else:
        r = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
        rc = r.returncode
        if rc != 0:
            msg = (r.stderr or "").strip()[:500]
            print(f"  {progress_prefix}ffmpeg failed ({rc}): {msg}")

    if rc != 0:
        _clear_ff_temp_jpegs(output_dir)
        if progress_cb:
            raise RuntimeError(f"ffmpeg exited {rc} for {video_path}")
        return 0

    count = _finalize_ff_temp_jpegs(output_dir)
    elapsed = time.time() - t0
    if progress_cb:
        progress_cb({"current": count, "total": expected_count, "fps": None, "eta_sec": 0})
    else:
        bar = format_progress_bar(count, max(expected_count, count) or 1)
        print(
            f"\r  {progress_prefix}{bar} {count}/{expected_count} frames | ffmpeg | {format_time(elapsed)} elapsed",
            flush=True,
        )
    return count


def extract_frames(video_path, output_dir, expected_count, progress_prefix=""):
    """Extract all frames from video to JPG files with progress display."""
    os.makedirs(output_dir, exist_ok=True)
    if use_system_ffmpeg_for_video(video_path):
        print(
            f"  {progress_prefix}Using system ffmpeg for AV1 (OpenCV decoder not reliable on this platform).",
            flush=True,
        )
        return extract_frames_via_ffmpeg_cli(
            video_path, output_dir, expected_count, progress_prefix=progress_prefix, progress_cb=None
        )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Fallback: OpenCV may fail on some codecs (e.g. AV1). Try system ffmpeg if available.
        if shutil.which("ffmpeg"):
            return extract_frames_via_ffmpeg_cli(
                video_path, output_dir, expected_count, progress_prefix=progress_prefix, progress_cb=None
            )
        print(f"  Warning: Cannot open video {video_path}")
        return 0

    import time

    start_time = time.time()
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
        # Update progress every 50 frames or at the end
        if count % 50 == 0 or count == expected_count:
            elapsed = time.time() - start_time
            fps = count / elapsed if elapsed > 0 else 0
            eta = (expected_count - count) / fps if fps > 0 else 0
            bar = format_progress_bar(count, expected_count)
            print(
                f"\r  {progress_prefix}{bar} {count}/{expected_count} | {fps:.0f} fps | ETA {format_time(eta)}",
                end="",
                flush=True,
            )
    cap.release()
    elapsed = time.time() - start_time
    print(f"\r  {progress_prefix}{format_progress_bar(count, count)} {count} frames | {format_time(elapsed)} elapsed")
    return count


def extract_frames_with_callback(video_path, output_dir, expected_count, progress_cb=None):
    """Extract frames and report progress via callback(progress_dict)."""
    os.makedirs(output_dir, exist_ok=True)
    if use_system_ffmpeg_for_video(video_path):
        return extract_frames_via_ffmpeg_cli(
            video_path, output_dir, expected_count, progress_prefix="", progress_cb=progress_cb
        )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Fallback: OpenCV may fail on some codecs (e.g. AV1). Try system ffmpeg if available.
        if shutil.which("ffmpeg"):
            return extract_frames_via_ffmpeg_cli(
                video_path, output_dir, expected_count, progress_prefix="", progress_cb=progress_cb
            )
        raise RuntimeError(f"Cannot open video: {video_path}")

    import time

    start_time = time.time()
    count = 0
    last_report = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
        if progress_cb and (count - last_report >= 25 or count == expected_count):
            last_report = count
            elapsed = time.time() - start_time
            fps = count / elapsed if elapsed > 0 else 0.0
            eta = (expected_count - count) / fps if fps > 0 else None
            progress_cb(
                {
                    "current": count,
                    "total": expected_count,
                    "fps": fps,
                    "eta_sec": eta,
                }
            )
    cap.release()
    if progress_cb:
        progress_cb({"current": count, "total": expected_count, "fps": None, "eta_sec": 0})
    return count


def extract_first_frame(video_path: str, out_path: str):
    """Extract the first frame to a single JPG file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if use_system_ffmpeg_for_video(video_path):
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video_path, "-frames:v", "1", out_path]
        subprocess.run(cmd, check=True)
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video_path, "-frames:v", "1", out_path]
            subprocess.run(cmd, check=True)
            return
        raise RuntimeError(f"Cannot open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Failed to read first frame")
    cv2.imwrite(out_path, frame)


def parquet_first_row(parquet_path: str, columns: list[str]):
    """Read first row efficiently (best-effort) using pyarrow, return pandas DataFrame."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    if pf.num_row_groups <= 0:
        return pd.DataFrame({c: [] for c in columns})
    # Read only first row group then slice to 1 row.
    tbl = pf.read_row_group(0, columns=columns)
    if tbl.num_rows > 1:
        tbl = tbl.slice(0, 1)
    return tbl.to_pandas()


def count_parquet_dim_in_range(parquet_path: str, keys: list[str], dim_index: int, low: float, high: float) -> tuple[int, int]:
    """Count rows where concatenated vector dim is within [low, high].

    Uses pyarrow batch iteration to avoid loading the full parquet into memory.
    Returns (in_range_count, total_rows_seen).
    """
    import pyarrow.parquet as pq

    if dim_index is None or dim_index < 0:
        return (0, 0)

    pf = pq.ParquetFile(parquet_path)
    total = 0
    hit = 0
    # Iterate row groups in batches
    for batch in pf.iter_batches(batch_size=2048, columns=keys):
        # batch is a pyarrow.RecordBatch
        cols = [batch.column(i) for i in range(batch.num_columns)]
        n = batch.num_rows
        for r in range(n):
            parts = []
            for c in cols:
                try:
                    parts.append(c[r].as_py())
                except Exception:
                    parts.append(None)
            vec = concat_clean_row(parts)
            if dim_index >= len(vec):
                total += 1
                continue
            v = vec[dim_index]
            if v is None:
                total += 1
                continue
            try:
                fv = float(v)
            except Exception:
                total += 1
                continue
            if math.isnan(fv) or math.isinf(fv):
                total += 1
                continue
            if low <= fv <= high:
                hit += 1
            total += 1
    return (hit, total)


def count_parquet_dim_in_range_multi(
    parquet_path: str,
    keys: list[str],
    dim_indices: list[int],
    low: float,
    high: float,
    op: str,
) -> tuple[int, int]:
    """Count rows where ALL/ANY of dims are within [low, high]. Returns (in_range, total)."""
    op = "and" if op == "and" else "or"
    dims = [d for d in dim_indices if isinstance(d, int) and d >= 0]
    if not dims:
        return (0, 0)

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    total = 0
    hit = 0
    for batch in pf.iter_batches(batch_size=2048, columns=keys):
        cols = [batch.column(i) for i in range(batch.num_columns)]
        n = batch.num_rows
        for r in range(n):
            parts = []
            for c in cols:
                try:
                    parts.append(c[r].as_py())
                except Exception:
                    parts.append(None)
            vec = concat_clean_row(parts)
            if op == "and":
                ok = True
                for d in dims:
                    if d >= len(vec):
                        ok = False
                        break
                    v = vec[d]
                    if v is None:
                        ok = False
                        break
                    try:
                        fv = float(v)
                    except Exception:
                        ok = False
                        break
                    if math.isnan(fv) or math.isinf(fv) or not (low <= fv <= high):
                        ok = False
                        break
                if ok:
                    hit += 1
            else:
                ok = False
                for d in dims:
                    if d >= len(vec):
                        continue
                    v = vec[d]
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if math.isnan(fv) or math.isinf(fv):
                        continue
                    if low <= fv <= high:
                        ok = True
                        break
                if ok:
                    hit += 1
            total += 1
    return (hit, total)


def count_parquet_constraints(
    parquet_path: str,
    keys: list[str],
    dim_indices: list[int],
    lows: list[float],
    highs: list[float],
    op: str,
) -> tuple[int, int]:
    """Count rows where constraints are satisfied (AND/OR across dims, each with its own [low, high])."""
    op = "and" if op == "and" else "or"
    if not dim_indices or len(dim_indices) != len(lows) or len(dim_indices) != len(highs):
        return (0, 0)
    dims = [int(d) for d in dim_indices]
    lows_f = [float(x) for x in lows]
    highs_f = [float(x) for x in highs]

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    total = 0
    hit = 0
    for batch in pf.iter_batches(batch_size=2048, columns=keys):
        cols = [batch.column(i) for i in range(batch.num_columns)]
        n = batch.num_rows
        for r in range(n):
            parts = []
            for c in cols:
                try:
                    parts.append(c[r].as_py())
                except Exception:
                    parts.append(None)
            vec = concat_clean_row(parts)

            if op == "and":
                ok = True
                for d, lo, hi in zip(dims, lows_f, highs_f):
                    if d >= len(vec):
                        ok = False
                        break
                    v = vec[d]
                    if v is None:
                        ok = False
                        break
                    try:
                        fv = float(v)
                    except Exception:
                        ok = False
                        break
                    if math.isnan(fv) or math.isinf(fv) or not (lo <= fv <= hi):
                        ok = False
                        break
                if ok:
                    hit += 1
            else:
                ok = False
                for d, lo, hi in zip(dims, lows_f, highs_f):
                    if d >= len(vec):
                        continue
                    v = vec[d]
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if math.isnan(fv) or math.isinf(fv):
                        continue
                    if lo <= fv <= hi:
                        ok = True
                        break
                if ok:
                    hit += 1
            total += 1
    return (hit, total)


def process_episode_frames(dataset_name, episode, base_dir, ep_idx, total_eps, cameras, head_camera):
    """Extract frames for each camera view into frames/<ds>/<ep>/<camera>/."""
    row_count = get_parquet_row_count(episode["parquet_path"])
    video_paths = episode.get("video_paths") or {}
    prefix = f"[{ep_idx}/{total_eps}] "
    migrate_legacy_flat_frames(base_dir, dataset_name, episode["name"], row_count, head_camera)

    for cam in cameras:
        frames_dir = frames_dir_for_camera(base_dir, dataset_name, episode["name"], cam)
        existing_count = count_jpg_files(frames_dir)

        if existing_count == row_count and row_count > 0:
            print(f"  {prefix}Skipping {episode['name']} / {cam} ({existing_count} frames)")
            continue

        if existing_count > 0 and existing_count != row_count:
            print(
                f"  {prefix}Re-extracting {episode['name']} / {cam} (existing: {existing_count}, expected: {row_count})"
            )
            shutil.rmtree(frames_dir)
        else:
            print(f"  {prefix}Extracting {episode['name']} / {cam}: {row_count} frames")

        vpath = video_paths.get(cam)
        if not vpath or not os.path.isfile(vpath):
            print(f"  {prefix}Warning: Video not found for {cam}: {vpath}")
            continue

        extract_frames(vpath, frames_dir, row_count, progress_prefix=f"{prefix}{cam} ")


class APIHandler(BaseHTTPRequestHandler):
    """HTTP handler with API endpoints for on-demand episode parsing."""

    def log_message(self, format, *args):
        """Suppress default logging for static files."""
        if args and isinstance(args[0], str) and args[0].startswith("GET /api/"):
            print(f"[API] {args[0]}")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.parse_qs(parsed.query)

        # API: Get datasets list
        if path == "/api/datasets":
            self.send_json_response(get_datasets_list())
            return

        # API: Paged preview index for thumbnails
        if path == "/api/preview_index":
            offset = _safe_int(query.get("offset", ["0"])[0], 0, 0, None)
            limit = _safe_int(query.get("limit", ["200"])[0], 200, 1, 2000)
            ds_filter = query.get("dataset", [None])[0]
            out = []

            def iter_items():
                """Yield one representative (ds, first_ep) per dataset."""
                for ds in DATASETS_INFO:
                    if ds_filter and os.path.normpath(ds.get("path", "")) != os.path.normpath(ds_filter):
                        continue
                    eps = list(ds.get("episodes") or [])
                    if not eps:
                        continue
                    # Stable "first episode": sort by episode name
                    eps.sort(key=lambda e: str(e.get("name", "")))
                    yield (ds, eps[0])

            i = 0
            for ds, ep in iter_items():
                if i < offset:
                    i += 1
                    continue
                if len(out) >= limit:
                    break
                # pick a stable default camera for preview (prefer center)
                tri = resolve_viewer_camera_triplet(ds["path"])
                cams = [tri[0], tri[1], tri[2]]
                cams = [c for c in cams if c]
                if not cams:
                    continue
                # require at least one existing video (otherwise thumbnail cannot be generated)
                vmap = ep.get("video_paths") or {}
                cam = None
                for c in cams:
                    vp = vmap.get(c)
                    if vp and os.path.isfile(vp):
                        cam = c
                        break
                if not cam:
                    # allow legacy flat pre-extracted frames? not for preview; skip
                    i += 1
                    continue
                out.append(
                    {
                        "dataset_path": ds["path"],
                        "dataset_name": ds["name"],
                        "dataset_frames_key": ds["name"],
                        "episode": ep["name"],
                        "camera": cam,
                    }
                )
                i += 1
            self.send_json_response({"items": out, "offset": offset, "limit": limit, "next_offset": offset + len(out)})
            return

        # API: Paged preview of all episodes for a given dataset (repo_id)
        if path == "/api/preview_episodes":
            ds_name = query.get("dataset", [None])[0]
            if not ds_name:
                self.send_error(400, "Missing dataset parameter")
                return
            offset = _safe_int(query.get("offset", ["0"])[0], 0, 0, None)
            limit = _safe_int(query.get("limit", ["50"])[0], 50, 1, 500)
            ds = resolve_dataset_entry(ds_name)
            if not ds:
                self.send_error(404, "Dataset not found")
                return
            eps = list(ds.get("episodes") or [])
            eps.sort(key=lambda e: str(e.get("name", "")))
            out = []
            i = 0
            for ep in eps:
                if i < offset:
                    i += 1
                    continue
                if len(out) >= limit:
                    break
                tri = resolve_viewer_camera_triplet(ds.get("path") or ds_name)
                cams = [c for c in [tri[0], tri[1], tri[2]] if c]
                vmap = ep.get("video_paths") or {}
                cam = None
                for c in cams:
                    vp = vmap.get(c)
                    if vp and os.path.isfile(vp):
                        cam = c
                        break
                if not cam:
                    i += 1
                    continue
                out.append(
                    {
                        "dataset_path": ds["path"],
                        "dataset_name": ds["name"],
                        "dataset_frames_key": ds["name"],
                        "episode": ep["name"],
                        "camera": cam,
                    }
                )
                i += 1
            self.send_json_response({"items": out, "offset": offset, "limit": limit, "next_offset": offset + len(out)})
            return

        # API: Thumbnail image (frame0) for an episode
        if path == "/api/episode_thumbnail":
            ds_name = query.get("dataset", [None])[0]
            ep_name = query.get("episode", [None])[0]
            camera = query.get("camera", [None])[0]
            if not ds_name or not ep_name:
                self.send_error(400, "Missing dataset or episode parameter")
                return
            ds = resolve_dataset_entry(ds_name)
            if not ds:
                self.send_error(404, "Dataset not found")
                return
            ep_entry = None
            for ep in ds.get("episodes") or []:
                if ep.get("name") == ep_name:
                    ep_entry = ep
                    break
            if not ep_entry:
                self.send_error(404, "Episode not found")
                return
            if not camera:
                tri = resolve_viewer_camera_triplet(ds_name)
                camera = tri[1] or tri[0] or tri[2]
            thumb = ensure_episode_thumbnail(ds, ep_entry, BASE_DIR, camera)
            if not thumb:
                self.send_error(404, "Thumbnail not available")
                return
            self.send_static_file(thumb)
            return

        # API: Triptych preview thumbnail (left|center|right frame0)
        if path == "/api/episode_preview":
            ds_name = query.get("dataset", [None])[0]
            ep_name = query.get("episode", [None])[0]
            if not ds_name or not ep_name:
                self.send_error(400, "Missing dataset or episode parameter")
                return
            ds = resolve_dataset_entry(ds_name)
            if not ds:
                self.send_error(404, "Dataset not found")
                return
            ep_entry = None
            for ep in ds.get("episodes") or []:
                if ep.get("name") == ep_name:
                    ep_entry = ep
                    break
            if not ep_entry:
                self.send_error(404, "Episode not found")
                return
            thumb = ensure_episode_triptych_thumbnail(ds, ep_entry, BASE_DIR)
            if not thumb:
                self.send_error(404, "Preview not available")
                return
            self.send_static_file(thumb)
            return

        # API: Get episode frame count
        if path == "/api/episode_info":
            ds_name = query.get("dataset", [None])[0]
            ep_name = query.get("episode", [None])[0]
            if not ds_name or not ep_name:
                self.send_error(400, "Missing dataset or episode parameter")
                return
            info = get_episode_info(ds_name, ep_name)
            self.send_json_response(info)
            return

        # API: Threshold global stats are handled via POST now (see do_POST).
        if path == "/api/threshold_global_stats":
            self.send_error(405, "Use POST")
            return

        # API: Load episode (extract frames if needed + parse joints), with progress via SSE
        if path == "/api/load_episode":
            ds_name = query.get("dataset", [None])[0]
            ep_name = query.get("episode", [None])[0]
            if not ds_name or not ep_name:
                self.send_error(400, "Missing dataset or episode parameter")
                return
            preview_only = query.get("preview", ["0"])[0] in ("1", "true", "True", "yes", "on")
            self.send_sse_load_episode(ds_name, ep_name, preview_only=preview_only)
            return

        # Static file serving
        if path == "/" or path == "":
            path = "/viewer.html"
        rel = path.lstrip("/")
        # Serve caches from BASE_DIR/.cache while keeping URLs stable (frames/...)
        if rel.startswith("frames/") or rel.startswith("thumbs/") or rel.startswith("thumbs_triptych/"):
            file_path = os.path.join(BASE_DIR, ".cache", rel)
        else:
            file_path = os.path.join(BASE_DIR, rel)
        if os.path.isfile(file_path):
            self.send_static_file(file_path)
        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)

        # API: Threshold global stats across all repo_id (datasets) via JSON payload
        if path == "/api/threshold_global_stats":
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                self.send_error(400, "Invalid JSON")
                return

            mode = payload.get("mode", "state")
            op = payload.get("op", "or")
            dims = payload.get("dims", [])
            lows = payload.get("lows", [])
            highs = payload.get("highs", [])
            if mode not in ("state", "action"):
                self.send_error(400, "Invalid mode")
                return
            op = "and" if op == "and" else "or"
            try:
                dims = [int(x) for x in dims]
                lows = [float(x) for x in lows]
                highs = [float(x) for x in highs]
            except Exception:
                self.send_error(400, "Invalid dims/lows/highs")
                return
            if not dims or len(dims) != len(lows) or len(dims) != len(highs):
                self.send_error(400, "dims/lows/highs length mismatch")
                return
            for lo, hi in zip(lows, highs):
                if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                    self.send_error(400, "Invalid low/high range")
                    return

            cache_key = ("post", mode, tuple(dims), op, tuple(lows), tuple(highs))
            if cache_key in threshold_stats_cache:
                self.send_json_response(threshold_stats_cache[cache_key])
                return

            hit_all = 0
            total_all = 0
            for ds in DATASETS_INFO:
                ds_name = ds.get("path") or ds.get("name")
                info_meta = get_dataset_info_meta(ds_name)
                keys = (
                    list(info_meta.get("state_parquet_keys") or ["observation.state"])
                    if mode == "state"
                    else list(info_meta.get("action_parquet_keys") or ["action"])
                )
                for ep in (ds.get("episodes") or []):
                    parquet_path = ep.get("parquet_path")
                    if not parquet_path or not os.path.isfile(parquet_path):
                        continue
                    try:
                        avail = parquet_column_set(parquet_path)
                        if any(k not in avail for k in keys):
                            continue
                        h, t = count_parquet_constraints(parquet_path, keys, dims, lows, highs, op)
                        hit_all += h
                        total_all += t
                    except Exception:
                        continue

            out = {
                "mode": mode,
                "op": op,
                "dims": dims,
                "lows": lows,
                "highs": highs,
                "in_range": int(hit_all),
                "total": int(total_all),
            }
            threshold_stats_cache[cache_key] = out
            self.send_json_response(out)
            return

        # API: Static (velocity-based) global stats across all repo_id (datasets)
        if path == "/api/static_global_stats":
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                self.send_error(400, "Invalid JSON")
                return

            try:
                cfg = payload.get("cfg", {}) or {}
                fps = int(cfg.get("fps", 30) or 30)
                joint_vth = float(cfg.get("joint_velocity_threshold", 0.1) or 0.1)
                gripper_vth = float(cfg.get("gripper_velocity_threshold", 0.2) or 0.2)
                smoothing_h = int(cfg.get("smoothing_half_window", 2) or 2)
            except Exception:
                self.send_error(400, "Invalid cfg")
                return
            if fps <= 0 or smoothing_h < 0 or joint_vth < 0 or gripper_vth < 0:
                self.send_error(400, "Invalid cfg values")
                return

            cache_key = (fps, float(joint_vth), float(gripper_vth), smoothing_h)
            if cache_key in static_stats_cache:
                self.send_json_response(static_stats_cache[cache_key])
                return

            hit_all = 0
            total_all = 0
            for ds in DATASETS_INFO:
                ds_name = ds.get("path") or ds.get("name")
                info_meta = get_dataset_info_meta(ds_name)
                state_keys = list(info_meta.get("state_parquet_keys") or ["observation.state"])
                for ep in (ds.get("episodes") or []):
                    parquet_path = ep.get("parquet_path")
                    if not parquet_path or not os.path.isfile(parquet_path):
                        continue
                    try:
                        avail = parquet_column_set(parquet_path)
                        if any(k not in avail for k in state_keys):
                            continue
                        h, t = count_parquet_static_frames(
                            parquet_path,
                            state_keys,
                            {
                                "fps": fps,
                                "joint_velocity_threshold": joint_vth,
                                "gripper_velocity_threshold": gripper_vth,
                                "smoothing_half_window": smoothing_h,
                            },
                        )
                        hit_all += h
                        total_all += t
                    except Exception:
                        continue

            out = {"in_static": int(hit_all), "total": int(total_all)}
            static_stats_cache[cache_key] = out
            self.send_json_response(out)
            return

        self.send_error(404, "Not found")

    def send_json_response(self, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_sse_load_episode(self, ds_name, ep_name, preview_only: bool = False):
        """Send Server-Sent Events.

        - preview_only=True: only ensure frame_000000.jpg for each camera, but still parse ALL parquet rows
        - preview_only=False: ensure all frames and parse all rows (full view)
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def send_event(event, data):
            msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
            self.wfile.write(msg.encode("utf-8"))
            self.wfile.flush()

        # Find episode paths (dataset param = full path or legacy basename)
        parquet_path = None
        video_paths = {}
        ds = resolve_dataset_entry(ds_name)
        if ds:
            for ep in ds["episodes"]:
                if ep["name"] == ep_name:
                    parquet_path = ep["parquet_path"]
                    video_paths = ep.get("video_paths") or {}
                    break

        if not parquet_path:
            send_event("error", {"message": "Episode not found"})
            return

        # Basename for frames/<ds>/<ep>/ (must not be a full path — unsafe for os.path.join)
        frames_key = ds["name"]
        canon_id = canonical_dataset_id(ds_name) or frames_key

        info_meta = get_dataset_info_meta(ds_name)
        cam_triplet = resolve_viewer_camera_triplet(ds_name)
        cam_labels = triplet_labels_and_meta(cam_triplet)

        # Determine total frames early (prefer parquet row count; fallback to video for empty parquet)
        try:
            frame_count = get_parquet_row_count(parquet_path)
        except Exception as e:
            send_event("error", {"message": f"Failed to read parquet metadata: {e}"})
            return

        # Some datasets ship empty parquet (0 rows) but have videos; fall back to video frame count.
        if frame_count <= 0:
            probe_candidates = []
            # Prefer center cam, then any available cam
            if cam_triplet[1]:
                probe_candidates.append(cam_triplet[1])
            probe_candidates.extend([c for c in cam_triplet if c and c not in probe_candidates])
            for cam in probe_candidates:
                vpath = (video_paths or {}).get(cam)
                n = video_frame_count_ffprobe(vpath) if vpath else None
                if n and n > 0:
                    frame_count = n
                    break
        center_for_legacy = cam_triplet[1] or HEAD_CAMERA_NAME
        send_event(
            "meta",
            {
                "frame_count": frame_count,
                "cameras": list(cam_triplet),
                "camera_labels": cam_labels,
                "dataset_frames_key": frames_key,
                "preview_only": bool(preview_only),
                "tasks": get_episode_tasks(ds_name, ep_name),
                "robot_type": info_meta.get("robot_type"),
                "state_names": info_meta.get("state_names") or [],
                "action_names": info_meta.get("action_names") or [],
                "state_parquet_keys": info_meta.get("state_parquet_keys") or [],
                "action_parquet_keys": info_meta.get("action_parquet_keys") or [],
            },
        )

        lock = get_episode_lock(canon_id, ep_name)
        with lock:
            migrate_legacy_flat_frames(BASE_DIR, frames_key, ep_name, frame_count, center_for_legacy)

            def parse_state_action_all_rows():
                # Stage 2: parse state + action + (action - state) (cached per episode)
                cache_key = (canon_id, ep_name)
                if cache_key in episode_cache:
                    send_event("progress", {"percent": 100, "phase": "parse", "status": "cached"})
                    send_event("done", {"data": episode_cache[cache_key]})
                    return True

                send_event("progress", {"percent": 70, "phase": "parse", "status": "reading parquet"})
                try:
                    state_keys = list(info_meta.get("state_parquet_keys") or ["observation.state"])
                    action_keys = list(info_meta.get("action_parquet_keys") or ["action"])
                    cols_needed = list(dict.fromkeys(state_keys + action_keys))
                    avail = parquet_column_set(parquet_path)
                    missing = [c for c in cols_needed if c not in avail]
                    if missing:
                        send_event(
                            "error",
                            {"message": f"Parquet 缺少列（与 info.json 中 state/action 相关键不一致）: {missing}"},
                        )
                        return False

                    df = pd.read_parquet(parquet_path, columns=cols_needed)
                    total_rows = len(df)
                    state_cols = [df[k].tolist() for k in state_keys]
                    action_cols = [df[k].tolist() for k in action_keys]

                    send_event(
                        "progress",
                        {"percent": 75, "phase": "parse", "status": f"parsing {total_rows} frames (state+action)"},
                    )

                    clean_states = []
                    clean_actions = []
                    clean_diff = []
                    for i in range(total_rows):
                        s_row = concat_clean_row([state_cols[j][i] for j in range(len(state_keys))])
                        a_row = concat_clean_row([action_cols[j][i] for j in range(len(action_keys))])
                        clean_states.append(s_row)
                        clean_actions.append(a_row)
                        clean_diff.append(vector_diff_row(a_row, s_row))
                        if (i + 1) % 200 == 0:
                            pct_stage = int(100 * (i + 1) / total_rows) if total_rows > 0 else 100
                            overall = int(65 + 35 * pct_stage / 100)
                            send_event(
                                "progress",
                                {"percent": overall, "phase": "parse", "status": f"parse {i+1}/{total_rows}"},
                            )

                    bundle = {"state": clean_states, "action": clean_actions, "action_state": clean_diff}
                    send_event("progress", {"percent": 98, "phase": "parse", "status": "caching"})
                    episode_cache[cache_key] = bundle
                    send_event("progress", {"percent": 100, "phase": "parse", "status": "done"})
                    send_event("done", {"data": bundle})
                    return True
                except Exception as e:
                    send_event("error", {"message": str(e)})
                    return False

            if preview_only:
                # Fast path: ensure only the first frame exists for each camera.
                send_event("progress", {"percent": 0, "phase": "extract", "status": "extracting first frame"})
                for cam in [c for c in cam_triplet if c]:
                    vpath = video_paths.get(cam)
                    if not vpath or not os.path.isfile(vpath):
                        continue
                    frames_dir = frames_dir_for_camera(BASE_DIR, frames_key, ep_name, cam)
                    f0 = os.path.join(frames_dir, "frame_000000.jpg")
                    if os.path.isfile(f0):
                        continue
                    try:
                        extract_first_frame(vpath, f0)
                    except Exception as e:
                        send_event("error", {"message": f"First-frame extraction failed ({cam}): {e}"})
                        return
                send_event("progress", {"percent": 65, "phase": "extract", "status": "first frame ready"})

                # Parse ALL rows (so action/state curves are complete), but keep images as first-frame only.
                parse_state_action_all_rows()
                return

            # Cameras that still need full extraction (skip missing videos → blank panel)
            def needs_extract(cam):
                d = frames_dir_for_camera(BASE_DIR, frames_key, ep_name, cam)
                return frame_count > 0 and count_jpg_files(d) != frame_count

            cameras_todo = []
            for c in cam_triplet:
                if not c:
                    continue
                vpath = video_paths.get(c)
                if not vpath or not os.path.isfile(vpath):
                    continue
                if needs_extract(c):
                    cameras_todo.append(c)

            # Stage 1: extract frames per camera if needed
            if frame_count > 0 and cameras_todo:
                send_event("progress", {"percent": 0, "phase": "extract", "status": "extracting frames"})

                n_cam = len(cameras_todo)
                for ci, cam in enumerate(cameras_todo):
                    vpath = video_paths.get(cam)
                    if not vpath or not os.path.isfile(vpath):
                        continue

                    frames_dir = frames_dir_for_camera(BASE_DIR, frames_key, ep_name, cam)
                    existing_count = count_jpg_files(frames_dir)
                    if existing_count > 0 and existing_count != frame_count:
                        try:
                            shutil.rmtree(frames_dir)
                        except Exception:
                            pass

                    def make_cb(camera_name, cam_index, total_cams):
                        def cb(p):
                            cur = int(p.get("current", 0) or 0)
                            tot = int(p.get("total", frame_count) or frame_count or 1)
                            local = (cur / tot) if tot > 0 else 1.0
                            # 0-65% total for all camera extractions
                            overall = int(65 * (cam_index + local) / total_cams)
                            overall = min(65, overall)
                            status = f"extract {camera_name} {cur}/{tot}"
                            fps = p.get("fps", None)
                            if fps:
                                status += f" | {fps:.0f} fps"
                            eta = p.get("eta_sec", None)
                            if eta is not None and eta >= 0:
                                status += f" | ETA {format_time(eta)}"
                            send_event("progress", {"percent": overall, "phase": "extract", "status": status})

                        return cb

                    try:
                        extract_frames_with_callback(
                            vpath,
                            frames_dir,
                            frame_count,
                            progress_cb=make_cb(cam, ci, n_cam),
                        )
                    except Exception as e:
                        send_event("error", {"message": f"Frame extraction failed ({cam}): {e}"})
                        return

                send_event("progress", {"percent": 65, "phase": "extract", "status": "frames ready"})
            else:
                send_event("progress", {"percent": 65, "phase": "extract", "status": "frames ready"})

            parse_state_action_all_rows()

    def send_static_file(self, file_path):
        """Serve static files with proper MIME types."""
        mime_types = {
            ".html": "text/html",
            ".js": "application/javascript",
            ".css": "text/css",
            ".json": "application/json",
            ".jpg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
        }
        ext = os.path.splitext(file_path)[1].lower()
        mime = mime_types.get(ext, "application/octet-stream")

        with open(file_path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)


def get_datasets_list():
    """Return list of datasets with episode names (path is unique key for the viewer)."""
    return [
        {
            "name": ds["name"],
            "path": ds["path"],
            "episodes": [ep["name"] for ep in ds["episodes"]],
        }
        for ds in DATASETS_INFO
    ]


def get_episode_info(ds_name, ep_name):
    """Return frame count for an episode."""
    ds = resolve_dataset_entry(ds_name)
    if not ds:
        return {"frame_count": 0}
    for ep in ds["episodes"]:
        if ep["name"] == ep_name:
            row_count = get_parquet_row_count(ep["parquet_path"])
            return {"frame_count": row_count}
    return {"frame_count": 0}


def generate_html(datasets):
    """Generate viewer.html that loads data on-demand via API."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>P Visualizer</title>
<style>
  body { font-family: sans-serif; max-width: 1680px; margin: 0 auto; padding: 16px; background: #f5f5f5; }
  h1 { text-align: center; color: #333; }
  .controls { display: flex; gap: 16px; align-items: center; margin: 16px 0; flex-wrap: wrap; }
  .controls label { font-weight: bold; }
  .controls select { padding: 6px 12px; font-size: 14px; }
  .frame-triptych { display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: center; gap: 12px; margin: 12px 0; }
  .frame-panel { margin: 0; text-align: center; flex: 1 1 200px; max-width: 420px; min-width: 160px; }
  .frame-panel figcaption { font-size: 12px; color: #555; margin-bottom: 6px; word-break: break-all; }
  .frame-panel img {
    display: block;
    width: 100%;
    aspect-ratio: 4 / 3;
    object-fit: contain;
    border: 1px solid #ccc;
    background: #222;
  }
  .slider-container { margin: 12px 0; text-align: center; }
  #frame-slider { width: 100%; max-width: 800px; }
  #frame-counter { font-size: 16px; font-weight: bold; margin: 8px 0; }
  .arm-label { font-size: 14px; font-weight: bold; color: #555; margin: 12px 0 4px 0; }
  .canvas-row { display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px; }
  .canvas-cell { text-align: center; }
  .canvas-cell canvas { width: 100%; aspect-ratio: 2/1; border: 1px solid #ddd; background: #fff; }
  .canvas-cell .joint-title { font-size: 11px; color: #666; margin-top: 2px; }
  #progress-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.5); display: none; justify-content: center; align-items: center;
    z-index: 1000;
  }
  #progress-box {
    background: white; padding: 24px 32px; border-radius: 8px; text-align: center; min-width: 300px;
  }
  #progress-bar-bg { width: 100%; height: 20px; background: #eee; border-radius: 4px; margin: 12px 0; }
  #progress-bar-fill { height: 100%; background: #4a90d9; border-radius: 4px; width: 0%; transition: width 0.2s; }
  #progress-text { font-size: 14px; color: #666; }
</style>
</head>
<body>
<h1>Episode Frame Visualizer</h1>
<div class="controls">
  <label>Dataset:</label>
  <select id="dataset-select"><option value="">Loading...</option></select>
  <label>Episode:</label>
  <select id="episode-select"><option value="">Select dataset first</option></select>
</div>
<div id="robot-type-bar" style="display:none; font-size:13px; color:#333; margin:8px 0 0 0; font-weight:600;"></div>
<div id="prompt-box" style="margin: 10px 0 0 0; background: #fff; border: 1px solid #ddd; padding: 10px 12px; border-radius: 6px;">
  <div style="font-weight: 700; color: #333; margin-bottom: 6px;">Prompt (tasks)</div>
  <div id="prompt-text" style="white-space: pre-wrap; font-family: ui-monospace, Consolas, monospace; font-size: 12px; color: #222;"></div>
</div>
<div class="slider-container">
  <div id="frame-counter">Frame 0 / 0</div>
  <input type="range" id="frame-slider" min="0" max="0" value="0" disabled>
  <div style="margin-top:8px; display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
    <button id="load-all-frames-btn" type="button" style="display:none; font-size:12px; padding:6px 10px; border:1px solid #bbb; border-radius:8px; background:#fff; cursor:pointer;">
      加载所有帧
    </button>
    <div id="preview-hint" style="display:none; font-size:12px; color:#666; align-self:center;">
      当前仅加载首帧以加快预览
    </div>
  </div>
</div>
<div class="frame-triptych">
  <figure class="frame-panel"><figcaption id="cam-caption-left"><span id="cam-label-left">左</span></figcaption><img id="frame-image-left" src="" alt=""></figure>
  <figure class="frame-panel frame-head">
    <figcaption style="display:flex; align-items:center; justify-content:space-between; gap:8px;">
      <span id="cam-label-center">中</span>
      <button id="rotate-head-btn" type="button" style="font-size:12px; padding:2px 8px; border:1px solid #bbb; border-radius:999px; background:#fff; cursor:pointer;">Rotate 180°</button>
    </figcaption>
    <img id="frame-image" src="" alt="">
  </figure>
  <figure class="frame-panel"><figcaption id="cam-caption-right"><span id="cam-label-right">右</span></figcaption><img id="frame-image-right" src="" alt=""></figure>
</div>
<div class="controls" id="pairing-controls" style="margin-top: 6px;">
  <label>配对逻辑:</label>
  <select id="pairing-select">
    <option value="index">按维度 index</option>
    <option value="name">按名字精确匹配</option>
    <option value="similarity">按名字相似度</option>
  </select>
  <span id="pairing-sim-box" style="display:none; font-size:12px; color:#444;">
    阈值 <input id="pairing-sim-threshold" type="number" min="0" max="1" step="0.05" value="0.7" style="width:64px; padding:2px 6px; font-size:12px;">
  </span>
</div>
<div id="canvas-container">
  <div id="curve-legend" style="font-size:12px;color:#666;margin:8px 0 4px 0;">
    <span style="color:#4a90d9;font-weight:600">S</span> state &nbsp;
    <span style="color:#2d8f47;font-weight:600">A</span> action &nbsp;
    <span style="color:#8B4513;font-weight:600">Δ</span> action−state
  </div>
  <div id="joint-charts-root"></div>
</div>

<div class="controls" style="margin-top: 8px; flex-direction:column; align-items:stretch; gap:8px;">
  <div style="display:flex; flex-direction:column; gap:4px; max-width:420px;">
    <label style="font-weight:700;">预处理器</label>
    <select id="preproc-type-select">
      <option value="none">无</option>
      <option value="joint_threshold">关节阈值筛选</option>
      <option value="static_velocity">静止帧（速度阈值）</option>
    </select>
  </div>
  <div id="preproc-joint-panel" style="display:none; flex-direction:column; gap:8px; width:100%;">
    <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
      <label style="font-size:12px; color:#444;">约束关系</label>
      <select id="threshold-op-select">
        <option value="or">或</option>
        <option value="and">且</option>
      </select>
      <button id="threshold-apply" type="button" style="font-size:12px; padding:6px 10px; border:1px solid #bbb; border-radius:8px; background:#fff; cursor:pointer;">
        应用
      </button>
      <button id="threshold-clear" type="button" style="font-size:12px; padding:6px 10px; border:1px solid #bbb; border-radius:8px; background:#fff; cursor:pointer;">
        清除
      </button>
    </div>
    <button id="threshold-add" type="button" style="align-self:flex-start; font-size:12px; padding:6px 10px; border:1px solid #bbb; border-radius:8px; background:#fff; cursor:pointer;">
      + 添加关节约束
    </button>
  </div>
  <div id="preproc-static-panel" style="display:none; flex-direction:column; gap:8px; width:100%; max-width:920px;">
    <div style="font-size:12px; color:#555; line-height:1.45;">
      与训练侧 <code>VelocityBasedStaticDetector</code> 一致：对 <code>observation.state</code> 逐维求时间梯度（除以 fps），再按半窗做滑动平均；每帧若所有维度 |速度| 均低于对应阈值则判为静止帧（state_dim=14 时夹爪维用单独阈值；非 14 维时每维均使用「关节速度阈值」）。
    </div>
    <div style="display:flex; flex-wrap:wrap; gap:10px; align-items:center; font-size:12px;">
      <label>fps <input id="static-fps" type="number" min="1" step="1" value="30" style="width:56px; padding:2px 6px;"></label>
      <label>关节速度阈值 <input id="static-joint-vth" type="number" min="0" step="0.01" value="0.1" style="width:72px; padding:2px 6px;"></label>
      <label>夹爪速度阈值 <input id="static-gripper-vth" type="number" min="0" step="0.01" value="0.2" style="width:72px; padding:2px 6px;"></label>
      <label>平滑半窗 <input id="static-smooth-h" type="number" min="0" step="1" value="2" style="width:48px; padding:2px 6px;"></label>
      <button id="static-apply" type="button" style="font-size:12px; padding:6px 10px; border:1px solid #bbb; border-radius:8px; background:#fff; cursor:pointer;">应用</button>
    </div>
  </div>
</div>
<div id="threshold-constraints" style="margin-top:6px; display:none; flex-direction:column; gap:6px;"></div>
<div style="display:flex; justify-content:center; gap:14px; flex-wrap:wrap; margin: 10px auto 0 auto; width:min(1200px, 90vw);">
  <div style="background:#fff; border:1px solid #e3e3e3; border-radius:10px; padding:14px 16px; min-width: 360px; flex:1 1 420px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); text-align:center;">
    <div style="font-weight:700; color:#333; font-size:12px; margin-bottom:6px;">本 episode：区间帧占比</div>
    <canvas id="pie-episode" width="220" height="220" style="width:220px; height:220px; display:block; margin:0 auto;"></canvas>
    <div id="pie-episode-text" style="font-size:12px; color:#444; margin-top:6px;">-</div>
  </div>
  <div style="background:#fff; border:1px solid #e3e3e3; border-radius:10px; padding:14px 16px; min-width: 360px; flex:1 1 420px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); text-align:center;">
    <div style="font-weight:700; color:#333; font-size:12px; margin-bottom:6px;">所有 repo_id：区间帧占比</div>
    <canvas id="pie-global" width="220" height="220" style="width:220px; height:220px; display:block; margin:0 auto;"></canvas>
    <div id="pie-global-text" style="font-size:12px; color:#444; margin-top:6px;">-</div>
  </div>
</div>

<div id="preview-root" style="margin-top: 18px;">
  <div class="controls" style="justify-content: space-between; width:100%; gap:12px;">
    <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
      <label style="display:flex; align-items:center; gap:8px; font-weight:700;">
        <input id="preview-toggle" type="checkbox">
        预览 repo_id（每个 repo 的第一个 episode 首帧）
      </label>
      <label style="display:flex; align-items:center; gap:8px; font-weight:700;">
        <input id="preview-current-toggle" type="checkbox">
        预览当前 repo_id 的所有 episode
      </label>
      <span style="font-size:12px; color:#666;">点击缩略图可直接加载该 episode</span>
    </div>
    <button id="preview-load-more" type="button" style="display:none; font-size:12px; padding:6px 10px; border:1px solid #bbb; border-radius:8px; background:#fff; cursor:pointer;">加载更多</button>
  </div>
  <div id="preview-grid" style="display:none; margin-top:10px; display:grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap:10px;"></div>
</div>

<div id="progress-overlay">
  <div id="progress-box">
    <div id="progress-title">Loading episode data...</div>
    <div id="progress-bar-bg"><div id="progress-bar-fill"></div></div>
    <div id="progress-text">0%</div>
  </div>
</div>

<script>
let JOINT_BUNDLE = null;
let JOINT_DIM_COUNT = 0;
let CHART_DEFS = []; // [{id:number, name:string, stateDim:number|null, actionDim:number|null}]
let jointStateNames = [];
let jointActionNames = [];
let currentDataset = '';
/** Matches frames/<this>/<episode>/ on disk; basename, not full dataset path */
let datasetFramesKey = '';
let currentEpisode = '';
let currentFrame = 0;
let totalFrames = 0;
let previewOnlyImages = false;
let currentLoadToken = 0;
let pendingFrameRetry = null;
const curvePoints = {};
let pairingMode = 'index'; // 'index' | 'name' | 'similarity'
let pairingSimThreshold = 0.7;
let thresholdCfg = {enabled: false, op: 'or', constraints: []}; // constraints: [{defId, mode, low, high}]
let preprocType = 'none'; // 'none' | 'joint_threshold' | 'static_velocity'
/** Same defaults as VelocityBasedStaticDetector (static_detector.py) */
let staticCfg = {fps: 30, joint_velocity_threshold: 0.1, gripper_velocity_threshold: 0.2, smoothing_half_window: 2};
let thresholdMask = null; // boolean[] aligned to JOINT_BUNDLE frames (combined)
let thresholdMasksByDefId = {}; // {defId: boolean[]} union mask for highlight per joint
let isDraggingMarker = false;

const datasetSelect = document.getElementById('dataset-select');
const episodeSelect = document.getElementById('episode-select');
const pairingSelect = document.getElementById('pairing-select');
const pairingSimBox = document.getElementById('pairing-sim-box');
const pairingSimThresholdInput = document.getElementById('pairing-sim-threshold');
const pairingControls = document.getElementById('pairing-controls');
const previewToggle = document.getElementById('preview-toggle');
const previewCurrentToggle = document.getElementById('preview-current-toggle');
const previewGrid = document.getElementById('preview-grid');
const previewLoadMoreBtn = document.getElementById('preview-load-more');
const frameSlider = document.getElementById('frame-slider');
const frameCounter = document.getElementById('frame-counter');
const loadAllFramesBtn = document.getElementById('load-all-frames-btn');
const previewHint = document.getElementById('preview-hint');
const promptText = document.getElementById('prompt-text');
const frameImageLeft = document.getElementById('frame-image-left');
const frameImage = document.getElementById('frame-image');
const frameImageRight = document.getElementById('frame-image-right');
const rotateHeadBtn = document.getElementById('rotate-head-btn');
const LEGACY_VIEWER_CAMERAS = ['observation.images.left_wrist', 'observation.images.head', 'observation.images.right_wrist'];
let viewerCameras = [null, null, null];
const progressOverlay = document.getElementById('progress-overlay');
const progressBarFill = document.getElementById('progress-bar-fill');
const progressText = document.getElementById('progress-text');
const robotTypeBar = document.getElementById('robot-type-bar');
const jointChartsRoot = document.getElementById('joint-charts-root');
const thresholdOpSelect = document.getElementById('threshold-op-select');
const preprocTypeSelect = document.getElementById('preproc-type-select');
const preprocJointPanel = document.getElementById('preproc-joint-panel');
const preprocStaticPanel = document.getElementById('preproc-static-panel');
const staticFpsInput = document.getElementById('static-fps');
const staticJointVthInput = document.getElementById('static-joint-vth');
const staticGripperVthInput = document.getElementById('static-gripper-vth');
const staticSmoothHInput = document.getElementById('static-smooth-h');
const staticApplyBtn = document.getElementById('static-apply');
const thresholdConstraintsRoot = document.getElementById('threshold-constraints');
const thresholdAddBtn = document.getElementById('threshold-add');
const thresholdApplyBtn = document.getElementById('threshold-apply');
const thresholdClearBtn = document.getElementById('threshold-clear');
const pieEpisode = document.getElementById('pie-episode');
const pieGlobal = document.getElementById('pie-global');
const pieEpisodeText = document.getElementById('pie-episode-text');
const pieGlobalText = document.getElementById('pie-global-text');

function segmentForFrameUrls() {
  if (datasetFramesKey) return datasetFramesKey;
  const d = currentDataset;
  if (!d) return '';
  const t = d.replace(/\\/+$/, '');
  const i = t.lastIndexOf('/');
  return i >= 0 ? t.slice(i + 1) : t;
}

function _isFiniteNumber(x) {
  return typeof x === 'number' && isFinite(x);
}

function drawPie(canvas, inRange, total) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const r = Math.min(w, h) * 0.45;
  const cx = w / 2, cy = h / 2;
  const safeTotal = Math.max(0, total || 0);
  const safeIn = Math.max(0, Math.min(inRange || 0, safeTotal));
  const frac = safeTotal > 0 ? (safeIn / safeTotal) : 0;

  ctx.beginPath();
  ctx.fillStyle = '#eef1f4';
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.closePath();
  ctx.fill();

  ctx.beginPath();
  // Sophisticated palette: deep teal vs cool gray
  ctx.fillStyle = 'rgba(16, 92, 97, 0.92)'; // deep teal
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2);
  ctx.closePath();
  ctx.fill();

  ctx.beginPath();
  ctx.fillStyle = '#fbfbfc';
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, r * 0.62, 0, Math.PI * 2);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = '#1f2a33';
  ctx.font = '700 16px ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText((frac * 100).toFixed(1) + '%', cx, cy);
}

/** Per-DOF velocity cap; matches build_velocity_threshold (utils.py) for 14-DOF bimanual. */
function buildVelocityThresholdJs(stateDim, jointTh, gripTh) {
  if (stateDim === 14) {
    const out = [];
    for (let i = 0; i < 6; i++) out.push(jointTh);
    out.push(gripTh);
    for (let i = 0; i < 6; i++) out.push(jointTh);
    out.push(gripTh);
    return out;
  }
  return Array.from({length: stateDim}, () => jointTh);
}

/** First derivative along time with spacing dt (same structure as np.gradient(..., axis=0)). */
function gradient1AlongTime(f, dt) {
  const n = f.length;
  const g = new Array(n);
  if (n === 0) return g;
  if (n === 1) {
    g[0] = 0;
    return g;
  }
  g[0] = (f[1] - f[0]) / dt;
  for (let i = 1; i < n - 1; i++) g[i] = (f[i + 1] - f[i - 1]) / (2 * dt);
  g[n - 1] = (f[n - 1] - f[n - 2]) / dt;
  return g;
}

/** Moving average along axis 0; matches preprocessors.utils.moving_average(..., axis=0). */
function movingAverageAxis0(matrix, h) {
  const T = matrix.length;
  const D = T ? matrix[0].length : 0;
  const out = Array.from({length: T}, () => new Array(D).fill(0));
  if (T === 0 || D === 0) return out;
  const hh = Math.max(0, h | 0);
  for (let d = 0; d < D; d++) {
    const cum = new Array(T);
    cum[0] = matrix[0][d];
    for (let i = 1; i < T; i++) cum[i] = cum[i - 1] + matrix[i][d];
    for (let i = 0; i < T; i++) {
      const right = Math.min(T - 1, i + hh);
      const left = Math.max(0, i - hh);
      const windowSum = cum[right] - (left > 0 ? cum[left - 1] : 0);
      const windowSize = right - left + 1;
      out[i][d] = windowSum / windowSize;
    }
  }
  return out;
}

/** Smoothed joint velocities; matches compute_smoothed_velocities (utils.py). */
function computeSmoothedVelocitiesJs(states, fps, smoothingHalfWindow) {
  const dt = 1.0 / Math.max(1, fps);
  const T = states.length;
  if (T === 0) return [];
  const D = states[0].length;
  const raw = Array.from({length: T}, () => new Array(D).fill(0));
  for (let d = 0; d < D; d++) {
    const col = new Array(T);
    for (let i = 0; i < T; i++) col[i] = Number(states[i][d]);
    const g = gradient1AlongTime(col, dt);
    for (let i = 0; i < T; i++) raw[i][d] = g[i];
  }
  return movingAverageAxis0(raw, smoothingHalfWindow);
}

/** Per-frame is_static; matches _compute_is_static_episode (static_detector.py). */
function computeStaticMaskFromStates(states, fps, smoothingHalfWindow, jointTh, gripTh) {
  const T = states.length;
  if (T === 0) return [];
  const D = states[0].length;
  const th = buildVelocityThresholdJs(D, jointTh, gripTh);
  const vel = computeSmoothedVelocitiesJs(states, fps, smoothingHalfWindow);
  const mask = new Array(T);
  for (let i = 0; i < T; i++) {
    let ok = true;
    const row = vel[i];
    for (let d = 0; d < D; d++) {
      const v = row[d];
      if (!Number.isFinite(v) || !Number.isFinite(th[d])) {
        ok = false;
        break;
      }
      if (Math.abs(v) >= th[d]) {
        ok = false;
        break;
      }
    }
    mask[i] = ok;
  }
  return mask;
}

function computeStaticMask() {
  thresholdMask = null;
  thresholdMasksByDefId = {};
  if (preprocType !== 'static_velocity' || !JOINT_BUNDLE || !JOINT_BUNDLE.state || !JOINT_BUNDLE.state.length) return;
  const states = JOINT_BUNDLE.state;
  thresholdMask = computeStaticMaskFromStates(
    states,
    staticCfg.fps,
    staticCfg.smoothing_half_window,
    staticCfg.joint_velocity_threshold,
    staticCfg.gripper_velocity_threshold
  );
  const m = thresholdMask;
  for (const def of CHART_DEFS || []) {
    thresholdMasksByDefId[String(def.id)] = m;
  }
}

function recomputePreprocessorMask() {
  thresholdMask = null;
  thresholdMasksByDefId = {};
  if (!JOINT_BUNDLE || !JOINT_BUNDLE.state || !JOINT_BUNDLE.state.length) return;
  if (preprocType === 'joint_threshold') computeThresholdMask();
  else if (preprocType === 'static_velocity') computeStaticMask();
}

function computeThresholdMask() {
  thresholdMask = null;
  thresholdMasksByDefId = {};
  if (!thresholdCfg.enabled || !JOINT_BUNDLE) return;
  const op = thresholdCfg.op === 'and' ? 'and' : 'or';
  const cs = Array.isArray(thresholdCfg.constraints) ? thresholdCfg.constraints : [];
  if (!cs.length) return;

  let combined = null;
  for (const c of cs) {
    const did = c && c.defId != null ? String(c.defId) : '';
    if (!did) continue;
    const def = CHART_DEFS.find(d => String(d.id) === did);
    if (!def) continue;
    const low = parseFloat(c.low);
    const high = parseFloat(c.high);
    const mode = (c.mode === 'action') ? 'action' : 'state';
    if (!_isFiniteNumber(low) || !_isFiniteNumber(high) || low > high) continue;

    let series = null;
    if (mode === 'state' && def.stateDim != null) series = JOINT_BUNDLE.state.map(row => row[def.stateDim]);
    else if (mode === 'action' && def.actionDim != null) series = JOINT_BUNDLE.action.map(row => row[def.actionDim]);
    if (!series) continue;

    const mask = series.map(v => (_isFiniteNumber(v) && v >= low && v <= high));

    if (!thresholdMasksByDefId[did]) thresholdMasksByDefId[did] = mask.slice();
    else {
      const u = thresholdMasksByDefId[did];
      for (let i = 0; i < u.length; i++) u[i] = u[i] || mask[i];
    }

    if (combined == null) combined = mask;
    else {
      for (let i = 0; i < combined.length; i++) {
        combined[i] = (op === 'and') ? (combined[i] && mask[i]) : (combined[i] || mask[i]);
      }
    }
  }
  thresholdMask = combined;
}

function updateThresholdEpisodePie() {
  if (!pieEpisodeText) return;
  const episodePieActive =
    (preprocType === 'static_velocity' && Array.isArray(thresholdMask) && thresholdMask.length > 0) ||
    (preprocType === 'joint_threshold' && thresholdCfg.enabled && Array.isArray(thresholdMask) && thresholdMask.length > 0);
  if (!episodePieActive) {
    drawPie(pieEpisode, 0, 0);
    pieEpisodeText.textContent = '-';
    return;
  }
  const total = thresholdMask.length;
  const hit = thresholdMask.reduce((acc, b) => acc + (b ? 1 : 0), 0);
  drawPie(pieEpisode, hit, total);
  pieEpisodeText.textContent = `${hit} / ${total}`;
}

async function updateThresholdGlobalPie() {
  if (!pieGlobalText) return;
  if (preprocType === 'static_velocity') {
    pieGlobalText.textContent = '计算中...';
    try {
      const payload = { cfg: staticCfg };
      const res = await fetch('/api/static_global_stats', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error('http ' + res.status);
      const data = await res.json();
      const hit = data.in_static || 0;
      const total = data.total || 0;
      drawPie(pieGlobal, hit, total);
      pieGlobalText.textContent = `${hit} / ${total}`;
    } catch (e) {
      drawPie(pieGlobal, 0, 0);
      pieGlobalText.textContent = '计算失败';
    }
    return;
  }
  if (!thresholdCfg.enabled) {
    drawPie(pieGlobal, 0, 0);
    pieGlobalText.textContent = '-';
    return;
  }
  // Needs chart defs from loaded episode to map defId -> dim index
  if (!CHART_DEFS || !Array.isArray(CHART_DEFS) || CHART_DEFS.length === 0) {
    drawPie(pieGlobal, 0, 0);
    pieGlobalText.textContent = '-';
    return;
  }
  const cs = Array.isArray(thresholdCfg.constraints) ? thresholdCfg.constraints : [];
  if (!cs.length) return;
  const modes = new Set(cs.map(c => (c && c.mode === 'action') ? 'action' : 'state'));
  if (modes.size > 1) {
    drawPie(pieGlobal, 0, 0);
    pieGlobalText.textContent = '全局统计暂不支持混合 state/action';
    return;
  }
  const mode = Array.from(modes)[0] || 'state';
  const dims = [];
  const lows = [];
  const highs = [];
  for (const c of cs) {
    const did = c && c.defId != null ? String(c.defId) : '';
    const def = CHART_DEFS.find(d => String(d.id) === did);
    if (!def) continue;
    const dim = (mode === 'state') ? def.stateDim : def.actionDim;
    if (dim == null) continue;
    const low = parseFloat(c.low);
    const high = parseFloat(c.high);
    if (!_isFiniteNumber(low) || !_isFiniteNumber(high) || low > high) continue;
    dims.push(dim); lows.push(low); highs.push(high);
  }
  if (!dims.length) {
    drawPie(pieGlobal, 0, 0);
    pieGlobalText.textContent = '无可用约束（维度/区间无效）';
    return;
  }
  pieGlobalText.textContent = '计算中...';
  try {
    const payload = { mode, op: (thresholdCfg.op === 'and' ? 'and' : 'or'), dims, lows, highs };
    const res = await fetch('/api/threshold_global_stats', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('http ' + res.status);
    const data = await res.json();
    const hit = data.in_range || 0;
    const total = data.total || 0;
    drawPie(pieGlobal, hit, total);
    pieGlobalText.textContent = `${hit} / ${total}`;
  } catch (e) {
    drawPie(pieGlobal, 0, 0);
    pieGlobalText.textContent = '计算失败';
  }
}

function applyDatasetMeta(data) {
  if (data.cameras && data.cameras.length >= 3) {
    viewerCameras = data.cameras.slice(0, 3).map(c => (c === undefined || c === null || c === '') ? null : c);
  } else {
    viewerCameras = LEGACY_VIEWER_CAMERAS.slice();
  }
  while (viewerCameras.length < 3) viewerCameras.push(null);
  const camRoles = ['左', '中', '右'];
  const labels = Array.isArray(data.camera_labels) ? data.camera_labels : [];
  for (let i = 0; i < 3; i++) {
    const id = i === 0 ? 'cam-label-left' : (i === 1 ? 'cam-label-center' : 'cam-label-right');
    const el = document.getElementById(id);
    if (!el) continue;
    const cam = viewerCameras[i];
    const sub = labels[i] ? (' · ' + labels[i]) : '';
    el.textContent = cam ? (camRoles[i] + sub) : (camRoles[i] + ' —');
  }
  if (rotateHeadBtn) {
    rotateHeadBtn.style.display = viewerCameras[1] ? '' : 'none';
  }
  if (robotTypeBar) {
    const rt = data.robot_type;
    if (rt) {
      robotTypeBar.style.display = 'block';
      robotTypeBar.textContent = 'Robot type: ' + rt;
    } else {
      robotTypeBar.style.display = 'none';
      robotTypeBar.textContent = '';
    }
  }
  jointStateNames = Array.isArray(data.state_names) ? data.state_names.map(String) : [];
  jointActionNames = Array.isArray(data.action_names) ? data.action_names.map(String) : [];
  if (data.dataset_frames_key != null && data.dataset_frames_key !== '') {
    datasetFramesKey = String(data.dataset_frames_key);
  } else {
    datasetFramesKey = '';
  }
  ensureJointCharts(null);
}

function applyPairingUi() {
  if (pairingSelect) pairingSelect.value = pairingMode;
  if (pairingSimBox) pairingSimBox.style.display = (pairingMode === 'similarity') ? '' : 'none';
  if (pairingSimThresholdInput) pairingSimThresholdInput.value = String(pairingSimThreshold);
}

let previewNextOffset = 0;
let previewLoading = false;
let previewMode = 'repo'; // 'repo' | 'episodes'

function thumbUrl(item) {
  return '/api/episode_preview?dataset=' + encodeURIComponent(item.dataset_path) +
    '&episode=' + encodeURIComponent(item.episode);
}

function setSelectValueIfExists(sel, v) {
  if (!sel) return false;
  const old = sel.value;
  sel.value = v;
  if (sel.value !== v) {
    sel.value = old;
    return false;
  }
  return true;
}

function loadEpisodeFromPreview(datasetPath, episode) {
  const ok = setSelectValueIfExists(datasetSelect, datasetPath);
  if (ok) {
    datasetSelect.dispatchEvent(new Event('change'));
    setSelectValueIfExists(episodeSelect, episode);
    episodeSelect.dispatchEvent(new Event('change'));
    return;
  }
  currentDataset = datasetPath;
  currentEpisode = episode;
  loadEpisodeData(datasetPath, episode);
}

async function fetchPreviewPage() {
  if (previewLoading) return;
  previewLoading = true;
  try {
    let url = '';
    if (previewMode === 'episodes') {
      if (!currentDataset) throw new Error('no dataset selected');
      url = '/api/preview_episodes?dataset=' + encodeURIComponent(currentDataset) + '&offset=' + previewNextOffset + '&limit=50';
    } else {
      url = '/api/preview_index?offset=' + previewNextOffset + '&limit=50';
    }
    const r = await fetch(url);
    const j = await r.json();
    const items = (j && j.items) ? j.items : [];
    previewNextOffset = (j && j.next_offset != null) ? j.next_offset : (previewNextOffset + items.length);
    for (const it of items) addPreviewCard(it);
    if (previewLoadMoreBtn) previewLoadMoreBtn.style.display = items.length ? '' : 'none';
  } catch (e) {
    if (previewLoadMoreBtn) previewLoadMoreBtn.style.display = 'none';
  } finally {
    previewLoading = false;
  }
}

function addPreviewCard(item) {
  if (!previewGrid) return;
  const card = document.createElement('div');
  card.style.background = '#fff';
  card.style.border = '1px solid #ddd';
  card.style.borderRadius = '10px';
  card.style.overflow = 'hidden';
  card.style.cursor = 'pointer';
  card.style.boxShadow = '0 1px 0 rgba(0,0,0,0.03)';
  const img = document.createElement('img');
  img.src = thumbUrl(item);
  img.alt = '';
  img.style.width = '100%';
  img.style.display = 'block';
  img.style.background = '#222';
  img.loading = 'lazy';
  const cap = document.createElement('div');
  cap.style.padding = '8px 10px';
  cap.style.fontSize = '12px';
  cap.style.color = '#333';
  cap.style.wordBreak = 'break-all';
  cap.innerHTML = '<div style="font-weight:700;">' + item.episode + '</div>' +
    '<div style="color:#666;margin-top:2px;">' + item.dataset_name + '</div>';
  card.appendChild(img);
  card.appendChild(cap);
  card.addEventListener('click', () => loadEpisodeFromPreview(item.dataset_path, item.episode));
  previewGrid.appendChild(card);
}

function applyPreviewUi() {
  const onRepo = !!(previewToggle && previewToggle.checked);
  const onEpisodes = !!(previewCurrentToggle && previewCurrentToggle.checked);
  const on = onRepo || onEpisodes;
  previewMode = onEpisodes ? 'episodes' : 'repo';
  if (!previewGrid) return;
  previewGrid.style.display = on ? 'grid' : 'none';
  if (previewLoadMoreBtn) previewLoadMoreBtn.style.display = (on ? '' : 'none');
  if (on) {
    previewGrid.innerHTML = '';
    previewNextOffset = 0;
    fetchPreviewPage();
  }
}

try {
  if (previewToggle) {
    const v = localStorage.getItem('previewToggle');
    previewToggle.checked = (v === null) ? true : (v === '1');
  }
  if (previewCurrentToggle) {
    const v2 = localStorage.getItem('previewCurrentToggle');
    previewCurrentToggle.checked = (v2 === null) ? false : (v2 === '1');
  }
} catch (e) {}
applyPreviewUi();
if (previewToggle) {
  previewToggle.addEventListener('change', () => {
    try { localStorage.setItem('previewToggle', previewToggle.checked ? '1' : '0'); } catch (e) {}
    if (previewToggle.checked && previewCurrentToggle) previewCurrentToggle.checked = false;
    applyPreviewUi();
  });
}
if (previewCurrentToggle) {
  previewCurrentToggle.addEventListener('change', () => {
    try { localStorage.setItem('previewCurrentToggle', previewCurrentToggle.checked ? '1' : '0'); } catch (e) {}
    if (previewCurrentToggle.checked && previewToggle) previewToggle.checked = false;
    applyPreviewUi();
  });
}
if (previewLoadMoreBtn) {
  previewLoadMoreBtn.addEventListener('click', () => fetchPreviewPage());
}

try {
  const pm = localStorage.getItem('pairingMode');
  if (pm === 'index' || pm === 'name' || pm === 'similarity') pairingMode = pm;
  const th = parseFloat(localStorage.getItem('pairingSimThreshold') || '');
  if (!Number.isNaN(th) && th >= 0 && th <= 1) pairingSimThreshold = th;
} catch (e) {}

applyPairingUi();

function applyThresholdUiFromCfg() {
  if (thresholdOpSelect) thresholdOpSelect.value = thresholdCfg.op === 'and' ? 'and' : 'or';
  // constraints rows are rendered separately
}

function applyPreprocTypeUi() {
  if (preprocTypeSelect) preprocTypeSelect.value = preprocType;
  const showJoint = preprocType === 'joint_threshold';
  const showStatic = preprocType === 'static_velocity';
  if (preprocJointPanel) preprocJointPanel.style.display = showJoint ? 'flex' : 'none';
  if (preprocStaticPanel) preprocStaticPanel.style.display = showStatic ? 'flex' : 'none';
  if (thresholdConstraintsRoot) thresholdConstraintsRoot.style.display = showJoint ? 'flex' : 'none';
  // Only refresh heavy computations once episode data is loaded; otherwise we can
  // accidentally stomp initial UI state (especially when localStorage has old cfg).
  if (JOINT_BUNDLE && JOINT_BUNDLE.state && JOINT_BUNDLE.state.length) {
    recomputePreprocessorMask();
    updateThresholdEpisodePie();
    updateThresholdGlobalPie();
    updateFrame(currentFrame, {allowRetry: false});
  } else {
    if (pieEpisodeText) pieEpisodeText.textContent = '-';
    if (pieGlobalText) pieGlobalText.textContent = '-';
  }
}

try {
  const raw = localStorage.getItem('thresholdCfgV2');
  if (raw) {
    const obj = JSON.parse(raw);
    if (obj && typeof obj === 'object') {
      thresholdCfg.op = obj.op === 'and' ? 'and' : 'or';
      thresholdCfg.constraints = Array.isArray(obj.constraints) ? obj.constraints : [];
      thresholdCfg.enabled = thresholdCfg.constraints.length > 0;
    }
  } else {
    // Backward compatible single-joint keys
    const tDef = localStorage.getItem('thresholdDefId');
    const tMode = localStorage.getItem('thresholdMode');
    const tLow = parseFloat(localStorage.getItem('thresholdLow') || '');
    const tHigh = parseFloat(localStorage.getItem('thresholdHigh') || '');
    const tEn = localStorage.getItem('thresholdEnabled') === '1';
    if (tDef) {
      thresholdCfg.constraints = [
        {defId: String(tDef), mode: (tMode === 'action' ? 'action' : 'state'), low: (Number.isNaN(tLow) ? 0 : tLow), high: (Number.isNaN(tHigh) ? 1 : tHigh)},
      ];
    }
    thresholdCfg.op = 'or';
    thresholdCfg.enabled = tEn && thresholdCfg.constraints.length > 0;
  }
} catch (e) {}
applyThresholdUiFromCfg();

try {
  const sraw = localStorage.getItem('staticPreprocCfgV1');
  if (sraw) {
    const o = JSON.parse(sraw);
    if (o && typeof o === 'object') {
      const fps = parseInt(o.fps, 10);
      if (Number.isFinite(fps) && fps > 0) staticCfg.fps = fps;
      const jv = parseFloat(o.joint_velocity_threshold);
      if (Number.isFinite(jv) && jv >= 0) staticCfg.joint_velocity_threshold = jv;
      const gv = parseFloat(o.gripper_velocity_threshold);
      if (Number.isFinite(gv) && gv >= 0) staticCfg.gripper_velocity_threshold = gv;
      const sh = parseInt(o.smoothing_half_window, 10);
      if (Number.isFinite(sh) && sh >= 0) staticCfg.smoothing_half_window = sh;
    }
  }
} catch (e) {}

function syncStaticInputsFromCfg() {
  if (staticFpsInput) staticFpsInput.value = String(staticCfg.fps);
  if (staticJointVthInput) staticJointVthInput.value = String(staticCfg.joint_velocity_threshold);
  if (staticGripperVthInput) staticGripperVthInput.value = String(staticCfg.gripper_velocity_threshold);
  if (staticSmoothHInput) staticSmoothHInput.value = String(staticCfg.smoothing_half_window);
}
syncStaticInputsFromCfg();

try {
  const pt = localStorage.getItem('datasetViewerPreprocType');
  if (pt === 'joint_threshold' || pt === 'none' || pt === 'static_velocity') preprocType = pt;
} catch (e) {}
if (preprocType === 'none' && Array.isArray(thresholdCfg.constraints) && thresholdCfg.constraints.length > 0) {
  preprocType = 'joint_threshold';
}
applyPreprocTypeUi();

function _makeConstraintRow(constraint) {
  const row = document.createElement('div');
  row.style.display = 'flex';
  row.style.alignItems = 'center';
  row.style.gap = '8px';
  row.style.flexWrap = 'wrap';
  row.style.background = '#fff';
  row.style.border = '1px solid #e3e3e3';
  row.style.borderRadius = '8px';
  row.style.padding = '8px 10px';

  const jointSel = document.createElement('select');
  jointSel.style.minWidth = '260px';
  jointSel.innerHTML = '<option value="">（选择关节）</option>';
  for (const it of CHART_DEFS || []) {
    const opt = document.createElement('option');
    opt.value = String(it.id);
    opt.textContent = it.name;
    jointSel.appendChild(opt);
  }
  if (constraint && constraint.defId != null) jointSel.value = String(constraint.defId);

  const modeSel = document.createElement('select');
  modeSel.innerHTML = '<option value="state">state</option><option value="action">action</option>';
  modeSel.value = (constraint && constraint.mode === 'action') ? 'action' : 'state';

  const lowInp = document.createElement('input');
  lowInp.type = 'number'; lowInp.step = '0.01'; lowInp.style.width = '90px';
  lowInp.value = (constraint && constraint.low != null) ? String(constraint.low) : '0';

  const highInp = document.createElement('input');
  highInp.type = 'number'; highInp.step = '0.01'; highInp.style.width = '90px';
  highInp.value = (constraint && constraint.high != null) ? String(constraint.high) : '1';

  const rm = document.createElement('button');
  rm.type = 'button';
  rm.textContent = '删除';
  rm.style.fontSize = '12px';
  rm.style.padding = '6px 10px';
  rm.style.border = '1px solid #bbb';
  rm.style.borderRadius = '8px';
  rm.style.background = '#fff';
  rm.style.cursor = 'pointer';
  rm.addEventListener('click', () => row.remove());

  row.appendChild(jointSel);
  row.appendChild(modeSel);
  row.appendChild(document.createTextNode('区间'));
  row.appendChild(lowInp);
  row.appendChild(document.createTextNode('到'));
  row.appendChild(highInp);
  row.appendChild(rm);

  row._getValue = () => ({
    defId: jointSel.value,
    mode: modeSel.value,
    low: parseFloat(lowInp.value),
    high: parseFloat(highInp.value),
  });
  return row;
}

function renderThresholdConstraintsFromCfg() {
  if (!thresholdConstraintsRoot) return;
  thresholdConstraintsRoot.innerHTML = '';
  const cs = Array.isArray(thresholdCfg.constraints) ? thresholdCfg.constraints : [];
  for (const c of cs) thresholdConstraintsRoot.appendChild(_makeConstraintRow(c));
  if (!cs.length) thresholdConstraintsRoot.appendChild(_makeConstraintRow({}));
}
renderThresholdConstraintsFromCfg();

if (thresholdAddBtn) {
  thresholdAddBtn.addEventListener('click', () => {
    if (!thresholdConstraintsRoot) return;
    thresholdConstraintsRoot.appendChild(_makeConstraintRow({}));
  });
}

if (thresholdApplyBtn) {
  thresholdApplyBtn.addEventListener('click', () => {
    const op = thresholdOpSelect ? (thresholdOpSelect.value || 'or') : 'or';
    const cs = [];
    if (thresholdConstraintsRoot) {
      for (const child of Array.from(thresholdConstraintsRoot.children)) {
        if (child && typeof child._getValue === 'function') {
          const v = child._getValue();
          if (v.defId) cs.push(v);
        }
      }
    }
    thresholdCfg = {enabled: cs.length > 0, op: (op === 'and' ? 'and' : 'or'), constraints: cs};
    if (cs.length > 0) {
      preprocType = 'joint_threshold';
      try { localStorage.setItem('datasetViewerPreprocType', 'joint_threshold'); } catch (e) {}
    }
    try {
      localStorage.setItem('thresholdCfgV2', JSON.stringify(thresholdCfg));
      localStorage.setItem('thresholdEnabled', thresholdCfg.enabled ? '1' : '0');
    } catch (e) {}
    applyPreprocTypeUi();
  });
}
if (thresholdClearBtn) {
  thresholdClearBtn.addEventListener('click', () => {
    thresholdCfg.enabled = false;
    thresholdCfg.constraints = [];
    applyThresholdUiFromCfg();
    renderThresholdConstraintsFromCfg();
    try { localStorage.setItem('thresholdEnabled', '0'); } catch (e) {}
    try { localStorage.setItem('thresholdCfgV2', JSON.stringify(thresholdCfg)); } catch (e) {}
    recomputePreprocessorMask();
    updateThresholdEpisodePie();
    updateThresholdGlobalPie();
    updateFrame(currentFrame, {allowRetry: false});
  });
}

if (preprocTypeSelect) {
  preprocTypeSelect.addEventListener('change', () => {
    const v = preprocTypeSelect.value || 'none';
    preprocType = (v === 'joint_threshold' || v === 'static_velocity') ? v : 'none';
    try { localStorage.setItem('datasetViewerPreprocType', preprocType); } catch (e) {}
    applyPreprocTypeUi();
    renderThresholdConstraintsFromCfg();
  });
}

function readStaticCfgFromInputs() {
  const fps = parseInt(staticFpsInput && staticFpsInput.value, 10);
  staticCfg.fps = (Number.isFinite(fps) && fps > 0) ? fps : 30;
  const jv = parseFloat(staticJointVthInput && staticJointVthInput.value);
  staticCfg.joint_velocity_threshold = (Number.isFinite(jv) && jv >= 0) ? jv : 0.1;
  const gv = parseFloat(staticGripperVthInput && staticGripperVthInput.value);
  staticCfg.gripper_velocity_threshold = (Number.isFinite(gv) && gv >= 0) ? gv : 0.2;
  const sh = parseInt(staticSmoothHInput && staticSmoothHInput.value, 10);
  staticCfg.smoothing_half_window = (Number.isFinite(sh) && sh >= 0) ? sh : 2;
}

if (staticApplyBtn) {
  staticApplyBtn.addEventListener('click', () => {
    readStaticCfgFromInputs();
    try { localStorage.setItem('staticPreprocCfgV1', JSON.stringify(staticCfg)); } catch (e) {}
    syncStaticInputsFromCfg();
    if (JOINT_BUNDLE && JOINT_BUNDLE.state && JOINT_BUNDLE.state.length) {
      recomputePreprocessorMask();
      updateThresholdEpisodePie();
      updateThresholdGlobalPie();
      updateFrame(currentFrame, {allowRetry: false});
    }
  });
}

if (pairingSelect) {
  pairingSelect.addEventListener('change', () => {
    pairingMode = pairingSelect.value || 'index';
    try { localStorage.setItem('pairingMode', pairingMode); } catch (e) {}
    applyPairingUi();
    if (JOINT_BUNDLE && JOINT_BUNDLE.state && JOINT_BUNDLE.state.length) {
      ensureJointCharts(JOINT_BUNDLE.state[0]);
      drawAllCurves();
      recomputePreprocessorMask();
      updateThresholdEpisodePie();
      updateFrame(currentFrame, {allowRetry: false});
    }
  });
}
if (pairingSimThresholdInput) {
  pairingSimThresholdInput.addEventListener('change', () => {
    const v = parseFloat(pairingSimThresholdInput.value);
    if (!Number.isNaN(v) && v >= 0 && v <= 1) {
      pairingSimThreshold = v;
      try { localStorage.setItem('pairingSimThreshold', String(v)); } catch (e) {}
      if (pairingMode === 'similarity' && JOINT_BUNDLE && JOINT_BUNDLE.state && JOINT_BUNDLE.state.length) {
        ensureJointCharts(JOINT_BUNDLE.state[0]);
        drawAllCurves();
        recomputePreprocessorMask();
        updateThresholdEpisodePie();
        updateFrame(currentFrame, {allowRetry: false});
      }
    }
  });
}

function jointNameAt(arr, i) {
  if (arr[i] != null && arr[i] !== '') return String(arr[i]);
  return 'joint_' + i;
}

function _normName(s) {
  return String(s || '')
    .toLowerCase()
    .replace(/\\s+/g, '')
    .replace(/^(observation\\.|action\\.|state\\.)/g, '')
    .replace(/(state|action)$/g, '')
    .replace(/[:_]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '');
}

function _diceBigrams(a, b) {
  a = _normName(a); b = _normName(b);
  if (!a || !b) return 0;
  if (a === b) return 1;
  const bg = (x) => {
    const out = new Map();
    for (let i = 0; i < x.length - 1; i++) {
      const k = x.slice(i, i + 2);
      out.set(k, (out.get(k) || 0) + 1);
    }
    return out;
  };
  const A = bg(a), B = bg(b);
  let inter = 0, na = 0, nb = 0;
  for (const v of A.values()) na += v;
  for (const v of B.values()) nb += v;
  for (const [k, va] of A.entries()) {
    const vb = B.get(k) || 0;
    inter += Math.min(va, vb);
  }
  return (2 * inter) / (na + nb || 1);
}

function buildChartDefs(mode, threshold, nDataDims) {
  const nS = jointStateNames.length;
  const nA = jointActionNames.length;
  const defs = [];

  if (mode === 'index') {
    const n = Math.max(nS, nA, nDataDims || 0);
    for (let i = 0; i < n; i++) {
      const sName = (i < nS) ? jointNameAt(jointStateNames, i) : '';
      const aName = (i < nA) ? jointNameAt(jointActionNames, i) : '';
      const name = (sName && aName) ? ((sName === aName) ? sName : (sName + ' | ' + aName)) : (sName || aName || ('joint_' + i));
      defs.push({ id: i, name, stateDim: (i < nS ? i : null), actionDim: (i < nA ? i : null) });
    }
    return defs;
  }

  const stateIdxByName = new Map();
  for (let i = 0; i < nS; i++) {
    const nm = jointNameAt(jointStateNames, i);
    if (!stateIdxByName.has(nm)) stateIdxByName.set(nm, i);
  }
  const actionIdxByName = new Map();
  for (let i = 0; i < nA; i++) {
    const nm = jointNameAt(jointActionNames, i);
    if (!actionIdxByName.has(nm)) actionIdxByName.set(nm, i);
  }

  if (mode === 'name') {
    const nameSet = new Set([...stateIdxByName.keys(), ...actionIdxByName.keys()]);
    const names = Array.from(nameSet);
    names.sort();
    for (let k = 0; k < names.length; k++) {
      const nm = names[k];
      defs.push({
        id: k,
        name: nm,
        stateDim: stateIdxByName.has(nm) ? stateIdxByName.get(nm) : null,
        actionDim: actionIdxByName.has(nm) ? actionIdxByName.get(nm) : null,
      });
    }
    return defs;
  }

  // similarity: greedy max-weight matching by dice score
  const sNames = Array.from(stateIdxByName.keys());
  const aNames = Array.from(actionIdxByName.keys());
  const candidates = [];
  for (let si = 0; si < sNames.length; si++) {
    for (let ai = 0; ai < aNames.length; ai++) {
      const sc = _diceBigrams(sNames[si], aNames[ai]);
      if (sc >= threshold) candidates.push({ si, ai, sc });
    }
  }
  candidates.sort((x, y) => y.sc - x.sc);
  const usedS = new Set();
  const usedA = new Set();
  const pairs = [];
  for (const c of candidates) {
    if (usedS.has(c.si) || usedA.has(c.ai)) continue;
    usedS.add(c.si); usedA.add(c.ai);
    pairs.push(c);
  }
  const paired = pairs.map(p => {
    const sn = sNames[p.si], an = aNames[p.ai];
    const name = (_normName(sn) === _normName(an)) ? (sn.length >= an.length ? sn : an) : (sn + ' ≈ ' + an);
    return { name, stateDim: stateIdxByName.get(sn), actionDim: actionIdxByName.get(an) };
  });
  const unpairedS = sNames.filter((_, i) => !usedS.has(i)).map(sn => ({ name: sn, stateDim: stateIdxByName.get(sn), actionDim: null }));
  const unpairedA = aNames.filter((_, i) => !usedA.has(i)).map(an => ({ name: an, stateDim: null, actionDim: actionIdxByName.get(an) }));
  const all = [...paired, ...unpairedS, ...unpairedA];
  all.sort((x, y) => String(x.name).localeCompare(String(y.name)));
  for (let i = 0; i < all.length; i++) defs.push({ id: i, ...all[i] });
  return defs;
}

function ensureJointCharts(dataRow) {
  const nMeta = Math.max(jointStateNames.length, jointActionNames.length);
  const nData = (dataRow && dataRow.length) ? dataRow.length : 0;
  let n = nMeta;
  if (nMeta === 0 && nData) n = nData;
  else if (nMeta > 0 && nData > 0) n = Math.min(nMeta, nData);
  else if (nMeta > 0) n = nMeta;
  else if (nData) n = nData;
  else n = 0;
  if (!jointChartsRoot) return;
  if (n <= 0) {
    jointChartsRoot.innerHTML = '<div style="color:#888;font-size:12px">选择 episode 并等待数据加载后将显示关节曲线</div>';
    JOINT_DIM_COUNT = 0;
    CHART_DEFS = [];
    Object.keys(curvePoints).forEach(k => delete curvePoints[k]);
    return;
  }
  const items = buildChartDefs(pairingMode, pairingSimThreshold, nData);
  const existing = jointChartsRoot.querySelectorAll('canvas').length;
  if (JOINT_DIM_COUNT === items.length && existing === items.length) return;
  JOINT_DIM_COUNT = items.length;
  CHART_DEFS = items;
  // Refresh threshold constraint rows (joint options)
  if (thresholdConstraintsRoot) {
    // best-effort preserve current rows values
    const existing = Array.from(thresholdConstraintsRoot.children || []).map(ch => {
      if (ch && typeof ch._getValue === 'function') return ch._getValue();
      return null;
    }).filter(Boolean);
    thresholdConstraintsRoot.innerHTML = '';
    if (existing.length) {
      for (const c of existing) thresholdConstraintsRoot.appendChild(_makeConstraintRow(c));
    } else {
      renderThresholdConstraintsFromCfg();
    }
  }
  Object.keys(curvePoints).forEach(k => delete curvePoints[k]);
  jointChartsRoot.innerHTML = '';
  function classify(it) {
    const x = String(it.name).toLowerCase();
    if (/\\bright\\b/.test(x)) return 'r';
    if (/\\bleft\\b/.test(x)) return 'l';
    return 'o';
  }
  const left = items.filter(it => classify(it) === 'l');
  const right = items.filter(it => classify(it) === 'r');
  const other = items.filter(it => classify(it) === 'o');
  let groups = [];
  if (left.length && right.length) {
    groups.push({ title: '左臂 (left)', items: left });
    groups.push({ title: '右臂 (right)', items: right });
    if (other.length) groups.push({ title: '其他', items: other });
  } else if (items.length) {
    groups.push({ title: '关节', items: items });
  }
  function pairingLabel() {
    if (pairingMode === 'name') return '按名字精确匹配';
    if (pairingMode === 'similarity') return `按名字相似度≥${(pairingSimThreshold ?? 0.7)}`;
    return '按维度 index';
  }
  const COLS = 7;
  for (const g of groups) {
    const lab = document.createElement('div');
    lab.className = 'arm-label';
    if (g.title === '关节') {
      lab.style.display = 'flex';
      lab.style.alignItems = 'center';
      lab.style.justifyContent = 'flex-start';
      lab.style.gap = '8px';
      const leftTitle = document.createElement('span');
      leftTitle.textContent = g.title;
      lab.appendChild(leftTitle);
      // Move the "配对逻辑" controls next to the title, wrapped in parentheses.
      if (pairingControls) {
        const wrap = document.createElement('span');
        wrap.style.display = 'inline-flex';
        wrap.style.alignItems = 'center';
        wrap.style.gap = '8px';
        wrap.style.marginLeft = '6px';
        wrap.style.fontWeight = '600';
        wrap.style.color = '#666';
        const l = document.createElement('span');
        l.textContent = '（';
        const r = document.createElement('span');
        r.textContent = '）';
        pairingControls.style.margin = '0';
        pairingControls.style.gap = '8px';
        pairingControls.style.flexWrap = 'nowrap';
        pairingControls.style.alignItems = 'center';
        wrap.appendChild(l);
        wrap.appendChild(pairingControls);
        wrap.appendChild(r);
        lab.appendChild(wrap);
      } else {
        const rightText = document.createElement('span');
        rightText.textContent = `（${pairingLabel()}）`;
        rightText.style.fontWeight = '600';
        rightText.style.color = '#666';
        lab.appendChild(rightText);
      }
    } else {
      lab.textContent = g.title;
    }
    jointChartsRoot.appendChild(lab);
    for (let c = 0; c < g.items.length; c += COLS) {
      const row = document.createElement('div');
      row.className = 'canvas-row';
      const chunk = g.items.slice(c, c + COLS);
      row.style.gridTemplateColumns = 'repeat(' + chunk.length + ', 1fr)';
      for (const it of chunk) {
        const cell = document.createElement('div');
        cell.className = 'canvas-cell';
        const canvas = document.createElement('canvas');
        canvas.id = 'canvas-' + it.id;
        cell.appendChild(canvas);
        const titleDiv = document.createElement('div');
        titleDiv.className = 'joint-title';
        titleDiv.textContent = it.name;
        cell.appendChild(titleDiv);
        row.appendChild(cell);
      }
      jointChartsRoot.appendChild(row);
    }
  }
}

let headRotate180 = false;
try {
  headRotate180 = localStorage.getItem('headRotate180') === '1';
} catch (e) {}

function applyHeadRotation() {
  if (!frameImage) return;
  if (headRotate180) frameImage.style.transform = 'rotate(180deg)';
  else frameImage.style.transform = '';
  if (rotateHeadBtn) {
    rotateHeadBtn.textContent = headRotate180 ? 'Rotate 0°' : 'Rotate 180°';
    rotateHeadBtn.style.borderColor = headRotate180 ? '#4a90d9' : '#bbb';
    rotateHeadBtn.style.color = headRotate180 ? '#1f5f9d' : '#222';
  }
}

applyHeadRotation();

if (rotateHeadBtn) {
  rotateHeadBtn.addEventListener('click', () => {
    headRotate180 = !headRotate180;
    try { localStorage.setItem('headRotate180', headRotate180 ? '1' : '0'); } catch (e) {}
    applyHeadRotation();
  });
}

window.addEventListener('keydown', (e) => {
  if (e.key === 'r' || e.key === 'R') {
    headRotate180 = !headRotate180;
    try { localStorage.setItem('headRotate180', headRotate180 ? '1' : '0'); } catch (err) {}
    applyHeadRotation();
  }
});

fetch('/api/datasets')
  .then(r => r.json())
  .then(datasets => {
    datasetSelect.innerHTML = '';
    datasets.forEach(ds => {
      const opt = document.createElement('option');
      const p = ds.path || ds.name;
      opt.value = p;
      opt.textContent = p;
      opt.dataset.episodes = JSON.stringify(ds.episodes);
      datasetSelect.appendChild(opt);
    });
    datasetSelect.dispatchEvent(new Event('change'));
  });

datasetSelect.addEventListener('change', () => {
  currentDataset = datasetSelect.value;
  const selected = datasetSelect.selectedOptions[0];
  const episodes = JSON.parse(selected.dataset.episodes || '[]');
  episodeSelect.innerHTML = '';
  episodes.forEach(ep => {
    const opt = document.createElement('option');
    opt.value = ep;
    opt.textContent = ep;
    episodeSelect.appendChild(opt);
  });
  episodeSelect.dispatchEvent(new Event('change'));
});

episodeSelect.addEventListener('change', () => {
  currentEpisode = episodeSelect.value;
  if (!currentDataset || !currentEpisode) return;
  loadEpisodeData(currentDataset, currentEpisode, true);
});

function loadEpisodeData(dataset, episode, previewOnly) {
  currentLoadToken++;
  const myToken = currentLoadToken;
  JOINT_BUNDLE = null;
  datasetFramesKey = '';
  if (robotTypeBar) { robotTypeBar.style.display = 'none'; robotTypeBar.textContent = ''; }
  if (jointChartsRoot) {
    jointChartsRoot.innerHTML = '<div style="color:#888;font-size:12px">加载中…</div>';
    JOINT_DIM_COUNT = 0;
    Object.keys(curvePoints).forEach(k => delete curvePoints[k]);
  }
  viewerCameras = [null, null, null];
  [['cam-label-left', '左'], ['cam-label-center', '中'], ['cam-label-right', '右']].forEach(([id, r]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = r + ' —';
  });
  if (rotateHeadBtn) rotateHeadBtn.style.display = 'none';
  frameSlider.disabled = true;
  frameSlider.value = 0;
  totalFrames = 0;
  frameCounter.textContent = 'Frame 0 / 0';
  promptText.textContent = '';
  frameImageLeft.alt = frameImage.alt = frameImageRight.alt = 'Loading...';
  frameImageLeft.src = frameImage.src = frameImageRight.src = '';

  if (loadAllFramesBtn) loadAllFramesBtn.style.display = 'none';
  if (previewHint) previewHint.style.display = 'none';
  showProgress(previewOnly ? '加载首帧（快速预览）...' : 'Extracting frames + parsing curves...', 0);
  const eventSource = new EventSource(
    '/api/load_episode?dataset=' + encodeURIComponent(dataset) +
    '&episode=' + encodeURIComponent(episode) +
    (previewOnly ? '&preview=1' : '')
  );

  eventSource.addEventListener('meta', (e) => {
    if (myToken !== currentLoadToken) return;
    const data = JSON.parse(e.data);
    const fc = data.frame_count || 0;
    if (data.tasks && data.tasks.length) {
      promptText.textContent = data.tasks.join('\\n');
    } else {
      promptText.textContent = '';
    }
    applyDatasetMeta(data);
    const isPreview = !!data.preview_only;
    previewOnlyImages = isPreview;
    if (isPreview) {
      totalFrames = 0;
      frameSlider.max = 0;
      frameSlider.value = 0;
      frameSlider.disabled = false;
      if (loadAllFramesBtn) loadAllFramesBtn.style.display = '';
      if (previewHint) previewHint.style.display = '';
    } else {
      totalFrames = Math.max(0, fc - 1);
      frameSlider.max = totalFrames;
      frameSlider.value = 0;
      frameSlider.disabled = false;
      if (loadAllFramesBtn) loadAllFramesBtn.style.display = 'none';
      if (previewHint) previewHint.style.display = 'none';
    }
    updateFrame(0, {allowRetry: true});
  });

  eventSource.addEventListener('progress', (e) => {
    if (myToken !== currentLoadToken) return;
    const data = JSON.parse(e.data);
    updateProgress(data.percent, data.status);
  });

  eventSource.addEventListener('done', (e) => {
    if (myToken !== currentLoadToken) return;
    const data = JSON.parse(e.data);
    JOINT_BUNDLE = data.data;
    eventSource.close();
    hideProgress();
    initViewer();
  });

  eventSource.addEventListener('error', (e) => {
    if (myToken !== currentLoadToken) return;
    if (e.data) {
      const data = JSON.parse(e.data);
      alert('Error: ' + data.message);
    }
    eventSource.close();
    hideProgress();
  });

  eventSource.onerror = () => {
    if (myToken !== currentLoadToken) return;
    eventSource.close();
    hideProgress();
  };
}

if (loadAllFramesBtn) {
  loadAllFramesBtn.addEventListener('click', () => {
    if (!currentDataset || !currentEpisode) return;
    loadEpisodeData(currentDataset, currentEpisode, false);
  });
}

function showProgress(title, percent) {
  document.getElementById('progress-title').textContent = title;
  progressBarFill.style.width = percent + '%';
  progressText.textContent = percent + '%';
  progressOverlay.style.display = 'flex';
}

function updateProgress(percent, status) {
  progressBarFill.style.width = percent + '%';
  progressText.textContent = (status ? status + ' - ' : '') + percent + '%';
}

function hideProgress() {
  progressOverlay.style.display = 'none';
}

function initViewer() {
  if (JOINT_BUNDLE && JOINT_BUNDLE.state && JOINT_BUNDLE.state.length) {
    ensureJointCharts(JOINT_BUNDLE.state[0]);
    totalFrames = JOINT_BUNDLE.state.length - 1;
    frameSlider.max = totalFrames;
    frameSlider.disabled = false;
    currentFrame = 0;
    drawAllCurves();
    recomputePreprocessorMask();
    updateThresholdEpisodePie();
    updateThresholdGlobalPie();
    updateFrame(0, {allowRetry: false});
  }
}

frameSlider.addEventListener('input', () => {
  updateFrame(parseInt(frameSlider.value), {allowRetry: true});
});

function clampInt(v, lo, hi) {
  v = Math.round(v);
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

function frameFromCanvasX(canvas, clientX) {
  if (!canvas) return null;
  const id = String(canvas.id || '');
  if (!id.startsWith('canvas-')) return null;
  const defId = id.slice('canvas-'.length);
  const cp = curvePoints[defId];
  if (!cp) return null;
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left; // CSS pixels
  const plotW = cp.w - cp.pad.left - cp.pad.right; // CSS pixels
  const maxF = Math.max(1, totalFrames || 1);
  if (plotW <= 1) return 0;
  const t = (x - cp.pad.left) / plotW;
  const idx = clampInt(t * maxF, 0, maxF);
  return idx;
}

function dragToFrame(idx) {
  if (idx == null) return;
  if (!Number.isFinite(idx)) return;
  const maxF = Math.max(0, totalFrames || 0);
  const clamped = clampInt(idx, 0, maxF);
  if (clamped === currentFrame) return;
  updateFrame(clamped, {allowRetry: true});
}

// Drag red marker on joint canvases
if (jointChartsRoot) {
  jointChartsRoot.addEventListener('mousedown', (e) => {
    const t = e.target;
    if (!(t instanceof HTMLCanvasElement)) return;
    if (!JOINT_BUNDLE || !JOINT_BUNDLE.state || !JOINT_BUNDLE.state.length) return;
    const idx = frameFromCanvasX(t, e.clientX);
    if (idx == null) return;
    isDraggingMarker = true;
    e.preventDefault();
    dragToFrame(idx);
  });
  window.addEventListener('mousemove', (e) => {
    if (!isDraggingMarker) return;
    const el = document.elementFromPoint(e.clientX, e.clientY);
    const canvas = (el instanceof HTMLCanvasElement && el.id && String(el.id).startsWith('canvas-')) ? el : null;
    // If cursor moves outside canvas, keep last active frame by using currentFrame,
    // but if we are over any joint canvas, update to that x.
    if (canvas) {
      const idx = frameFromCanvasX(canvas, e.clientX);
      dragToFrame(idx);
    }
  });
  window.addEventListener('mouseup', () => {
    isDraggingMarker = false;
  });
  window.addEventListener('mouseleave', () => {
    isDraggingMarker = false;
  });
}

function frameUrlForCamera(padded, camera) {
  return 'frames/' + encodeURIComponent(segmentForFrameUrls()) + '/' + encodeURIComponent(currentEpisode) + '/' + encodeURIComponent(camera) + '/frame_' + padded + '.jpg';
}

function legacyFlatFrameUrl(padded) {
  return 'frames/' + encodeURIComponent(segmentForFrameUrls()) + '/' + encodeURIComponent(currentEpisode) + '/frame_' + padded + '.jpg';
}

function updateFrame(idx, opts) {
  opts = opts || {};
  currentFrame = idx;
  frameSlider.value = idx;
  const imgIdx = previewOnlyImages ? 0 : idx;
  const padded = String(imgIdx).padStart(6, '0');
  const leftCam = viewerCameras[0];
  const headCam = viewerCameras[1];
  const rightCam = viewerCameras[2];

  function bindError(img, camera) {
    if (!camera) {
      img.onerror = null;
      return;
    }
    img.onerror = function() {
      if (camera === headCam && headCam && !img.dataset.legacyFlat) {
        img.dataset.legacyFlat = '1';
        img.src = legacyFlatFrameUrl(padded);
        return;
      }
      this.alt = 'Frame not ready yet (still extracting)...';
      this.src = '';
      if (!previewOnlyImages && opts.allowRetry) scheduleFrameRetry(idx);
    };
    delete img.dataset.legacyFlat;
  }

  bindError(frameImageLeft, leftCam);
  bindError(frameImage, headCam);
  bindError(frameImageRight, rightCam);

  function setPanelImg(img, cam) {
    if (!cam) {
      img.src = '';
      img.alt = '（无此视角）';
      img.style.background = '#2a2a2a';
      return;
    }
    img.alt = 'Frame';
    img.style.background = '';
    img.src = frameUrlForCamera(padded, cam);
  }
  setPanelImg(frameImageLeft, leftCam);
  setPanelImg(frameImage, headCam);
  setPanelImg(frameImageRight, rightCam);
  applyHeadRotation();
  frameCounter.textContent = 'Frame ' + idx + ' / ' + totalFrames;
  if (JOINT_BUNDLE) drawMarker(idx);
}

function scheduleFrameRetry(idx) {
  if (pendingFrameRetry) clearTimeout(pendingFrameRetry);
  // Retry quickly for a short period; user can scrub while frames are being extracted
  pendingFrameRetry = setTimeout(() => {
    if (idx !== currentFrame) return;
    updateFrame(idx, {allowRetry: true});
  }, 500);
}

function drawAllCurves() {
  if (!JOINT_BUNDLE || !JOINT_BUNDLE.state || !JOINT_BUNDLE.state.length) return;
  for (const def of CHART_DEFS) {
    const canvas = document.getElementById('canvas-' + def.id);
    if (!canvas) continue;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    const w = rect.width;
    const h = rect.height;
    const pad = {top: 4, right: 4, bottom: 4, left: 4};
    const valsS = (def.stateDim != null) ? JOINT_BUNDLE.state.map(row => row[def.stateDim]) : null;
    const valsA = (def.actionDim != null) ? JOINT_BUNDLE.action.map(row => row[def.actionDim]) : null;
    const valsD = (valsS && valsA) ? valsA.map((v, idx) => {
      const s = valsS[idx];
      if (typeof v === 'number' && isFinite(v) && typeof s === 'number' && isFinite(s)) return v - s;
      return null;
    }) : null;
    let minY = Infinity, maxY = -Infinity;
    function scan(arr) {
      if (!arr) return;
      for (const v of arr) {
        if (v !== null && v !== undefined && typeof v === 'number' && isFinite(v)) {
          if (v < minY) minY = v;
          if (v > maxY) maxY = v;
        }
      }
    }
    scan(valsS); scan(valsA); scan(valsD);
    if (!isFinite(minY)) { minY = 0; maxY = 1; }
    const rangeY = maxY - minY || 1;
    minY -= rangeY * 0.05;
    maxY += rangeY * 0.05;
    curvePoints[def.id] = {minY, maxY, w, h, pad, hasS: !!valsS, hasA: !!valsA, hasD: !!valsD};
  }
}

function formatDimValue(v) {
  if (v === null || v === undefined) return 'N/A';
  if (typeof v === 'number' && !isFinite(v)) return 'N/A';
  return Number(v).toFixed(4);
}

function drawSingleMarker(ctx, frameIdx, maxFrame, minY, maxY, w, h, pad) {
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const x = pad.left + (frameIdx / maxFrame) * plotW;
  ctx.beginPath();
  ctx.strokeStyle = 'red';
  ctx.lineWidth = 1.5;
  ctx.moveTo(x, pad.top);
  ctx.lineTo(x, pad.top + plotH);
  ctx.stroke();
}

function plotSeries(ctx, vals, minY, maxY, plotW, plotH, pad, color) {
  const len = vals.length;
  const maxF = Math.max(1, len - 1);
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  let started = false;
  for (let i = 0; i < len; i++) {
    const v = vals[i];
    if (v === null || v === undefined || typeof v !== 'number' || !isFinite(v)) {
      if (started) { ctx.stroke(); ctx.beginPath(); started = false; }
      continue;
    }
    const x = pad.left + (i / maxF) * plotW;
    const y = pad.top + (1 - (v - minY) / (maxY - minY)) * plotH;
    if (!started) { ctx.moveTo(x, y); started = true; }
    else ctx.lineTo(x, y);
  }
  if (started) ctx.stroke();
}

function drawMarker(frameIdx) {
  if (!JOINT_BUNDLE || !JOINT_BUNDLE.state) return;
  for (const def of CHART_DEFS) {
    const canvas = document.getElementById('canvas-' + def.id);
    if (!canvas) continue;
    const cp = curvePoints[def.id];
    if (!cp) continue;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    ctx.clearRect(0, 0, cp.w, cp.h);

    const valsS = (def.stateDim != null) ? JOINT_BUNDLE.state.map(row => row[def.stateDim]) : null;
    const valsA = (def.actionDim != null) ? JOINT_BUNDLE.action.map(row => row[def.actionDim]) : null;
    const valsD = (valsS && valsA) ? valsA.map((v, idx) => {
      const s = valsS[idx];
      if (typeof v === 'number' && isFinite(v) && typeof s === 'number' && isFinite(s)) return v - s;
      return null;
    }) : null;
    const plotW = cp.w - cp.pad.left - cp.pad.right;
    const plotH = cp.h - cp.pad.top - cp.pad.bottom;
    const len0 = (valsS && valsS.length) ? valsS.length : ((valsA && valsA.length) ? valsA.length : 0);
    const maxF = Math.max(1, len0 - 1);

    // Highlight: joint-threshold (per def) or velocity-static (same mask on all charts)
    let _hm = thresholdMasksByDefId ? thresholdMasksByDefId[String(def.id)] : null;
    if (preprocType === 'static_velocity' && Array.isArray(thresholdMask) && thresholdMask.length) {
      _hm = thresholdMask;
    }
    const highlightOn =
      (preprocType === 'static_velocity' && _hm && Array.isArray(_hm)) ||
      (preprocType === 'joint_threshold' && thresholdCfg && thresholdCfg.enabled && _hm && Array.isArray(_hm));
    if (highlightOn) {
      ctx.save();
      ctx.fillStyle = 'rgba(255, 165, 0, 0.18)';
      let start = -1;
      const N = _hm.length;
      for (let i = 0; i <= N; i++) {
        const on = (i < N) ? !!_hm[i] : false;
        if (on && start < 0) start = i;
        if (!on && start >= 0) {
          const a = start;
          const b = i - 1;
          const x0 = cp.pad.left + (a / maxF) * plotW;
          const x1 = cp.pad.left + ((b + 1) / maxF) * plotW;
          ctx.fillRect(x0, cp.pad.top, Math.max(1, x1 - x0), plotH);
          start = -1;
        }
      }
      ctx.restore();
    }

    if (valsS) plotSeries(ctx, valsS, cp.minY, cp.maxY, plotW, plotH, cp.pad, '#4a90d9');
    if (valsA) plotSeries(ctx, valsA, cp.minY, cp.maxY, plotW, plotH, cp.pad, '#2d8f47');
    if (valsD) plotSeries(ctx, valsD, cp.minY, cp.maxY, plotW, plotH, cp.pad, '#8B4513');

    drawSingleMarker(ctx, frameIdx, maxF, cp.minY, cp.maxY, cp.w, cp.h, cp.pad);

    const mx = cp.pad.left + (frameIdx / maxF) * plotW;
    function dotAt(vals, color, r) {
      if (!vals) return;
      const cur = vals[frameIdx];
      if (cur === null || cur === undefined || typeof cur !== 'number' || !isFinite(cur)) return;
      const my = cp.pad.top + (1 - (cur - cp.minY) / (cp.maxY - cp.minY)) * plotH;
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(mx, my, r, 0, Math.PI * 2);
      ctx.fill();
    }
    dotAt(valsS, '#4a90d9', 2.5);
    dotAt(valsA, '#2d8f47', 2.5);
    dotAt(valsD, '#8B4513', 2.5);

    const lineRows = [];
    if (valsS) lineRows.push({ text: 'S ' + formatDimValue(valsS[frameIdx]), color: '#4a90d9' });
    if (valsA) lineRows.push({ text: 'A ' + formatDimValue(valsA[frameIdx]), color: '#2d8f47' });
    if (valsD) lineRows.push({ text: '\u0394 ' + formatDimValue(valsD[frameIdx]), color: '#8B4513' });
    ctx.save();
    ctx.font = '9px ui-monospace, Consolas, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    const tx = cp.w - cp.pad.right - 2;
    const ty = cp.pad.top + 2;
    const padX = 3;
    const padY = 2;
    const lineH = 10;
    let maxTw = 0;
    for (const row of lineRows) maxTw = Math.max(maxTw, ctx.measureText(row.text).width);
    const boxW = maxTw + padX * 2;
    const boxH = lineH * lineRows.length + padY * 2;
    ctx.fillStyle = 'rgba(255,255,255,0.94)';
    ctx.strokeStyle = 'rgba(0,0,0,0.12)';
    ctx.lineWidth = 1;
    ctx.fillRect(tx - boxW, ty, boxW, boxH);
    ctx.strokeRect(tx - boxW, ty, boxW, boxH);
    for (let k = 0; k < lineRows.length; k++) {
      ctx.fillStyle = lineRows[k].color;
      ctx.fillText(lineRows[k].text, tx - padX, ty + padY + k * lineH);
    }
    ctx.restore();
  }
}
</script>
</body>
</html>"""
    return html


def save_html(html_content, output_path):
    """Write HTML to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Generated {output_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
