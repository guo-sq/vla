import os

REPO_ID = []

# 下象棋&叠衣服 @weilong
ROOT_DIR = "/mnt/"
sub_path = "oss_data/anyverse"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [
    os.path.join(sub_path, task)
    for task in tasks
    if "bipiper" in task
    and "pmt" not in task
    and task.startswith("record")
    and task
    not in [
        "record.xiangqi.bipiper.v1128.1",
        "record.xiangqi.bipiper.v1215.7",
        "record.fold_towel.bipiper.v1114.13",
        "record.xiangqi.bipiper.v1205.1",
    ]
]

sub_path = "oss_data/anyverse/bipiper_clothes_sfp"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [
    os.path.join(sub_path, task)
    for task in tasks
    if task.startswith("record")
    if task
    not in [
        "record.clothes.bipiper.v0209.7",
        "record.clothes.bipiper.v0209.sfp.1",
    ]
]

sub_path = "oss_data/anyverse/bipiper_clothes"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
tasks = [
    task
    for task in tasks
    if task
    not in [
        "record.clothes.bipiper.v0113.5",
        "record.clothes.bipiper.v0116.10",
        "record.clothes.bipiper.v0128.r.policy.13",
        "record.clothes.bipiper.v1224.1",
        "record.clothes.bipiper.v0112.policy.7",
        "record.clothes.bipiper.v0116.7",
        "record.clothes.bipiper.v1229.1",
    ]
]
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task.startswith("record")]

sub_path = "oss_data/anyverse/bipiper_clothes49"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [
    os.path.join(sub_path, task)
    for task in tasks
    if task.startswith("record")
    and task
    not in [
        "record.fold.clothes.bipiper.v1225.1",
        "record.fold.clothes.bipiper.v1225.2",
    ]
]


# 叠纸盒任务 @zengqi
sub_path = "shared/datasets/anyverse_human_data_record_raw/arxx5_bimanual/fold_box"
tasks = [
    "record-arxx5_bimanual-fold-box-0126-1",
    "record-arxx5_bimanual-fold-box-0126-2",
    "record-arxx5_bimanual-fold-box-0126-3",
    "record-arxx5_bimanual-fold-box-0126-4",
    "record-arxx5_bimanual-fold-box-0126-5",
    "record-arxx5_bimanual-fold-box-0126-6",
    "record-arxx5_bimanual-fold-box-0126-7",
    "record-arxx5_bimanual-fold-box-0126-8",
    "record-arxx5_bimanual-fold-box-0126-9",
    "record-arxx5_bimanual-fold-box-0126-11",
    "record-arxx5_bimanual-fold-box-0126-12",
    "record-arxx5_bimanual-fold-box-0126-13",
    "record-arxx5_bimanual-fold-box-0126-14",
    "record-arxx5_bimanual-fold-box-0126-15",
    "record-arxx5_bimanual-fold-box-0126-17",
    "record-arxx5_bimanual-fold-box-0126-18",
    "record-arxx5_bimanual-fold-box-0126-19",
    "record-arxx5_bimanual-fold-box-0126-20",
    "record-arxx5_bimanual-fold-box-0126-21",
    "record-arxx5_bimanual-fold-box-0126-22",
    "record-arxx5_bimanual-fold-box-0126-23",
    "record-arxx5_bimanual-fold-box-0126-24",
    "record-arxx5_bimanual-fold-box-0126-25",
    "record-arxx5_bimanual-fold-box-0127-1",
    "record-arxx5_bimanual-fold-box-0127-2",
    "record-arxx5_bimanual-fold-box-0127-3",
    "record-arxx5_bimanual-fold-box-0127-4",
    "record-arxx5_bimanual-fold-box-0127-5",
    "record-arxx5_bimanual-fold-box-0127-6",
    "record-arxx5_bimanual-fold-box-0127-7",
    "record-arxx5_bimanual-fold-box-0127-9",
    "record-arxx5_bimanual-fold-box-0127-10",
    "record-arxx5_bimanual-fold-box-0127-11",
    "record-arxx5_bimanual-fold-box-0127-12",
    "record-arxx5_bimanual-fold-box-0128-1",
    "record-arxx5_bimanual-fold-box-0128-2",
    "record-arxx5_bimanual-fold-box-0128-3",
    "record-arxx5_bimanual-fold-box-0128-4",
    "record-arxx5_bimanual-fold-box-0128-5",
    "record-arxx5_bimanual-fold-box-0128-6",
    "record-arxx5_bimanual-fold-box-0128-7",
    "record-arxx5_bimanual-fold-box-0128-8",
    "record-arxx5_bimanual-fold-box-0128-9",
    "record-arxx5_bimanual-fold-box-0128-10",
    "record-arxx5_bimanual-fold-box-0128-11",
    "record-arxx5_bimanual-fold-box-0128-12",
    "record-arxx5_bimanual-fold-box-0128-13",
    "record-arxx5_bimanual-fold-box-0128-14",
    "record-arxx5_bimanual-fold-box-0128-15",
    "record-arxx5_bimanual-fold-box-0128-16",
    "record-arxx5_bimanual-fold-box-0128-17",
    "record-arxx5_bimanual-fold-box-0128-18",
    "record-arxx5_bimanual-fold-box-0128-19",
    "record-arxx5_bimanual-fold-box-0129-1",
    # "record-arxx5_bimanual-fold-box-0129-2",
    "record-arxx5_bimanual-fold-box-0129-3",
    "record-arxx5_bimanual-fold-box-0129-4",
    "record-arxx5_bimanual-fold-box-0129-5",
    "record-arxx5_bimanual-fold-box-0129-6",
    "record-arxx5_bimanual-fold-box-0129-7",
    "record-arxx5_bimanual-fold-box-0129-8",
    "record-arxx5_bimanual-fold-box-0129-9",
    "record-arxx5_bimanual-fold-box-0129-10",
    "record-arxx5_bimanual-fold-box-0129-11",
    "record-arxx5_bimanual-fold-box-0129-12",
    "record-arxx5_bimanual-fold-box-0129-13",
    "record-arxx5_bimanual-fold-box-0129-14",
    "record-arxx5_bimanual-fold-box-0129-15",
    "record-arxx5_bimanual-fold-box-0130-1",
    "record-arxx5_bimanual-fold-box-0130-5",
    "record-arxx5_bimanual-fold-box-0130-6",
    "record-arxx5_bimanual-fold-box-0130-7",
    "record-arxx5_bimanual-fold-box-0130-8",
    "record-arxx5_bimanual-fold-box-0130-9",
    "record-arxx5_bimanual-fold-box-0130-10",
    "record-arxx5_bimanual-fold-box-0202-2",
    "record-arxx5_bimanual-fold-box-0202-3",
    "record-arxx5_bimanual-fold-box-0202-4",
    "record-arxx5_bimanual-fold-box-0202-5",
    "record-arxx5_bimanual-fold-box-0202-6",
    "record-arxx5_bimanual-fold-box-0202-7",
    "record-arxx5_bimanual-fold-box-0202-8",
    "record-arxx5_bimanual-fold-box-0202-9",
    "record-arxx5_bimanual-fold-box-0203-1",
    "record-arxx5_bimanual-fold-box-0203-2",
    "record-arxx5_bimanual-fold-box-0203-3",
    "record-arxx5_bimanual-fold-box-0203-4",
    "record-arxx5_bimanual-fold-box-0203-5",
    "record-arxx5_bimanual-fold-box-0204-fast_1",
    "record-arxx5_bimanual-fold-box-0204-fast_2",
    "record-arxx5_bimanual-fold-box-0204-fast_3",
    "record-arxx5_bimanual-fold-box-0204-fast_4",
    "record-arxx5_bimanual-fold-box-0204-fast_5",
    "record-arxx5_bimanual-fold-box-0204-fast_6",
    "record-arxx5_bimanual-fold-box-0206-fast_1",
    "record-arxx5_bimanual-fold-box-0206-fast_2",
]
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task.startswith("record")]

# 倒水任务 @fangrui
sub_path = "shared/datasets/anyverse_pour_water_record"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [
    os.path.join(sub_path, task)
    for task in tasks
    if "problem" not in task
    and task
    not in [
        "record.pourwater.bipiper.0214.2",
        "record.pourwater.bipiper.0214.1",
        "record.pourwater.bipiper.0214.3",
        "record.pourwater.bipiper.0214.4",
        "record.pourwater.bipiper.0214.5",
        "record.pourwater.bipiper.0214.6",
        "record.pourwater.bipiper.0214.7",
    ]
]

# 叠/翻袜子&插管& @tengfei @yuyang
sub_path = "oss_data/anyverse_human_data_record/arxx5_bimanual"
tasks = []
for sub in os.listdir(os.path.join(ROOT_DIR, sub_path)):
    if sub in [
        "fold_box",
        "fold_towel",
        "fold_shirt",
        "insert_tube",
        "invert_socks",
        "pack_socks",
        "seatbelt",
        "static_sort",
    ]:
        continue
    tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path, sub)))
    REPO_ID += [
        os.path.join(sub_path, sub, task)
        for task in tasks
        if task
        not in [
            "fold_towel.40s.1104.batch.1_",
            "fold_towel.40s.1209.batch.1",
            "infer_pi05_base_finetune_data_1201_1219_filter_head_tail_static_horizon_50.exp.1223.20251224.1",
            "pack_socks.white.medium.full.40s.20260106.batch.9",
            "grab_and_attach_tube.full_tray_52_tube.40s.1211.batch.4",
            "pack_socks.black.M.invert.200s.20260122.batch.5",
            "pack_socks.white.M.invert.200s.20260203.batch.1",
        ]
    ]

# 抓取 @yushan
ROOT_DIR = "/mnt/"
sub_path = "shared/datasets/anyverse_pickAndplace_record_tmp"
tasks = [
    # 0106 ~0.55h
    "record.pick.place.bipiper.v0106.1",
    "record.pick.place.bipiper.v0106.2",
    # 0107 ~1.81h
    "record.pick.place.bipiper.v0107.1",
    "record.pick.place.bipiper.v0107.2",
    "record.pick.place.bipiper.v0107.3",
    "record.pick.place.bipiper.v0107.4",
    "record.pick.place.bipiper.v0107.6",
    "record.pick.place.bipiper.v0107.7",
    "record.pick.place.bipiper.v0107.8",
    # 0108 ~1.93h
    "record.pick.place.bipiper.v0108.1",
    "record.pick.place.bipiper.v0108.2",
    "record.pick.place.bipiper.v0108.3",
    "record.pick.place.bipiper.v0108.4",
    "record.pick.place.bipiper.v0108.6",
    "record.pick.place.bipiper.v0108.7",
    "record.pick.place.bipiper.v0108.9",
    "record.pick.place.bipiper.v0108.10",
    "record.pick.place.bipiper.v0108.11",
    "record.pick.place.bipiper.v0108.12",
    "record.pick.place.bipiper.v0108.13",
    "record.pick.place.bipiper.v0108.14",
    "record.pick.place.bipiper.v0108.15",
    "record.pick.place.bipiper.v0108.16",
    "record.pick.place.bipiper.v0108.17",
    "record.pick.place.bipiper.v0108.18",
    "record.pick.place.bipiper.v0108.19",
    "record.pick.place.bipiper.v0108.20",
    "record.pick.place.bipiper.v0108.21",
    "record.pick.place.bipiper.v0108.22",
    # 0109 ~1.69h
    "record.pick.place.bipiper.v0109.01",
    "record.pick.place.scheme2.bipiper.v0109.01",
    "record.pick.place.scheme2.bipiper.v0109.02",
    "record.pick.place.scheme2.bipiper.v0109.03",
    "record.pick.place.scheme2.bipiper.v0109.04",
    "record.pick.place.scheme2.bipiper.v0109.05",
    "record.pick.place.scheme2.bipiper.v0109.06",
    "record.pick.place.scheme2.bipiper.v0109.07",
    "record.pick.place.scheme2.bipiper.v0109.08",
    "record.pick.place.scheme2.bipiper.v0109.09",
    "record.pick.place.scheme2.bipiper.v0109.10",
    "record.pick.place.scheme2.bipiper.v0109.11",
    "record.pick.place.scheme2.bipiper.v0109.12",
    "record.pick.place.scheme2.bipiper.v0109.13",
    # 0112 ~ 1.61h
    "record.pick.place.scheme2.bipiper.v0112.4",
    "record.pick.place.scheme2.otherbackgrounds.bipiper.v0112.6",
    "record.pick.place.scheme2.bipiper.v0112.1",
    "record.pick.place.scheme2.bipiper.v0112.5",
    "record.pick.place.scheme2.otherbackgrounds.bipiper.v0112.8",
    "record.pick.place.scheme2.otherbackgrounds.bipiper.v0112.11",
    "record.pick.place.scheme2.otherbackgrounds.bipiper.v0112.7",
    "record.pick.place.scheme2.bipiper.v0112.3",
    "record.pick.place.scheme2.otherbackgrounds.bipiper.v0112.10",
    "record.pick.place.scheme2.otherbackgrounds.bipiper.v0112.9",
    "record.pick.place.scheme2.bipiper.v0112.2",
    # 0113 ~3.34h
    # "record.pick.place.move.bipiper.v0113.1",
    # "record.pick.place.move.bipiper.v0113.2",
    # "record.pick.place.move.bipiper.v0113.4",
    # "record.pick.place.move.bipiper.v0113.5",
    # "record.pick.place.move.bipiper.v0113.6",
    "record.pick.place.move.bipiper.v0113.7",
    # "record.pick.place.move.bipiper.v0113.8",
    # v0114 ~1.91h
    "record.pick.place.move.targetintray.bipiper.v0114.1",
    "record.pick.place.move.targetintray.bipiper.v0114.2",
    "record.pick.place.move.targetintray.bipiper.v0114.3",
    # 0115 ~1.74h
    "record.pick.place.move.targetintray.bipiper.v0115.2",
    "record.pick.place.move.targetintray.bipiper.v0115.3",
    "record.pick.place.move.multiheight.bipiper.v0115.4",
    # 0116 ~1.76h
    "record.pick.place.pickout.bipiper.v0116.1",
    "record.pick.place.pickout.bipiper.v0116.2",
    "record.pick.place.pickout.bipiper.v0116.3",
    "record.pick.place.scheme3.multiheight_bipiper.v0116.4",
    "record.pick.place.multiobj.bipiper.v0116.5",
    "record.pick.place.multiobj.bipiper.v0116.6",
    # 0119 ~1.77h
    "record.pick.place.scheme3.multiheight_bipiper.v0119.1",
    "record.pick.place.scheme3.multiheight_bipiper.v0119.2",
    "record.pick.place.scheme3.multiheight_bipiper.v0119.3",
    "record.pick.place.scheme3.multiheight_bipiper.v0119.4",
    "record.pick.place.scheme3.multiheight_bipiper.v0119.5",
    "record.pick.place.scheme3.multiheight_bipiper.v0119.6",
    "record.pick.place.multiobj.bipiper.v0119.7",
    # 0120 ~0.9h
    "record.pick.place.pickout.bipiper.v0120.1",
    "record.pick.place.pickout.bipiper.v0120.2",
    "record.pick.place.scheme3.bipiper.v0120.3",
    # 0121 ~1.31h
    "record.pick.place.scheme3.bipiper.v0121.1",
    "record.pick.place.scheme3.bipiper.v0121.2",
    "record.pick.place.scheme3.bipiper.v0121.3",
    "record.pick.place.scheme3.bipiper.v0121.4",
    "record.pick.place.scheme3.bipiper.v0121.5",
    "record.pick.place.scheme3.bipiper.v0121.6",
    "record.pick.place.multiobj.bipiper.v0121.7",
    # 0122 ~1.05h
    "record.pick.place.scheme2.newobj.bipiper.v0122.2",
    "record.pick.place.scheme2.newobj.bipiper.v0122.3",
    "record.pick.place.scheme2.newobj.bipiper.v0122.4",
    "record.pick.place.scheme2.newobj.bipiper.v0122.5",
    # 0123 ~1.77h
    "record.pick.place.scheme2.newobj.bipiper.v0123.1",
    "record.pick.place.scheme2.newobj.bipiper.v0123.2",
    "record.pick.place.scheme2.newobj.bipiper.v0123.3",
    "record.pick.place.scheme2.newobj.bipiper.v0123.4",
    "record.pick.place.scheme2.newobj.bipiper.v0123.5",
    "record.pick.place.multiob.bipiper.v0123.6",
    # 0126 Total episodes: 158, total time: 1.60 hours
    "record.pick.place.move.newobj.bipiper.v0126.2",
    # "record.pick.place.move.newobj.bipiper.v0126.3",
    "record.pick.place.pickout.newobj.bipiper.v0126.4",
    "record.pick.place.pickout.newobj.bipiper.v0126.5",
    "record.pick.place.restore_container.newobj.bipiper.v0126.6",
    # 0127 Total episodes: 180, total time: 1.00 hours
    "record.pick.place.onlyrestore_container.bipiper.v0127.1",
    "record.pick.place.onlyrestore_container.bipiper.v0127.2",
    "record.pick.place.onlyrestore_container.bipiper.v0127.3",
    "record.pick.place.withrestore_container.bipiper.v0127.4",
    # 0128 Total episodes: 181, total time: 1.01 hours
    "record.pick.place.onlyrestore_container.bipiper.v0128.1",
    "record.pick.place.onlyrestore_container.bipiper.v0128.2",
    "record.pick.place.onlyrestore_container.bipiper.v0128.3",
    "record.pick.place.onlyrestore_container.bipiper.v0128.4",
    "record.pick.place.onlyrestore_container.bipiper.v0128.5",
    "record.pick.place.movetarget.bipiper.v0128.6",
    "record.pick.place.movetarget.bipiper.v0128.7",
    # 0129 Total episodes: 230, total time: 1.63 hours
    "record.pick.place.movetarget.bipiper.v0129.1",
    "record.pick.place.pickout.bipiper.v0129.2",
    "record.pick.place.pickout.bipiper.v0129.3",
    "record.pick.place.onlyrestore_container.bipiper.v0129.4",
    "record.pick.place.onlyrestore_container.bipiper.v0129.5",
    # 0130 Total episodes: 180, total time: 1.61 hours
    "record.pick.place.withrestore_container.bipiper.v0130.1",
    "record.pick.place.withrestore_container.bipiper.v0130.2",
    "record.pick.place.withrestore_container.bipiper.v0130.3",
    "record.pick.place.withrestore_container.bipiper.v0130.4",
    "record.pick.place.withrestore_container.bipiper.v0130.5",
    # 0202 Total episodes: 165, total time: 2.29 hours
    "record.pick.place.onlyrestore_container.multitime.bipiper.v0202.1",
    "record.pick.place.onlyrestore_container.multitime.bipiper.v0202.2",
    "record.pick.place.onlyrestore_container.multitime.bipiper.v0202.3",
    "record.pick.place.withrestore_container.bipiper.v0202.4",
    "record.pick.place.withrestore_container.bipiper.v0202.5",
    "record.pick.place.multiobjpickout.bipiper.v0202.6",
    # 0203 Total episodes: 156, total time: 1.51 hours
    "record.pick.place.withrestore_container.bipiper.v0203.1",
    "record.pick.place.withrestore_container.bipiper.v0203.2",
    "record.pick.place.withrestore_container.bipiper.v0203.3",
    "record.pick.place.movetarget.bipiper.v0203.4",
    "record.pick.place.movetarget.bipiper.v0203.5",
    "record.pick.place.onlyrestore_container.multitime.bipiper.v0203.6",
    # 0204 Total episodes: 161, total time: 2.01 hours
    "record.pick.place.onlyrestore_container.multitime.bipiper.v0204.1",
    "record.pick.place.onlyrestore_container.multitime.bipiper.v0204.2",
    "record.pick.place.withrestore_container.multitime.bipiper.v0204.3",
    "record.pick.place.withrestore_container.multitime.bipiper.v0204.4",
    "record.pick.place.withrestore_container.multitime.bipiper.v0204.5",
    "record.pick.place.multiobjpickout.multitime.bipiper.v0204.6",
    "record.pick.place.multiobjpickout.multitime.bipiper.v0204.7",
    # 0205 Total episodes: 130, total time: 1.21 hours
    "record.pick.place.withrestore_container.multitime.bipiper.v0205.1",
    "record.pick.place.withrestore_container.multitime.bipiper.v0205.2",
    "record.pick.place.withrestore_container.multitime.bipiper.v0205.3",
    # 0206 Total episodes: 70, total time: 1.12 hours
    "record.pick.place.withrestoremulti_container.multitime.bipiper.v0206.1",
    "record.pick.place.withrestoremulti_container.multitime.bipiper.v0206.2",
    # 0209 Total episodes: 230, total time: 2.51 hours
    "record.pick.place.withrestore_container.multitime.bipiper.v0209.1",
    "record.pick.place.withrestore_container.multitime.bipiper.v0209.2",
    "record.pick.place.withrestoremulti_container.multitime.bipiper.v0209.3",
    "record.pick.place.withrestoremulti_container.multitime.bipiper.v0209.4",
    "record.pick.place.withrestoremulti_container.multitime.bipiper.v0209.5",
    "record.pick.place.placeout.bipiper.v0209.6",
    "record.pick.place.placeout.bipiper.v0209.7",
    # 0210 Total episodes: 365, total time: 2.04 hours
    "record.pick.place.placeout.bipiper.v0210.1",
    "record.pick.place.placeout.bipiper.v0210.2",
    "record.pick.place.placeout.bipiper.v0210.3",
    "record.pick.place.placeout.bipiper.v0210.4",
    "record.pick.place.placeout.bipiper.v0210.5",
    "record.pick.place.placeout.bipiper.v0210.6",
    "record.pick.place.placeout.bipiper.v0210.7",
    "record.pick.place.placeout.bipiper.v0210.8",
    "record.pick.place.pickfromcontainer.bipiper.v0210.9",
    "record.pick.place.pickfromcontainer.bipiper.v0210.10",
    "record.pick.place.pickfromcontainer.bipiper.v0210.11",
    # 0211 Total episodes: 83, total time: 0.45 hours
    "record.pick.place.pickfromcontainer.bipiper.v0211.1",
    "record.pick.place.pickfromcontainer.bipiper.v0211.2",
    "record.pick.place.pickfromcontainer.bipiper.v0211.3",
    "record.pick.place.pushotherout.bipiper.v0211.4",
]
REPO_ID += [os.path.join(sub_path, task) for task in tasks]
