import os

REPO_ID = []
ROOT_DIR = "/mnt/"  # 根目录保持/mnt不变
sub_path = "oss_data/anyverse/bipiper/pick_place/anyverse_pickAndplace_record"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
black_list = [
    "record.pick.place.move.bipiper.v0113.8",
    "record.pick.place.scheme3.dirprompt.multiheight_bipiper.v0119.2",
    "record.pick.place.scheme3.dirprompt.bipiper.v0121.5",
    "record.pick.place.pushotherout.bipiper.v0226.1",
    "record.pick.place.scheme3.dirprompt.multiheight_bipiper.v0119.1",
    "record.pick.place.scheme3.dirprompt.bipiper.v0121.2",
    "record.pick.place.scheme3.dirprompt.bipiper.v0121.4",
    "record.pick.place.scheme2.otherbackgrounds.withmiss.bipiper.v0112.12",
    "record.pick.place.move.bipiper.v0113.2",
    "record.pick.place.scheme3.dirprompt.multiheight_bipiper.v0119.6",
    "record.pick.place.move.bipiper.v0113.1",
    "record.pick.place.move.withmiss.bipiper.v0114.5",
    "record.pick.place.move.newobj.bipiper.v0126.3",
    "record.pick.place.scheme3.dirprompt.multiheight_bipiper.v0116.4",
    "record.pick.place.scheme3.dirprompt.multiheight_bipiper.v0119.4",
    "record.pick.place.scheme3.dirprompt.bipiper.v0121.6",
    "record.pick.place.onelyunzip_container.bipiper.v0304.3",
    "record.pick.place.scheme3.dirprompt.bipiper.v0121.1",
    "record.pick.place.scheme3.dirprompt.multiheight_bipiper.v0119.5",
    "record.pick.place.pushotherout.bipiper.v0225.3",
    "record.pick.place.pushotherout.bipiper.v0226.4",
    "record.pick.place.move.bipiper.v0113.4",
    "new_record.pick.place.onelyunzip_container.bipiper.v0304.3",
    "record.pick.place.scheme3.dirprompt.bipiper.v0121.3",
    "record.pick.place.move.bipiper.v0113.6",
    "record.pick.place.move.bipiper.v0113.5",
    "record.pick.place.pushotherout.bipiper.v0226.2",
    "record.pick.place.scheme3.dirprompt.multiheight_bipiper.v0119.3",
    "record.pick.place.scheme3.dirprompt.bipiper.v0120.3",
    "record.pick.place.withunzipandrestore_container.bipiper.v0309.2",
]
REPO_ID += [os.path.join(sub_path, task) for task in tasks if task not in black_list]
print(REPO_ID)
