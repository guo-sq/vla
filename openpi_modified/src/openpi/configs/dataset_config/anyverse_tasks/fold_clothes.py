import os

tasks = [
    # 白T 铺平叠好 55
    "record.clothes.bipiper.v1215.1",
    "record.clothes.bipiper.v1215.2",
    # 白T 铺平叠好 78
    "record.clothes.bipiper.v1216.1",
    "record.clothes.bipiper.v1216.2",
    "record.clothes.bipiper.v1216.3",
    "record.clothes.bipiper.v1216.4",
    "record.clothes.bipiper.v1216.5",
    "record.clothes.bipiper.v1216.6",
    # # 拧转30
    # "record.clothes.bipiper.v0109.1",
    # "record.clothes.bipiper.v0109.2",
    # "record.clothes.bipiper.v0109.3",
    # 上袖口掖进去4
    "record.clothes.bipiper.v0109.8",
    "record.clothes.bipiper.v0109.9",
    # 拧转50
    # "record.clothes.bipiper.v0112.1",
    # "record.clothes.bipiper.v0112.2",
    # "record.clothes.bipiper.v0112.3",
    # 领口朝右10
    # "record.clothes.bipiper.v0112.4",
    # "record.clothes.bipiper.v0112.5",
    # 接管数据 25
    "record.clothes.bipiper.v0112.policy.1",
    "record.clothes.bipiper.v0112.policy.2",
    "record.clothes.bipiper.v0112.policy.3",
    "record.clothes.bipiper.v0112.policy.4",
    "record.clothes.bipiper.v0112.policy.5",
    # 领口朝右30
    # "record.clothes.bipiper.v0113.1",
    # "record.clothes.bipiper.v0113.2",
    # "record.clothes.bipiper.v0113.3",
    # "record.clothes.bipiper.v0113.4",
    # 领口朝上 30
    "record.clothes.bipiper.v0113.6",
    # 领口朝下 30
    "record.clothes.bipiper.v0113.7",
    "record.clothes.bipiper.v0113.8",
    "record.clothes.bipiper.v0113.9",
    "record.clothes.bipiper.v0113.10",
    # 领口朝右 25
    # "record.clothes.bipiper.v0114.1",
    # "record.clothes.bipiper.v0114.2",
    # "record.clothes.bipiper.v0114.3",
    # 人为干扰纠错-领口左下 29
    # "record.clothes.bipiper.v0114.4",
    # "record.clothes.bipiper.v0114.5",
    # "record.clothes.bipiper.v0114.6",
    # 领口朝右 11
    # "record.clothes.bipiper.v0114.7",
    # 人为干扰纠错 袖口重叠 35
    "record.clothes.bipiper.v0115.1",
    "record.clothes.bipiper.v0115.2",
    "record.clothes.bipiper.v0115.3",
    "record.clothes.bipiper.v0115.4",
    "record.clothes.bipiper.v0115.5",
    # 人为干扰-领口左下恢复平行 10
    # "record.clothes.bipiper.v0115.9",
    # 人为干扰-对折干扰衣角袖口 9
    "record.clothes.bipiper.v0115.10",
    # 人为干扰-下摆左向/衣领右向叠好 10
    "record.clothes.bipiper.v0115.11",
    # 人为干扰-叠好放到固定位置 10
    "record.clothes.bipiper.v0115.12",
    # 同一平面褶皱一团 30
    "record.clothes.bipiper.v0116.1",
    # 平面内部扭转 30
    # "record.clothes.bipiper.v0116.2",
    # "record.clothes.bipiper.v0116.3",
    # "record.clothes.bipiper.v0116.4",
    # 平面内部拧转 23
    # "record.clothes.bipiper.v0116.5",
    # "record.clothes.bipiper.v0116.6",
    # "record.clothes.bipiper.v0116.8",
    # 尾部拧在一起无法区分边角 30
    "record.clothes.bipiper.v0116.9",
    # 接管纠错 23
    "record.clothes.bipiper.v0116.policy.1",
    # 接管接错 6
    "record.clothes.bipiper.v0119.policy.1",
    "record.clothes.bipiper.v0119.policy.4",
    # 叠衣服时抽走衣服 复位 50
    "record.clothes.bipiper.v0119.policy.2",
    "record.clothes.bipiper.v0119.policy.3",
    # 全流程叠衣服 20
    # "record.clothes.bipiper.v0120.policy.1",
    # "record.clothes.bipiper.v0120.policy.2",
    # "record.clothes.bipiper.v0120.policy.3",
    # 衣服重叠纠错 8
    "record.clothes.bipiper.v0120.policy.4",
    # 臂从静止叠铺平衣服10
    "record.clothes.bipiper.v0121.policy.1",
    # 空中衣物纠缠-丢下40
    "record.clothes.bipiper.v0121.policy.2",
    "record.clothes.bipiper.v0121.policy.3",
    # 半折叠衣袖重叠 40
    "record.clothes.bipiper.v0121.policy.4",
    "record.clothes.bipiper.v0121.policy.5",
    # 一步到位 衣摆接触桌面
    # 随机扔 20
    "record.clothes.bipiper.v0123.policy.1",
    # 拧转 44
    "record.clothes.bipiper.v0123.policy.2",
    "record.clothes.bipiper.v0123.policy.3",
    "record.clothes.bipiper.v0123.policy.4",
    # 打结 扔衣服 叠好 23
    "record.clothes.bipiper.v0123.policy.5",
    "record.clothes.bipiper.v0123.policy.6",
    # 毛质桌面背景 20
    "record.clothes.bipiper.v0123.policy.7",
    # 绿色桌布背景20
    "record.clothes.bipiper.v0123.policy.8",
    # 一步到位 衣摆接触桌面
    # 随机扔 60
    "record.clothes.bipiper.v0126.policy.1",
    "record.clothes.bipiper.v0126.policy.2",
    "record.clothes.bipiper.v0126.policy.6",
    "record.clothes.bipiper.v0126.policy.7",
    "record.clothes.bipiper.v0126.policy.8",
    "record.clothes.bipiper.v0126.policy.9",
    # 衣服转圈 60
    "record.clothes.bipiper.v0126.policy.3",
    "record.clothes.bipiper.v0126.policy.4",
    "record.clothes.bipiper.v0126.policy.5",
    "record.clothes.bipiper.v0126.policy.10",
    # 随机扔叠好 白衣服 62
    "record.clothes.bipiper.v0127.policy.1",
    "record.clothes.bipiper.v0127.policy.2",
    "record.clothes.bipiper.v0127.policy.3",
    "record.clothes.bipiper.v0127.policy.6",
    "record.clothes.bipiper.v0127.policy.7",
    "record.clothes.bipiper.v0127.policy.9",
    # 随机扔叠好红色T 41
    "record.clothes.bipiper.v0127.policy.4",
    "record.clothes.bipiper.v0127.policy.5",
    "record.clothes.bipiper.v0127.policy.8",
    # 干扰找角铺平 107
    "record.clothes.bipiper.v0128.policy.1",
    "record.clothes.bipiper.v0128.policy.2",
    "record.clothes.bipiper.v0128.policy.3",
    "record.clothes.bipiper.v0128.policy.4",
    "record.clothes.bipiper.v0128.policy.5",
    "record.clothes.bipiper.v0128.policy.6",
    # 随机扔之后铺平 红色T 27
    "record.clothes.bipiper.v0128.policy.7",
    # 随机扔铺平 白色T 28
    "record.clothes.bipiper.v0128.policy.8",
    # 叠好的衣服打乱 白色T 30条
    "record.clothes.bipiper.v0128.r.policy.9",
    "record.clothes.bipiper.v0128.r.policy.10",
    # 白T 叠平臂不回 104
    "record.clothes.bipiper.v0129.policy.1",
    "record.clothes.bipiper.v0129.policy.3",
    "record.clothes.bipiper.v0129.policy.4",
    "record.clothes.bipiper.v0129.policy.7",
    "record.clothes.bipiper.v0129.policy.8",
    "record.clothes.bipiper.v0129.policy.9",
    "record.clothes.bipiper.v0129.policy.10",
    "record.clothes.bipiper.v0129.policy.14",
    # 随机扔红T铺平-臂不回 72
    "record.clothes.bipiper.v0129.policy.2",
    "record.clothes.bipiper.v0129.policy.5",
    "record.clothes.bipiper.v0129.policy.6",
    "record.clothes.bipiper.v0129.policy.11",
    "record.clothes.bipiper.v0129.policy.12",
    # 叠好的衣服打乱 白T 30
    "record.clothes.bipiper.v0129.r.policy.13",
    # 随机扔白T叠平 臂不回 70
    "record.clothes.bipiper.v0130.policy.1",
    "record.clothes.bipiper.v0130.policy.2",
    "record.clothes.bipiper.v0130.policy.6",
    # 随机扔红T 叠平 臂不回 70
    "record.clothes.bipiper.v0130.policy.3",
    "record.clothes.bipiper.v0130.policy.4",
    "record.clothes.bipiper.v0130.policy.5",
    # 白T随机丢 铺平 40
    "record.clothes.bipiper.v0202.1",
    "record.clothes.bipiper.v0202.2",
    # 白T随机丢叠好 20
    "record.clothes.bipiper.v0202.6",
    # 红T 347铺平23 58叠好 40
    "record.clothes.bipiper.v0202.3",
    "record.clothes.bipiper.v0202.4",
    "record.clothes.bipiper.v0202.5",
    "record.clothes.bipiper.v0202.7",
    "record.clothes.bipiper.v0202.8",
    # 叠好的衣服打乱 抽走衣服臂回去 10
    "record.clothes.bipiper.v0203.policy.1",
    # 领口朝右 随机扔 白T39
    "record.clothes.bipiper.v0203.1",
    "record.clothes.bipiper.v0203.2",
    "record.clothes.bipiper.v0203.3",
    "record.clothes.bipiper.v0203.4",
    # 领口朝右 随机扔 红衣服 32
    "record.clothes.bipiper.v0203.5",
    "record.clothes.bipiper.v0203.6",
    "record.clothes.bipiper.v0203.7",
    # 随机 红衣服  15
    "record.clothes.bipiper.v0203.8",
    # 随机 白 10
    "record.clothes.bipiper.v0203.9",
    # 大尺寸 浅蓝色 带logo 铺平叠好 10
    "record.clothes.bipiper.v0203.10",
    "record.clothes.bipiper.v0203.11",
    # 随机 红T20
    "record.clothes.bipiper.v0204.3",
    # 随机 白48
    "record.clothes.bipiper.v0204.1",
    "record.clothes.bipiper.v0204.2",
    "record.clothes.bipiper.v0204.4",
    "record.clothes.bipiper.v0204.5",
    # 青色大T 从铺平叠好 10
    "record.clothes.bipiper.v0204.6",
    # 青色大T lvl3 10
    "record.clothes.bipiper.v0204.7",
    "record.clothes.bipiper.v0204.8",
    "record.clothes.bipiper.v0204.9",
    # 随机扔红T 39
    "record.clothes.bipiper.v0205.1",
    "record.clothes.bipiper.v0205.2",
    "record.clothes.bipiper.v0205.3",
    "record.clothes.bipiper.v0205.4",
    # 领口朝右 白 13
    "record.clothes.bipiper.v0205.5",
    "record.clothes.bipiper.v0205.6",
    "record.clothes.bipiper.v0205.7",
    # 领口朝右 白22
    "record.clothes.bipiper.v0206.1",
    "record.clothes.bipiper.v0206.2",
    "record.clothes.bipiper.v0206.3",
    # 黑篮 lvl1叠好 10
    "record.clothes.bipiper.v0206.4",
    # 黑篮 lvl3叠好 10
    "record.clothes.bipiper.v0206.5",
    # 黑篮 lvl5 10
    "record.clothes.bipiper.v0206.6",
    "record.clothes.bipiper.v0206.7",
    # 青色大T 从铺平叠好10
    "record.clothes.bipiper.v0206.8",
    # 青色大T lvl3 20
    "record.clothes.bipiper.v0206.9",
    "record.clothes.bipiper.v0206.10",
    # 黑色 lv1 10
    "record.clothes.bipiper.v0209.1",
    # 黑色lv3 10
    "record.clothes.bipiper.v0209.2",
    # 黑色lv5 10
    "record.clothes.bipiper.v0209.3",
    # 红色lv1 10
    "record.clothes.bipiper.v0209.4",
    # 红色lv3 10
    "record.clothes.bipiper.v0209.5",
    # 红色lv5 10
    "record.clothes.bipiper.v0209.6",
    # 红色随机叠好 25
    "record.clothes.bipiper.v0209.7",
    "record.clothes.bipiper.v0209.8",
    "record.clothes.bipiper.v0209.9",
    "record.clothes.bipiper.v0209.10",
    # 黑色随机叠好 35
    "record.clothes.bipiper.v0209.11",
    "record.clothes.bipiper.v0209.12",
    "record.clothes.bipiper.v0209.13",
    # 黑色随机 80
    "record.clothes.bipiper.v0210.1",
    "record.clothes.bipiper.v0210.2",
    "record.clothes.bipiper.v0210.3",
    "record.clothes.bipiper.v0210.4",
    # 黑色随机 人工干扰 12
    "record.clothes.bipiper.v0210.5",
    # 青色短袖 随机 22
    "record.clothes.bipiper.v0211.1",
    "record.clothes.bipiper.v0211.2",
    # 青色短袖 随机 60
    "record.clothes.bipiper.v0212.1",
    "record.clothes.bipiper.v0212.2",
    "record.clothes.bipiper.v0212.3",
    "record.clothes.bipiper.v0212.4",
    "record.clothes.bipiper.v0212.5",
    "record.clothes.bipiper.v0212.6",
    # 青色短袖 随机 50
    "record.clothes.bipiper.v0213.5",
    "record.clothes.bipiper.v0213.6",
    "record.clothes.bipiper.v0213.7",
]


REPO_ID = []
ROOT_DIR = "/mnt/"  # 根目录保持/mnt不变
sub_path = "oss_data/anyverse/bipiper/fold_clothes"
# tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [os.path.join(sub_path, task) for task in tasks]
print(REPO_ID)
