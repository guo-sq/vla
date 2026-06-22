import numpy as np
from ..sensor import Sensor
from ..configs import SensorConfig
from ..utils import make_sensors_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
import pandas as pd
from PIL import Image, ImageDraw
from scipy.interpolate import griddata

# 帕西尼触觉传感器
import serial
import time
import serial.tools.list_ports
from threading import Event, Lock, Thread
from typing import Optional, Dict, List, Tuple
import os


class PaxiniTactileSensor(Sensor):

    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.config = config
        self.baudrate: int | None = config.baudrate
        self.com_port: str | None = config.com_port

        self.index_or_path = config.index_or_path

        self.thread: Thread | None = None
        self.heat_map_thread: Thread | None = None
        self.stop_event: Event | None = None
        self._is_connected_flag = False  # 使用私有属性避免递归
        self.latest_frame = {}  # 存储传感器数据
        self.latest_heat_map_frame = {}  # 存储热力图数据
        self.new_frame_event: Event = Event()
        self.new_data_available_event: Event = Event()
        self.new_heat_map_frame_event: Event = Event()
        self.frame_lock: Lock = Lock()
        self.last_valid_frame = {}
        self.last_valid_heat_map_frame = None  # 存储最近有效的热力图帧
        self.ser = None
        self.cycle_read_running = False
        self.heat_map_cycle_read_running = False
        self.module_cycle_read_running = False  # 模组循环读取状态标志
        self.calibration_state = "未标定"
        self.distribution_points_cache = {}  # 存储传感器分布力点数的缓存
        self.sensor_point_counts = {}  # 缓存每个传感器的点数，避免重复查询
        self.num_sensors = 0
        # 传感器模组配置 (地址0500-05A7)
        self.module_count = 28  # 28个传感器模组
        self.module_force_bytes = 6  # 每个模组合力数据: 3轴×2字节
        self.module_total_bytes = self.module_count * self.module_force_bytes  # 168字节

        # 掌心传感器特殊配置 - 只解析前9个分布力点
        self.palm_sensor_limit = 9  # 掌心传感器最多解析9个点

        self.connected_sensors = []  # 存储连接成功的传感器名称
        #heatmap相关配置
        self.X_row = None  # 存储传感器点位X坐标的数组
        self.Y_column = None  # 存储传感器点位Y坐标的数组
        self.IMAGE_SIZE = (75, 90)  # 输出图片尺寸（宽度，高度）像素
        self.GRID_DENSITY = 90  # 插值网格密度
        self.INTERPOLATION_METHOD = "cubic"  # 插值方法：'linear', 'cubic', 'nearest'
        self.BACKGROUND_COLOR = (0, 0, 0)  # plasma色系的最小值颜色
        self.force_min = 0  # 力值最小值
        self.force_max = 255  # 力值最大值
        self.xi = None  # 插值网格X坐标
        self.yi = None  # 插值网格Y坐标
        self.interpolator = None  # 预计算的cubic插值器
        self.xi_grid = None  # 预计算的meshgrid x坐标
        self.yi_grid = None  # 预计算的meshgrid y坐标
        self.heat_map_buffer = None  # 预分配的热力图输出缓冲区
        # 传感器模组顺序
        self.module_names = [
            # 大拇指
            "大拇指近节",
            "大拇指中节",
            "大拇指指尖",
            "大拇指指甲",
            # 食指
            "食指近节",
            "食指中节",
            "食指指尖",
            "食指指甲",
            # 中指
            "中指近节",
            "中指中节",
            "中指指尖",
            "中指指甲",
            # 无名指
            "无名指近节",
            "无名指中节",
            "无名指指尖",
            "无名指指甲",
            # 小拇指
            "小拇指近节",
            "小拇指中节",
            "小拇指指尖",
            "小拇指指甲",
            # 掌心
            "掌心1",
            "掌心2",
            "掌心3",
            "掌心4",
            "掌心5",
            "掌心6",
            "掌心7",
            "掌心8",
        ]

        # 传感器分布力地址区间映射
        self.distribution_addr_ranges = {
            # 大拇指
            (0x1000, 0x11FF): "大拇指近节",
            (0x1200, 0x13FF): "大拇指中节",
            (0x1400, 0x15FF): "大拇指指尖",
            (0x1600, 0x17FF): "大拇指指甲",
            # 食指
            (0x1800, 0x19FF): "食指近节",
            (0x1A00, 0x1BFF): "食指中节",
            (0x1C00, 0x1DFF): "食指指尖",
            (0x1E00, 0x1FFF): "食指指甲",
            # 中指
            (0x2000, 0x21FF): "中指近节",
            (0x2200, 0x23FF): "中指中节",
            (0x2400, 0x25FF): "中指指尖",
            (0x2600, 0x27FF): "中指指甲",
            # 无名指
            (0x2800, 0x29FF): "无名指近节",
            (0x2A00, 0x2BFF): "无名指中节",
            (0x2C00, 0x2DFF): "无名指指尖",
            (0x2E00, 0x2FFF): "无名指指甲",
            # 小拇指
            (0x3000, 0x31FF): "小拇指近节",
            (0x3200, 0x33FF): "小拇指中节",
            (0x3400, 0x35FF): "小拇指指尖",
            (0x3600, 0x37FF): "小拇指指甲",
            # 掌心
            (0x3800, 0x38FF): "掌心1",
            (0x3900, 0x39FF): "掌心2",
            (0x3A00, 0x3AFF): "掌心3",
            (0x3B00, 0x3BFF): "掌心4",
            (0x3C00, 0x3CFF): "掌心5",
            (0x3D00, 0x3DFF): "掌心6",
            (0x3E00, 0x3EFF): "掌心7",
            (0x3F00, 0x3FFF): "掌心8",
        }

        # 传感器模组分布力点数地址映射
        self.distribution_points_addrs = {
            # 大拇指
            "大拇指近节": 0x0030,
            "大拇指中节": 0x0032,
            "大拇指指尖": 0x0034,
            "大拇指指甲": 0x0036,
            # 食指
            "食指近节": 0x0038,
            "食指中节": 0x003A,
            "食指指尖": 0x003C,
            "食指指甲": 0x003E,
            # 中指
            "中指近节": 0x0040,
            "中指中节": 0x0042,
            "中指指尖": 0x0044,
            "中指指甲": 0x0046,
            # 无名指
            "无名指近节": 0x0048,
            "无名指中节": 0x004A,
            "无名指指尖": 0x004C,
            "无名指指甲": 0x004E,
            # 小拇指
            "小拇指近节": 0x0050,
            "小拇指中节": 0x0052,
            "小拇指指尖": 0x0054,
            "小拇指指甲": 0x0056,
            # 掌心
            "掌心1": 0x0066,
            "掌心2": 0x0068,
            "掌心3": 0x006A,
            "掌心4": 0x006C,
            "掌心5": 0x006E,
            "掌心6": 0x0070,
            "掌心7": 0x0072,
            "掌心8": 0x0074,
        }

        # 传感器状态地址定义
        self.sensor_status_addrs = {
            0x0010: "大拇指和食指传感器",
            0x0011: "中指和无名指传感器",
            0x0012: "小拇指和掌心1-4传感器",
            0x0013: "掌心5-8传感器",
        }

        # 循环读取参数
        self.loop_read_addr = 0x1000  # 默认读取地址-大拇指近节
        self.loop_read_len = 60  # 默认读取长度
        self.loop_read_interval = 0.05  # 读取间隔（秒）

    def log(self, message: str):
        """控制台日志输出"""
        print(
            f"[{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}] {message}"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path})"

    def SerialPortConnect(self) -> bool:
        """连接串口设备"""
        if self.ser and self.ser.is_open:
            self.log("已连接设备")
            self._is_connected_flag = True
            return True

        try:
            self.ser = serial.Serial(
                port=self.com_port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.005,  # 接收完字节时间，不超时不等待
                write_timeout=0.005,  # 写超时同步调整
                dsrdtr=False,
                rtscts=False,
                xonxoff=False,  # 关闭流控，减少串口开销
            )
            if self.ser.is_open:
                self.ser.flushInput()  # 打开串口立即清空缓冲区，防止残留数据
                self.ser.flushOutput()
                self.log(f"成功连接到 {self.com_port}，波特率 {self.baudrate}")
                self._is_connected_flag = True
                self.check_sensor_status()
                # 缓存传感器点数，避免每次读取都查询
                self._cache_sensor_point_counts()
                # self.start_calibration()
                self.load_sensor_point_layout()
                return True
        except Exception as e:
            self.log(f"连接失败: {str(e)}")
            self._is_connected_flag = False
        return False
    def load_sensor_point_layout(self):
        try:
            # 读取Excel文件
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if self.points_per_module == 77:
                EXCEL_PATH = os.path.join(current_dir, "config", "M3025.xlsx")
            elif self.points_per_module == 25:  
                EXCEL_PATH = os.path.join(current_dir, "config", "S1610.xlsx")
            # if self.points_per_module ==77:
            #     EXCEL_PATH="/config/M3025.xlsx"
            # elif self.points_per_module == 25:
            #     EXCEL_PATH="/config/S1610.xlsx"
            df = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1")

            # 提取数据列
            self.X_row = df["X"].values
            self.Y_column = df["Y"].values
            print(f"数据点数：{len(self.X_row)}")
            print(f"数据点数：{len(self.Y_column)}")
            self.xi = np.linspace(self.X_row.min(), self.X_row.max(), self.IMAGE_SIZE[0])
            self.yi = np.linspace(self.Y_column.min(), self.Y_column.max(), self.IMAGE_SIZE[1])
            # 数据有效性检查
            assert (
                not np.any(np.isnan(self.X_row))
                and not np.any(np.isnan(self.Y_column))
            ), "数据包含空值！"

            print(f"✅ 数据加载成功！共{len(self.X_row)}个数据点")
            print(f"📊 数据范围：")
            print(f"   X轴: {self.X_row.min():.2f} ~ {self.X_row.max():.2f} mm")
            print(f"   Y轴: {self.Y_column.min():.2f} ~ {self.Y_column.max():.2f} mm")

            # 预计算插值相关数据，加速热力图生成
            self._init_interpolation_cache()

        except Exception as e:
            print(f"❌ 数据加载失败：{str(e)}")

    def _init_interpolation_cache(self):
        """预计算插值相关数据，加速热力图生成"""
        if self.X_row is None or self.Y_column is None or self.xi is None or self.yi is None:
            return

        from scipy.spatial import Delaunay
        import time

        start_time = time.time()

        # 预计算meshgrid
        self.xi_grid, self.yi_grid = np.meshgrid(self.xi, self.yi)

        # 预计算Delaunay三角剖分（这是griddata中最耗时的部分）
        points = np.vstack([self.X_row, self.Y_column]).T
        try:
            self._tri = Delaunay(points)
        except Exception as e:
            print(f"⚠️ 预计算三角剖分失败: {e}")
            self._tri = None

        # 预分配热力图输出缓冲区
        num_modules = self.num_modules if self.num_modules else 28
        canvas_width = self.IMAGE_SIZE[0] * num_modules
        canvas_height = self.IMAGE_SIZE[1]
        self.heat_map_buffer = np.zeros(
            (canvas_height, canvas_width, 3), dtype=np.uint8
        )

        elapsed = (time.time() - start_time) * 1000
        print(f"⚡ 插值缓存初始化完成，耗时 {elapsed:.2f}ms")
        print(f"   - meshgrid预计算: ({self.xi_grid.shape})")
        print(f"   - 三角剖分: {'成功' if self._tri is not None else '失败'}")
        print(f"   - 输出缓冲区: {self.heat_map_buffer.shape}")

    def start_calibration(self, calib_cmd_hex="55AA00170200010001E6"):
        """发送标定指令"""
        if not self.ser or not self.ser.is_open:
            self.log("请先连接设备")
            return
        self.log("===== 开始标定流程 =====")
        if not calib_cmd_hex:
            self.log("标定命令帧不能为空")
            return
        try:
            calib_frame = bytes.fromhex(calib_cmd_hex)
            self.ser.write(calib_frame)
            self.log(f"发送标定命令: {calib_cmd_hex}")
            time.sleep(1)
            response = self.ser.read(256)
            if not response:
                self.log("未收到标定响应，可能超时")
                return
            parsed = self.parse_response_frame(response)
            if parsed:
                self.log(
                    f"标定响应: 地址={parsed['reg_addr']:04X}, 数据={parsed['data'].hex()}"
                )
                if parsed["data"] == b"\x00":
                    self.log("标定成功！")
                else:
                    self.log(f"标定失败，响应码: {parsed['data'].hex()}")
            else:
                self.log(f"标定响应解析失败: {response.hex()}")
        except Exception as e:
            self.log(f"标定操作错误: {str(e)}")


    @property
    def is_connected(self) -> bool:
        """检查传感器是否已连接"""
        return (self.ser is not None and self.ser.is_open) or self._is_connected_flag

    def get_address_by_sensor_name(
        self, sensor_name: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """根据传感器名称获取对应的地址范围"""
        for (start, end), name in self.distribution_addr_ranges.items():
            if name == sensor_name:
                return (start, end)
        return (None, None)

    def check_sensor_status(self):
        """检查传感器连接状态"""
        self.connected_sensors = []
        if not self.ser or not self.ser.is_open:
            self.log("请先连接设备")
            return

        self.log("===== 开始检查传感器连接状态 =====")
        for addr in self.sensor_status_addrs:
            try:
                reg_addr = addr
                read_len = 1

                old_len = self.loop_read_len
                self.loop_read_len = read_len

                request_frame = self.build_request_frame(0x03, reg_addr, b"")
                self.ser.write(request_frame)
                # self.log(
                #     f"发送状态请求: 地址={reg_addr:04X} ({self.sensor_status_addrs[addr]}), 帧={request_frame.hex()}"
                # )

                time.sleep(0.01)
                response = self.ser.read(128)
                self.loop_read_len = old_len

                if not response:
                    self.log(f"未收到地址{reg_addr:04X}的响应")
                    continue

                parsed = self.parse_response_frame(response)
                if parsed and parsed["func_code"] == 0x03 and parsed["data_len"] == 1:
                    status_byte = parsed["data"][0]
                    # self.log(
                    #     f"地址{reg_addr:04X}状态字节: 0x{status_byte:02X} ({bin(status_byte)[2:].zfill(8)})"
                    # )
                    if addr == 0x0010:
                        self._parse_addr_0010(status_byte)
                    elif addr == 0x0011:
                        self._parse_addr_0011(status_byte)
                    elif addr == 0x0012:
                        self._parse_addr_0012(status_byte)
                    elif addr == 0x0013:
                        self._parse_addr_0013(status_byte)
                else:
                    self.log(f"地址{reg_addr:04X}状态读取失败")

            except Exception as e:
                self.log(f"读取地址{reg_addr:04X}错误: {str(e)}")

        self.log("===== 传感器连接状态检查完成 =====")
        self.log(
            f"共检测到 {len(self.connected_sensors)} 个连接成功的传感器: {self.connected_sensors}"
        )

    def _parse_addr_0010(self, status_byte):
        thumb_sensors = ["大拇指近节", "大拇指中节", "大拇指指尖", "大拇指指甲"]
        index_sensors = ["食指近节", "食指中节", "食指指尖", "食指指甲"]
        # self.log("  大拇指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << i)) else "未连接"
            # self.log(f"    {thumb_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(thumb_sensors[i])
        # self.log("  食指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << (i + 4))) else "未连接"
            # self.log(f"    {index_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(index_sensors[i])

    def _parse_addr_0011(self, status_byte):
        middle_sensors = ["中指近节", "中指中节", "中指指尖", "中指指甲"]
        ring_sensors = ["无名指近节", "无名指中节", "无名指指尖", "无名指指甲"]
        # self.log("  中指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << i)) else "未连接"
            # self.log(f"    {middle_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(middle_sensors[i])
        # self.log("  无名指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << (i + 4))) else "未连接"
            # self.log(f"    {ring_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(ring_sensors[i])

    def _parse_addr_0012(self, status_byte):
        pinky_sensors = ["小拇指近节", "小拇指中节", "小拇指指尖", "小拇指指甲"]
        palm_sensors = ["掌心1", "掌心2", "掌心3", "掌心4"]
        # self.log("  小拇指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << i)) else "未连接"
            # self.log(f"    {pinky_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(pinky_sensors[i])
        # self.log("  掌心传感器(1-4)状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << (i + 4))) else "未连接"
            # self.log(f"    {palm_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(palm_sensors[i])

    def _parse_addr_0013(self, status_byte):
        palm_sensors = ["掌心5", "掌心6", "掌心7", "掌心8"]
        # self.log("  掌心传感器(5-8)状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << i)) else "未连接"
            # self.log(f"    {palm_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(palm_sensors[i])

    def _cache_sensor_point_counts(self):
        """预读取所有连接传感器的点数并缓存，避免每次循环查询"""
        if not self.ser or not self.ser.is_open:
            return
        if not self.connected_sensors:
            return

        self.log("===== 开始缓存传感器点数 =====")
        cached_count = 0
        for sensor in self.connected_sensors:
            points_addr = self.distribution_points_addrs.get(sensor)
            if not points_addr:
                continue
            try:
                old_len = self.loop_read_len
                self.loop_read_len = 2
                request_frame = self.build_request_frame(0x03, points_addr, b"")
                self.ser.write(request_frame)
                self.ser.flushOutput()
                response = self.ser.read(128)
                self.loop_read_len = old_len

                if response:
                    parsed = self.parse_response_frame(response)
                    if parsed and parsed["func_code"] == 0x03 and parsed["data_len"] == 2:
                        point_count = int.from_bytes(parsed["data"], byteorder="little", signed=False)
                        if sensor.startswith("掌心"):
                            point_count = min(point_count, self.palm_sensor_limit)
                        self.sensor_point_counts[sensor] = point_count
                        cached_count += 1
            except Exception:
                pass  # 缓存失败不阻塞，后续读取时会回退到动态查询

        self.log(f"点数缓存完成: {cached_count}/{len(self.connected_sensors)} 个传感器")

    def build_request_frame(
        self, func_code: int, reg_addr: int, data: bytes = b""
    ) -> bytes:
        """构建请求帧"""
        head = b"\x55\xaa"
        reserved = b"\x00"
        reg_addr_bytes = reg_addr.to_bytes(2, byteorder="little", signed=False)

        if func_code == 0x10:  # 写操作
            data_len = len(data)
        else:  # 读操作
            data_len = self.loop_read_len

        data_len_bytes = data_len.to_bytes(2, byteorder="little", signed=False)
        frame_body = (
            head
            + reserved
            + bytes([func_code])
            + reg_addr_bytes
            + data_len_bytes
            + data
        )
        lrc = self.calculate_lrc(frame_body)
        return frame_body + bytes([lrc])

    def calculate_lrc(self, data: bytes) -> int:
        """计算LRC校验"""
        lrc = 0
        for byte in data:
            lrc = (lrc + byte) & 0xFF
        lrc = ((~lrc) + 1) & 0xFF
        return lrc

    def parse_response_frame(self, frame: bytes) -> Optional[Dict]:
        """解析响应帧"""
        if len(frame) < 8 or frame[:2] != b"\xaa\x55":
            return None

        return {
            "reserved": frame[2],
            "func_code": frame[3],
            "reg_addr": int.from_bytes(frame[4:6], byteorder="little"),
            "data_len": int.from_bytes(frame[6:8], byteorder="little"),
            "data": frame[8:-1],
        }

    def parse_normal_force_data(
        self, data: bytes, addr: int, source: str = ""
    ) -> List[Dict]:
        """解析分布力数据"""
        sensor_name = None
        for (start, end), name in self.distribution_addr_ranges.items():
            if start <= addr <= end:
                sensor_name = name
                break
        parsed = []
        result_list = []  # Define result_array to store parsed results
        total_groups = len(data) // 3
        # ===== 掌心传感器只解析前9个点 =====
        if sensor_name and sensor_name.startswith("掌心"):
            total_groups = min(total_groups, self.palm_sensor_limit)

        for i in range(total_groups):
            offset = i * 3
            b1, b2, b3 = data[offset], data[offset + 1], data[offset + 2]
            val1 = b1 if b1 <= 127 else b1 - 256
            val2 = b2 if b2 <= 127 else b2 - 256
            val3 = b3
            scaled1 = round(val1 * 0.1, 1)
            scaled2 = round(val2 * 0.1, 1)
            scaled3 = round(val3 * 0.1, 1)
            parsed.append(
                {
                    "index": i,
                    "raw_bytes": (b1, b2, b3),
                    "raw_hex": (f"0x{b1:02X}", f"0x{b2:02X}", f"0x{b3:02X}"),
                    "converted": (val1, val2, val3),
                    "scaled": (scaled1, scaled2, scaled3),
                }
            )
            result_list.append((val1 + 128, val2 + 128, val3))
        # 打印【前1组】力值数据
        # if parsed:
        #     for group in parsed:
        #         self.log(
        #             f"  组{group['index']:02d} | 物理值: X={group['scaled'][0]:5.1f}N, Y={group['scaled'][1]:5.1f}N, Z={group['scaled'][2]:5.1f}N"
        #         )
        return parsed, result_list

    def _get_sensor_data_len(self, sensor: str, end_addr: int | None, start_addr: int | None) -> int:
        """获取传感器数据长度，优先使用缓存的点数"""
        if not self.ser or not self.ser.is_open:
            return 0
        # 优先使用缓存的点数
        if sensor in self.sensor_point_counts:
            point_count = self.sensor_point_counts[sensor]
            return point_count * 3

        # 缓存未命中，回退到查询（首次或传感器重新连接）
        points_addr = self.distribution_points_addrs.get(sensor)
        if not points_addr:
            return 0

        try:
            old_len = self.loop_read_len
            self.loop_read_len = 2
            request_frame = self.build_request_frame(0x03, points_addr, b"")
            self.ser.write(request_frame)
            self.ser.flush()

            # 使用更短的超时轮询等待数据
            timeout_count = 0
            while self.ser.in_waiting < 8 and timeout_count < 10:
                time.sleep(0.001)
                timeout_count += 1

            response = self.ser.read(self.ser.in_waiting or 128)
            self.loop_read_len = old_len

            if response:
                parsed = self.parse_response_frame(response)
                if parsed and parsed["func_code"] == 0x03 and parsed["data_len"] == 2:
                    point_count = int.from_bytes(parsed["data"], byteorder="little", signed=False)
                    if sensor.startswith("掌心"):
                        point_count = min(point_count, self.palm_sensor_limit)
                    # 更新缓存
                    self.sensor_point_counts[sensor] = point_count
                    return point_count * 3
        except Exception:
            pass
        return 0

    def _read_sensor_data_fast(self, sensor: str, start_addr: int, data_len: int) -> tuple:
        """快速读取单个传感器数据，优化串口读取逻辑"""
        if not self.ser or not self.ser.is_open:
            return None, None
        try:
            max_possible_len = 0x200  # 每个传感器地址范围 0x200
            if data_len > max_possible_len:
                data_len = max_possible_len

            old_len = self.loop_read_len
            self.loop_read_len = data_len
            request_frame = self.build_request_frame(0x03, start_addr, b"")
            self.ser.write(request_frame)
            self.ser.flush()

            # 计算期望返回的字节数: 包头8 + 数据 + 校验1
            expected_len = 8 + data_len + 1

            # 使用 in_waiting 轮询等待数据，比固定超时更高效
            timeout_count = 0
            max_wait = min(50, 5 + data_len // 10)  # 动态超时，根据数据量调整
            while self.ser.in_waiting < expected_len and timeout_count < max_wait:
                time.sleep(0.001)  # 1ms 轮询
                timeout_count += 1

            # 只读取实际可用的数据
            available = self.ser.in_waiting
            if available > 0:
                data_response = self.ser.read(min(available, 1024))
            else:
                data_response = b""

            self.loop_read_len = old_len

            if not data_response:
                return None, None

            data_parsed = self.parse_response_frame(data_response)
            if data_parsed and data_parsed["func_code"] == 0x03:
                return self.parse_normal_force_data(data_parsed["data"], start_addr, source=f"[{sensor}] ")
            return None, None
        except Exception:
            return None, None

    def read_connected_sensors(self):
        """读取所有连接成功的传感器分布力数据 - 优化版"""
        if not self.ser or not self.ser.is_open:
            self.log("请先连接设备")
            return None, []

        if not self.connected_sensors:
            self.log("没有检测到连接成功的传感器，无法读取分布力数据")
            raise RuntimeError("未检测到任何连接的传感器")

        # 按手指分组的顺序读取（地址连续，但暂不实现真正的批量读取）
        sensor_order = [
            "大拇指近节", "大拇指中节", "大拇指指尖", "大拇指指甲",
            "食指近节", "食指中节", "食指指尖", "食指指甲",
            "中指近节", "中指中节", "中指指尖", "中指指甲",
            "无名指近节", "无名指中节", "无名指指尖", "无名指指甲",
            "小拇指近节", "小拇指中节", "小拇指指尖", "小拇指指甲",
            "掌心1", "掌心2", "掌心3", "掌心4", "掌心5", "掌心6", "掌心7", "掌心8",
        ]

        # 按固定顺序读取，确保结果列表顺序一致
        all_results = {}
        all_results_list = []
        success_count = 0
        fail_count = 0

        for sensor in sensor_order:
            if sensor not in self.connected_sensors:
                continue

            start_addr, end_addr = self.get_address_by_sensor_name(sensor)
            if not start_addr:
                fail_count += 1
                all_results_list.append(self._get_fallback_data(len(all_results_list)))
                continue

            # 获取数据长度（使用缓存）
            data_len = self._get_sensor_data_len(sensor, end_addr, start_addr)
            if data_len <= 0:
                # 回退：使用配置的默认点数
                points = self.points_per_module if self.points_per_module else 25
                if sensor.startswith("掌心"):
                    points = min(points, self.palm_sensor_limit)
                data_len = points * 3

            # 读取传感器数据
            parsed_data, result_list = self._read_sensor_data_fast(sensor, start_addr, data_len)

            if parsed_data is not None and result_list:
                all_results[sensor] = parsed_data
                all_results_list.append(result_list)
                success_count += 1
            else:
                fail_count += 1
                all_results_list.append(self._get_fallback_data(len(all_results_list)))

        if fail_count > 0 and fail_count == len(self.connected_sensors):
            self.log(f"警告: 所有 {fail_count} 个传感器读取失败，请检查连接")
        elif fail_count > 0:
            self.log(f"警告: 有 {fail_count} 个传感器读取失败")

        self.last_valid_frame = all_results_list.copy()
        return all_results, all_results_list

    def _get_fallback_data(self, index: int):
        """获取回退数据（上次有效数据或零值）"""
        if self.last_valid_frame and len(self.last_valid_frame) > index:
            return self.last_valid_frame[index]
        points = self.points_per_module if self.points_per_module else 25
        return np.zeros((1, points, 3), dtype=np.uint8).tolist()

    def read_registers(self):
        if (self.ser and self.ser.is_open) or self.is_connected:
            try:
                # reg_addr = self.loop_read_addr
                # request_frame = self.build_request_frame(0x03, reg_addr, b"")
                # self.ser.write(request_frame)
                # self.ser.flushOutput()  # 立即发送，不缓存

                # # 响应帧结构：包头8字节 + 自定义长度数据 + 校验位1字节
                # read_bytes = 8 + self.loop_read_len + 1
                # response = self.ser.read(read_bytes)

                # self.ser.flushInput()  # 读完立即清空缓冲区，无残留

                # parsed = self.parse_response_frame(response)
                # if parsed and parsed["func_code"] == 0x03:
                #     self.parse_normal_force_data(parsed["data"], reg_addr)
                results, results_list = self.read_connected_sensors()
                # 处理返回 None 的情况
                if results is None:
                    results = {}
                if results_list is None:
                    results_list = []
                # 打印汇总信息
                # print(
                #     f"\n[{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}] 本次读取汇总:"
                # )
                self.num_sensors = 0
                for sensor_name, force_data in results.items():
                    if force_data:
                        # 计算该传感器的最大值
                        max_x = max(point["scaled"][0] for point in force_data)
                        max_y = max(point["scaled"][1] for point in force_data)
                        max_z = max(point["scaled"][2] for point in force_data)
                        # print(
                        #     f"  {sensor_name}: 最大力值 X={max_x:5.1f}N, Y={max_y:5.1f}N, Z={max_z:5.1f}N"
                        # )
                        self.num_sensors += 1

                # print("-" * 60)
                if len(results_list) < self.num_sensors:
                    for _ in range(self.num_sensors - len(results_list)):
                        results_list.append(
                            np.zeros(
                                (1, self.points_per_module, 3), dtype=np.uint8
                            ).tolist()
                        )

                results_array = np.array(results_list, dtype=np.uint8)
                # print(results_array)
            except Exception as e:
                results_array = np.zeros(
                    (self.num_sensors, self.points_per_module, 3), dtype=np.uint8
                )  # 初始化为默认值
                pass  # 高频场景下，异常捕获不打印日志，避免拖慢速度
            return results_array
        else:
            raise DeviceNotConnectedError(f"{self} is not connected.")

    def disconnect(self):
        """断开连接"""
        if self.ser and self.ser.is_open:
            self.stop_cycle_read()
            self.stop_module_cycle_read()
            if not self.is_connected and self.thread is None:
                raise DeviceNotConnectedError(f"{self} not connected.")
            if self.thread is not None:
                self._stop_read_thread()

            self.ser.close()
            self.log("已断开连接")
        else:
            self.log("未连接设备")

    def stop_module_cycle_read(self):
        """停止循环读取传感器模组合力 - 仅修改状态标志位"""
        self.module_cycle_read_running = False
        self.log("已停止循环读取传感器模组数据")

    def start_cycle_read(self):
        self.cycle_read_running = True
        # 新开线程执行循环读取，不阻塞主线程的按键停止
        Thread(target=self.read_registers, daemon=True).start()

    def stop_cycle_read(self):
        """停止循环读取"""
        self.cycle_read_running = False
        self.log("已停止循环读取")

    def _start_read_thread(self):
        self.cycle_read_running = True
        # 新开线程执行循环读取，不阻塞主线程的按键停止
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()
    def _start_heat_map_read_thread(self):
        self.heat_map_cycle_read_running = True
        # 新开线程执行循环读取，不阻塞主线程的按键停止
        if self.heat_map_thread is not None and self.heat_map_thread.is_alive():
            self.heat_map_thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.heat_map_thread = Thread(target=self._draw_heat_map_loop, args=(), name=f"{self}_heat_map_read_loop")
        self.heat_map_thread.daemon = True
        self.heat_map_thread.start()

    def _stop_read_thread(self):
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None
        self.log("已停止读取线程")

    def _read_loop(self):
        if self.stop_event is None:
            return
        while not self.stop_event.is_set():
            try:
                result = self.read_registers()
                with self.frame_lock:
                    self.latest_frame = result
                self.new_frame_event.set()
                self.new_data_available_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                self.log(
                    f"Error reading frame in background thread for {self}: {e}"
                )

        return
    def _draw_heat_map_loop(self):
        if self.stop_event is None:
            return
        while not self.stop_event.is_set():
            self.new_data_available_event.wait(timeout=0.01)  # 等待新数据可用
            self.new_data_available_event.clear()
            try:
                result = self.read_heat_map()
                with self.frame_lock:
                    self.latest_heat_map_frame = result
                self.new_heat_map_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                self.log(
                    f"Error reading frame in background thread for {self}: {e}"
                )


    def async_read_ori(self, timeout_ms: float = 200) -> np.ndarray:
        """非阻塞读取，立即返回最新数据（新数据或上次有效数据）"""
        if not self.is_connected:
            self.log("未连接设备")
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 启动后台读取线程（如未启动）
        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        # 检查是否有新数据，有新数据则更新缓存，否则直接返回旧数据
        if self.new_frame_event.is_set():
            with self.frame_lock:
                self.last_valid_frame = self.latest_frame
                self.new_frame_event.clear()

        # 直接返回最新有效数据（不会阻塞等待）
        if self.last_valid_frame is None or len(self.last_valid_frame) == 0:
            # 初始时还没有数据，返回零值数组
            # 使用默认值：28个传感器模组，每个模组25或77个点
            num_sensors = self.num_modules if self.num_modules else 28
            points_per_module = self.points_per_module if self.points_per_module else 25
            return np.zeros(
                (num_sensors, points_per_module, 3), dtype=np.uint8
            )

        # 确保返回 ndarray
        if isinstance(self.last_valid_frame, np.ndarray):
            return self.last_valid_frame
        return np.array(self.last_valid_frame, dtype=np.uint8)
    def async_read_heat_map(self, timeout_ms: float = 200) -> np.ndarray:
        """非阻塞读取热力图，立即返回最新数据（新数据或上次有效数据）"""
        if not self.is_connected:
            self.log("未连接设备")
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 启动后台热力图读取线程（如未启动）
        if self.heat_map_thread is None or not self.heat_map_thread.is_alive():
            self._start_heat_map_read_thread()

        # 检查是否有新热力图数据，有新数据则更新缓存，否则直接返回旧数据
        if self.new_heat_map_frame_event.is_set():
            with self.frame_lock:
                self.last_valid_heat_map_frame = self.latest_heat_map_frame
                self.new_heat_map_frame_event.clear()

        # 直接返回最新有效热力图数据（不会阻塞等待）
        if self.last_valid_heat_map_frame is None or len(self.last_valid_heat_map_frame) == 0:
            # 初始时还没有数据，返回零值图像
            num_modules = self.num_modules if self.num_modules else 28
            canvas_width = self.IMAGE_SIZE[0] * num_modules
            canvas_height = self.IMAGE_SIZE[1]
            return np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # 确保返回 ndarray
        if isinstance(self.last_valid_heat_map_frame, np.ndarray):
            return self.last_valid_heat_map_frame
        return np.array(self.last_valid_heat_map_frame, dtype=np.uint8)


    def read_heat_map(self, timeout_ms: float = 200) -> np.ndarray:
        """生成热力图 - 优化版，使用预计算插值器"""
        frame = self.latest_frame
        points_per_module = self.points_per_module
        num_modules = self.num_modules
        if points_per_module is None or num_modules is None:
            raise ValueError("points_per_module and num_modules must be set")

        # 如果 frame 为空或不是数组类型，使用零值填充
        if frame is None or isinstance(frame, dict) or len(frame) == 0:
            canvas_width = self.IMAGE_SIZE[0] * num_modules
            canvas_height = self.IMAGE_SIZE[1]
            return np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        grid_width, grid_height = self.IMAGE_SIZE
        canvas_width = grid_width * num_modules
        canvas_height = grid_height

        # 使用预分配的缓冲区或创建新的
        if (self.heat_map_buffer is None or
            self.heat_map_buffer.shape != (canvas_height, canvas_width, 3)):
            self.heat_map_buffer = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        else:
            self.heat_map_buffer.fill(0)  # 清零重用

        # 检查是否可以使用预计算三角剖分
        use_cached_tri = (hasattr(self, '_tri') and self._tri is not None and
                          self.xi_grid is not None and
                          self.yi_grid is not None)

        for i in range(num_modules):
            if i >= len(frame):
                break
            module_frame = frame[i]

            if use_cached_tri:
                # 使用预计算的三角剖分创建插值器（跳过最耗时的Delaunay计算）
                from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
                if self.INTERPOLATION_METHOD == "cubic":
                    interp_x = CloughTocher2DInterpolator(self._tri, module_frame[:, 0], fill_value=0)
                    interp_y = CloughTocher2DInterpolator(self._tri, module_frame[:, 1], fill_value=0)
                    interp_z = CloughTocher2DInterpolator(self._tri, module_frame[:, 2], fill_value=0)
                else:
                    interp_x = LinearNDInterpolator(self._tri, module_frame[:, 0], fill_value=0)
                    interp_y = LinearNDInterpolator(self._tri, module_frame[:, 1], fill_value=0)
                    interp_z = LinearNDInterpolator(self._tri, module_frame[:, 2], fill_value=0)
                fxi = interp_x(self.xi_grid, self.yi_grid)
                fyi = interp_y(self.xi_grid, self.yi_grid)
                fzi = interp_z(self.xi_grid, self.yi_grid)
            else:
                fxi = griddata(
                    points=(self.X_row, self.Y_column),
                    values=module_frame[:, 0],
                    xi=(self.xi_grid, self.yi_grid),
                    method=self.INTERPOLATION_METHOD,
                    fill_value=0,
                )
                fyi = griddata(
                    points=(self.X_row, self.Y_column),
                    values=module_frame[:, 1],
                    xi=(self.xi_grid, self.yi_grid),
                    method=self.INTERPOLATION_METHOD,
                    fill_value=0,
                )
                fzi = griddata(
                    points=(self.X_row, self.Y_column),
                    values=module_frame[:, 2],
                    xi=(self.xi_grid, self.yi_grid),
                    method=self.INTERPOLATION_METHOD,
                    fill_value=0,
                )

            # 直接写入缓冲区，避免创建中间Image对象
            paste_x = i * grid_width
            self.heat_map_buffer[:, paste_x:paste_x + grid_width, 0] = np.clip(fxi, 0, 255).astype(np.uint8)
            self.heat_map_buffer[:, paste_x:paste_x + grid_width, 1] = np.clip(fyi, 0, 255).astype(np.uint8)
            self.heat_map_buffer[:, paste_x:paste_x + grid_width, 2] = np.clip(fzi, 0, 255).astype(np.uint8)

        return self.heat_map_buffer.copy()
            
            
