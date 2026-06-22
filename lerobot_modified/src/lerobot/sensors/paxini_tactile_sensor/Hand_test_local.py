import serial
import time
import serial.tools.list_ports
import threading
from typing import Optional, Dict, List, Tuple
import os


class HighSpeedCommBoard:
    def __init__(self):
        self.ser = None
        self.cycle_read_running = False
        self.module_cycle_read_running = False  # 模组循环读取状态标志
        self.calibration_state = "未标定"
        self.distribution_points_cache = {}  # 存储传感器分布力点数的缓存

        # 传感器模组配置 (地址0500-05A7)
        self.module_count = 28  # 28个传感器模组
        self.module_force_bytes = 6  # 每个模组合力数据: 3轴×2字节
        self.module_total_bytes = self.module_count * self.module_force_bytes  # 168字节

        # 掌心传感器特殊配置 - 只解析前9个分布力点
        self.palm_sensor_limit = 9  # 掌心传感器最多解析9个点

        self.connected_sensors = []  # 存储连接成功的传感器名称

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

        self.file_path = os.path.join(
            os.path.expanduser("~"), "Desktop", "sensor_data_log.txt"
        )  # 桌面文件路径
        self.init_data_file()  # 初始化文件

    def init_data_file(self):
        """初始化桌面的日志文件，添加表头"""
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(
                    f"【传感器数据日志】开始记录时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(
                    "时间戳 | 传感器名称 | 组号 | X轴(N) | Y轴(N) | Z轴(N) | 原始十六进制值\n"
                )
                f.write("-" * 100 + "\n")

    def save_to_desktop(self, content):
        """追加写入数据到桌面文件"""
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(content + "\n")

    def log(self, message: str):
        """控制台日志输出"""
        print(
            f"[{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}] {message}"
        )

    def list_com_ports(self):
        """列出可用的COM端口"""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.log(f"可用COM端口: {ports}")
        return ports

    def connect(self, com_port: str, baudrate: int = 921600) -> bool:
        """连接串口设备"""
        if self.ser and self.ser.is_open:
            self.log("已连接设备")
            return True

        try:
            self.ser = serial.Serial(
                port=com_port,
                baudrate=baudrate,
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
                self.log(f"成功连接到 {com_port}，波特率 {baudrate}")
                return True
        except Exception as e:
            self.log(f"连接失败: {str(e)}")
        return False

    def disconnect(self):
        """断开连接"""
        if self.ser and self.ser.is_open:
            self.stop_cycle_read()
            self.stop_module_cycle_read()
            self.ser.close()
            self.log("已断开连接")
        else:
            self.log("未连接设备")

    def calculate_lrc(self, data: bytes) -> int:
        """计算LRC校验"""
        lrc = 0
        for byte in data:
            lrc = (lrc + byte) & 0xFF
        lrc = ((~lrc) + 1) & 0xFF
        return lrc

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

    # ========== 1. 获取版本号 ==========
    def parse_version_data(self, data: bytes) -> str:
        """解析版本数据"""
        try:
            return data.decode("ascii").strip()
        except UnicodeDecodeError:
            return f"无法解析的版本数据: {data.hex().upper()}"

    def get_version(self):
        """获取高速通信集成板版本号"""
        if not self.ser or not self.ser.is_open:
            self.log("请先连接设备")
            return

        try:
            version_frame = bytes.fromhex("55AA000300000F00EF")
            self.ser.write(version_frame)
            self.log(f"发送版本号请求: {version_frame.hex().upper()}")

            time.sleep(0.1)
            response = self.ser.read(128)

            if not response:
                self.log("未收到版本号响应")
                return

            self.log(f"收到版本号响应: {response.hex().upper()}")

            parsed = self.parse_response_frame(response)
            if parsed and parsed["func_code"] == 0x03:
                version = self.parse_version_data(parsed["data"])
                self.log(f"设备版本号: {version}")
            else:
                self.log("版本号响应解析失败")

        except Exception as e:
            self.log(f"获取版本号错误: {str(e)}")

    # ========== 2. 检查传感器连接状态 ==========
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
                self.log(
                    f"发送状态请求: 地址={reg_addr:04X} ({self.sensor_status_addrs[addr]}), 帧={request_frame.hex()}"
                )

                time.sleep(0.01)
                response = self.ser.read(128)
                self.loop_read_len = old_len

                if not response:
                    self.log(f"未收到地址{reg_addr:04X}的响应")
                    continue

                parsed = self.parse_response_frame(response)
                if parsed and parsed["func_code"] == 0x03 and parsed["data_len"] == 1:
                    status_byte = parsed["data"][0]
                    self.log(
                        f"地址{reg_addr:04X}状态字节: 0x{status_byte:02X} ({bin(status_byte)[2:].zfill(8)})"
                    )
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
        self.log("  大拇指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << i)) else "未连接"
            self.log(f"    {thumb_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(thumb_sensors[i])
        self.log("  食指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << (i + 4))) else "未连接"
            self.log(f"    {index_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(index_sensors[i])

    def _parse_addr_0011(self, status_byte):
        middle_sensors = ["中指近节", "中指中节", "中指指尖", "中指指甲"]
        ring_sensors = ["无名指近节", "无名指中节", "无名指指尖", "无名指指甲"]
        self.log("  中指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << i)) else "未连接"
            self.log(f"    {middle_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(middle_sensors[i])
        self.log("  无名指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << (i + 4))) else "未连接"
            self.log(f"    {ring_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(ring_sensors[i])

    def _parse_addr_0012(self, status_byte):
        pinky_sensors = ["小拇指近节", "小拇指中节", "小拇指指尖", "小拇指指甲"]
        palm_sensors = ["掌心1", "掌心2", "掌心3", "掌心4"]
        self.log("  小拇指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << i)) else "未连接"
            self.log(f"    {pinky_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(pinky_sensors[i])
        self.log("  掌心传感器(1-4)状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << (i + 4))) else "未连接"
            self.log(f"    {palm_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(palm_sensors[i])

    def _parse_addr_0013(self, status_byte):
        palm_sensors = ["掌心5", "掌心6", "掌心7", "掌心8"]
        self.log("  掌心传感器(5-8)状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << i)) else "未连接"
            self.log(f"    {palm_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(palm_sensors[i])

    # ========== 3. 检查分布力点数 ==========
    def check_distribution_points(self):
        """检查传感器分布力点数"""
        if not self.ser or not self.ser.is_open:
            self.log("请先连接设备")
            return
        self.distribution_points_cache.clear()
        self.log("===== 开始检查分布力点数 =====")
        groups = {
            "大拇指": ["大拇指近节", "大拇指中节", "大拇指指尖", "大拇指指甲"],
            "食指": ["食指近节", "食指中节", "食指指尖", "食指指甲"],
            "中指": ["中指近节", "中指中节", "中指指尖", "中指指甲"],
            "无名指": ["无名指近节", "无名指中节", "无名指指尖", "无名指指甲"],
            "小拇指": ["小拇指近节", "小拇指中节", "小拇指指尖", "小拇指指甲"],
            "掌心": [
                "掌心1",
                "掌心2",
                "掌心3",
                "掌心4",
                "掌心5",
                "掌心6",
                "掌心7",
                "掌心8",
            ],
        }
        for group_name, modules in groups.items():
            self.log(f"\n===== {group_name}传感器分布力点数 =====")
            for module in modules:
                if module not in self.distribution_points_addrs:
                    self.log(f"  {module}: 无对应地址信息")
                    continue
                addr = self.distribution_points_addrs[module]
                try:
                    reg_addr = addr
                    read_len = 2
                    old_len = self.loop_read_len
                    self.loop_read_len = read_len
                    request_frame = self.build_request_frame(0x03, reg_addr, b"")
                    self.ser.write(request_frame)
                    time.sleep(0.01)
                    response = self.ser.read(128)
                    self.loop_read_len = old_len
                    if not response:
                        self.log(f"  {module} (地址{reg_addr:04X}): 未收到响应")
                        self.distribution_points_cache[module] = 0
                        continue
                    parsed = self.parse_response_frame(response)
                    if (
                        parsed
                        and parsed["func_code"] == 0x03
                        and parsed["data_len"] == 2
                    ):
                        point_count = int.from_bytes(
                            parsed["data"], byteorder="little", signed=False
                        )
                        self.distribution_points_cache[module] = point_count
                        byte_count = point_count * 3
                        if module.startswith("掌心"):
                            actual_points = min(point_count, self.palm_sensor_limit)
                            status = (
                                f"已连接 (将只解析前{actual_points}个点)"
                                if point_count > 0
                                else "未连接"
                            )
                        else:
                            status = "已连接" if point_count > 0 else "未连接"
                        self.log(
                            f"  {module} (地址{reg_addr:04X}): 点数={point_count}, 字节数={byte_count}, 状态={status}"
                        )
                    else:
                        self.log(f"  {module} (地址{reg_addr:04X}): 读取失败")
                        self.distribution_points_cache[module] = 0
                except Exception as e:
                    self.log(f"  {module} (地址{reg_addr:04X}) 错误: {str(e)}")
                    self.distribution_points_cache[module] = 0
        self.log("\n===== 分布力点数检查完成 =====")

    # ========== 4. 发送标定指令 ==========
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

    # ========== 5. 读取传感器模组合力 ==========
    def parse_module_forces(self, data: bytes) -> List[Dict]:
        """解析传感器模组合力数据"""
        parsed = []
        if len(data) != self.module_total_bytes:
            self.log(
                f"警告: 传感器模组合力数据长度不符 - 预期: {self.module_total_bytes}字节, 实际: {len(data)}字节"
            )
        valid_modules = min(len(data) // self.module_force_bytes, self.module_count)
        self.log(
            f"传感器模组合力数据: 共{valid_modules}/{self.module_count}个有效模组数据（仅解析低字节）"
        )
        for i in range(valid_modules):
            offset = i * self.module_force_bytes
            if offset + self.module_force_bytes > len(data):
                self.log(f"警告: 数据长度不足，终止解析")
                break
            fx_low_byte = data[offset]
            fy_low_byte = data[offset + 2]
            fz_low_byte = data[offset + 4]
            fx_bytes = data[offset : offset + 2]
            fy_bytes = data[offset + 2 : offset + 4]
            fz_bytes = data[offset + 4 : offset + 6]
            fx_raw = fx_low_byte if fx_low_byte <= 127 else fx_low_byte - 256
            fy_raw = fy_low_byte if fy_low_byte <= 127 else fy_low_byte - 256
            fz_raw = fz_low_byte
            fx_scaled = round(fx_raw * 0.1, 1)
            fy_scaled = round(fy_raw * 0.1, 1)
            fz_scaled = round(fz_raw * 0.1, 1)
            parsed.append(
                {
                    "index": i,
                    "name": (
                        self.module_names[i]
                        if i < len(self.module_names)
                        else f"未知模组{i}"
                    ),
                    "raw_hex": (
                        fx_bytes.hex().upper(),
                        fy_bytes.hex().upper(),
                        fz_bytes.hex().upper(),
                    ),
                    "used_byte": (
                        f"0x{fx_low_byte:02X}",
                        f"0x{fy_low_byte:02X}",
                        f"0x{fz_low_byte:02X}",
                    ),
                    "converted": (fx_raw, fy_raw, fz_raw),
                    "scaled": (fx_scaled, fy_scaled, fz_scaled),
                }
            )
        if parsed:
            self.log("\n===== 大拇指传感器 =====")
            for module in parsed[0:4]:
                self.log_module_force(module)
            self.log("\n===== 食指传感器 =====")
            for module in parsed[4:8]:
                self.log_module_force(module)
            self.log("\n===== 中指传感器 =====")
            for module in parsed[8:12]:
                self.log_module_force(module)
            self.log("\n===== 无名指传感器 =====")
            for module in parsed[12:16]:
                self.log_module_force(module)
            self.log("\n===== 小拇指传感器 =====")
            for module in parsed[16:20]:
                self.log_module_force(module)
            self.log("\n===== 掌心传感器 =====")
            for module in parsed[20:28]:
                self.log_module_force(module)
        else:
            self.log("无有效传感器模组合力数据")
        return parsed

    def log_module_force(self, module: Dict):
        """记录模组受力日志"""
        self.log(
            f"{module['name']:8s} | Fx={module['scaled'][0]:5.1f}N Fy={module['scaled'][1]:5.1f}N Fz={module['scaled'][2]:5.1f}N "
        )

    def read_module_forces_single(self):
        """【单次读取】传感器模组合力 - 纯单次调用，无循环逻辑"""
        if not self.ser or not self.ser.is_open:
            self.log("请先连接设备")
            return
        try:
            reg_addr = 0x0500
            read_len = self.module_total_bytes
            old_len = self.loop_read_len
            self.loop_read_len = read_len
            request_frame = self.build_request_frame(0x03, reg_addr, b"")
            self.ser.write(request_frame)
            self.log(
                f"发送传感器模组合力请求: 地址={reg_addr:04X}, 长度={read_len}, 帧={request_frame.hex()}"
            )
            time.sleep(0.01)
            response = self.ser.read(512)
            self.loop_read_len = old_len
            if not response:
                self.log(f"未收到传感器模组合力响应")
                return
            parsed = self.parse_response_frame(response)
            if parsed and parsed["func_code"] == 0x03:
                self.log(
                    f"传感器模组合力读取成功: 地址={parsed['reg_addr']:04X}, 长度={parsed['data_len']}"
                )
                self.parse_module_forces(parsed["data"])
            else:
                self.log(f"传感器模组合力读取失败")
        except Exception as e:
            self.log(f"传感器模组合力读取错误: {str(e)}")

    def module_forces_loop_read(self):
        log_prefix = "[循环读模组]"
        while self.module_cycle_read_running and self.ser and self.ser.is_open:
            try:
                reg_addr = 0x0500
                read_len = self.module_total_bytes
                old_len = self.loop_read_len
                self.loop_read_len = read_len
                request_frame = self.build_request_frame(0x03, reg_addr, b"")
                self.ser.write(request_frame)
                self.ser.flushOutput()  # 立即发送，无缓存
                response = self.ser.read(512)
                self.ser.flushInput()  # 读完清空缓冲区，无残留数据堆积
                self.loop_read_len = old_len

                if not response:
                    continue
                parsed = self.parse_response_frame(response)
                if parsed and parsed["func_code"] == 0x03:
                    self.parse_module_forces(parsed["data"])
                time.sleep(0.01)  # 循环间隔
            except Exception as e:
                pass  # 高频采集场景，异常静默处理，不打印日志拖慢速度

    def start_module_cycle_read(self):
        """开始循环读取传感器模组合力"""
        if self.module_cycle_read_running:
            self.log("已在循环读取传感器模组数据中")
            return
        if not self.ser or not self.ser.is_open:
            self.log("请先连接设备")
            return
        self.module_cycle_read_running = True
        self.log("开始循环读取传感器模组数据...")
        # 启动守护线程运行循环读取，不阻塞主线程
        threading.Thread(target=self.module_forces_loop_read, daemon=True).start()

    def stop_module_cycle_read(self):
        """停止循环读取传感器模组合力 - 仅修改状态标志位"""
        self.module_cycle_read_running = False
        self.log("已停止循环读取传感器模组数据")

    # ========== 6. 读取所有连接成功的传感器分布力数据 ==========
    def get_address_by_sensor_name(
        self, sensor_name: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """根据传感器名称获取对应的地址范围"""
        for (start, end), name in self.distribution_addr_ranges.items():
            if name == sensor_name:
                return (start, end)
        return (None, None)

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
        # 打印【前1组】力值数据
        if parsed:
            for group in parsed:
                self.log(
                    f"  组{group['index']:02d} | 物理值: X={group['scaled'][0]:5.1f}N, Y={group['scaled'][1]:5.1f}N, Z={group['scaled'][2]:5.1f}N"
                )
        return parsed

    def read_connected_sensors(self):
        """读取所有连接成功的传感器分布力数据"""
        if not self.ser or not self.ser.is_open:
            self.log("请先连接设备")
            return
        self.log("===== 正在更新传感器连接状态 =====")
        self.check_sensor_status()
        if not self.connected_sensors:
            self.log("没有检测到连接成功的传感器，无法读取分布力数据")
            return
        self.log("\n===== 开始读取所有连接成功的传感器分布力 =====")
        self.log(f"共检测到 {len(self.connected_sensors)} 个连接成功的传感器")
        groups = {
            "大拇指": [],
            "食指": [],
            "中指": [],
            "无名指": [],
            "小拇指": [],
            "掌心": [],
        }
        for sensor in self.connected_sensors:
            if sensor.startswith("大拇指"):
                groups["大拇指"].append(sensor)
            elif sensor.startswith("食指"):
                groups["食指"].append(sensor)
            elif sensor.startswith("中指"):
                groups["中指"].append(sensor)
            elif sensor.startswith("无名指"):
                groups["无名指"].append(sensor)
            elif sensor.startswith("小拇指"):
                groups["小拇指"].append(sensor)
            elif sensor.startswith("掌心"):
                groups["掌心"].append(sensor)

        all_results = {}
        success_count = 0
        fail_count = 0
        for group_name, sensors in groups.items():
            if sensors:
                self.log(f"\n----- {group_name}传感器 -----")
                for idx, sensor in enumerate(sensors):
                    self.log(f"\n[{idx+1}/{len(sensors)}] 处理 {sensor}...")
                    start_addr, end_addr = self.get_address_by_sensor_name(sensor)
                    if not start_addr or not end_addr:
                        self.log(f"  {sensor}: 未找到对应的地址范围")
                        fail_count += 1
                        continue
                    points_addr = self.distribution_points_addrs.get(sensor)
                    self.log(f"\n----- 1111-----")

                    if not points_addr:
                        self.log(f"  {sensor}: 未找到点数地址信息")
                        fail_count += 1
                        continue
                    try:
                        self.log(f"\n----- 222222222-----")

                        old_len = self.loop_read_len
                        self.loop_read_len = 2
                        request_frame = self.build_request_frame(0x03, points_addr, b"")
                        self.log(f"\n----- 23333333333333-----")
                        self.ser.write(request_frame)
                        self.ser.flushOutput()
                        response = self.ser.read(128)
                        self.log(f"\n----- 2444444444444444-----")
                        self.loop_read_len = old_len
                        if not response:
                            self.log(f"  {sensor}: 未收到点数响应")
                            fail_count += 1
                            continue
                        self.log(f"\n----- 25555555555555555-----")
                        parsed = self.parse_response_frame(response)
                        self.log(f"\n----- 3333333333333333333----")
                        if (
                            parsed
                            and parsed["func_code"] == 0x03
                            and parsed["data_len"] == 2
                        ):
                            point_count = int.from_bytes(
                                parsed["data"], byteorder="little", signed=False
                            )
                            if sensor.startswith("掌心"):
                                actual_points = min(point_count, self.palm_sensor_limit)
                                self.log(
                                    f"  {sensor}: 原始点数={point_count}, 将只解析前{actual_points}个点"
                                )
                                data_len = actual_points * 3
                            else:
                                data_len = point_count * 3
                            if data_len <= 0:
                                self.log(f"  {sensor}: 无效的点数({point_count})")
                                fail_count += 1
                                continue
                            max_possible_len = end_addr - start_addr + 1
                            if data_len > max_possible_len:
                                self.log(
                                    f"  {sensor}: 数据长度({data_len})超过地址范围最大长度({max_possible_len})，将截断读取"
                                )
                                data_len = max_possible_len
                            self.log(f"\n----- 33333333333333333-----")
                            old_len = self.loop_read_len
                            self.loop_read_len = data_len
                            request_frame = self.build_request_frame(
                                0x03, start_addr, b""
                            )
                            self.ser.write(request_frame)
                            self.ser.flushOutput()
                            data_response = self.ser.read(1024)
                            self.log(f"\n----- 444444444444444-----")
                            self.loop_read_len = old_len
                            if not data_response:
                                self.log(f"  {sensor}: 未收到分布力数据")
                                fail_count += 1
                                continue
                            data_parsed = self.parse_response_frame(data_response)
                            if data_parsed and data_parsed["func_code"] == 0x03:
                                self.log(
                                    f"  成功读取 {sensor} 分布力数据 (地址:0x{start_addr:04X}, 长度:{data_len}字节)"
                                )
                                parsed_data = self.parse_normal_force_data(
                                    data_parsed["data"],
                                    start_addr,
                                    source=f"[{sensor}] ",
                                )
                                all_results[sensor] = parsed_data
                                success_count += 1
                            else:
                                self.log(f"  {sensor}: 分布力数据解析失败")
                                fail_count += 1
                        else:
                            self.log(f"  {sensor}: 点数读取失败")
                            fail_count += 1
                    except Exception as e:
                        self.log(f"  {sensor} 读取错误: {str(e)}")
                        fail_count += 1
        self.log("\n===== 所有连接成功的传感器分布力读取完成 =====")
        self.log(f"读取结果: 成功 {success_count} 个, 失败 {fail_count} 个")
        return all_results

    # ========== 7. 循环读取指定地址的分布力  ==========
    def set_cycle_read_params(self, addr: int, length: int, interval: float = 0.1):
        """设置循环读取参数"""
        self.loop_read_addr = addr
        self.loop_read_len = length
        self.loop_read_interval = interval
        self.log(
            f"循环读取参数已设置: 地址=0x{addr:04X}, 长度={length}字节, 间隔={interval}秒"
        )

    def read_registers(self):
        while self.cycle_read_running and self.ser and self.ser.is_open:
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
                results = self.read_connected_sensors()
                # 打印汇总信息
                print(
                    f"\n[{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}] 本次读取汇总:"
                )
                for sensor_name, force_data in results.items():
                    if force_data:
                        # 计算该传感器的最大值
                        max_x = max(point["scaled"][0] for point in force_data)
                        max_y = max(point["scaled"][1] for point in force_data)
                        max_z = max(point["scaled"][2] for point in force_data)
                        print(
                            f"  {sensor_name}: 最大力值 X={max_x:5.1f}N, Y={max_y:5.1f}N, Z={max_z:5.1f}N"
                        )

                print("-" * 60)

            except Exception as e:
                pass  # 高频场景下，异常捕获不打印日志，避免拖慢速度

    def start_cycle_read(self):
        self.cycle_read_running = True
        # 新开线程执行循环读取，不阻塞主线程的按键停止
        threading.Thread(target=self.read_registers, daemon=True).start()

    def stop_cycle_read(self):
        """停止循环读取"""
        self.cycle_read_running = False
        self.log("已停止循环读取")


# ========== 主程序  ==========
if __name__ == "__main__":
    # 创建实例
    comm = HighSpeedCommBoard()

    # ===== 端口可选模式 =====
    ports = comm.list_com_ports()
    if not ports:
        print("未检测到可用COM端口！")
        exit(1)
    print("\n可用COM端口列表:")
    for i, port in enumerate(ports, 1):
        print(f"{i}. {port}")
    while True:
        try:
            choice = int(input(f"\n请选择COM端口(1-{len(ports)}): "))
            if 1 <= choice <= len(ports):
                com_port = ports[choice - 1]
                break
            else:
                print(f"请输入1到{len(ports)}之间的数字！")
        except ValueError:
            print("输入错误，请输入数字！")
    comm.connect(com_port, 921600)

    # 1. 获取版本号
    comm.get_version()
    time.sleep(5)

    # 2. 检查传感器连接状态
    comm.check_sensor_status()

    # 3. 检查分布力点数
    comm.check_distribution_points()

    # 4. 发送标定指令
    comm.start_calibration()
    time.sleep(2)

    # 5. 单次读取模组合力
    comm.read_module_forces_single()

    # 6. 读取所有连接成功的传感器分布力
    comm.read_connected_sensors()

    # ===== 7.循环读取按键控制 =====
    while True:
        print("\n===== 循环读取控制菜单 =====")
        print("1. 开始分布力循环读取")
        print("2. 开始模组合力循环读取")
        print("3. 程序退出")
        print("4. 手动输入指定地址和长度读取")
        print("============================")
        choice = input("请输入选择(1/2/3/4): ").strip()
        if choice == "1":
            comm.set_cycle_read_params(0x1200, 75, 0.001)  # 大拇指中节 0x1200 75字节
            comm.start_cycle_read()
            input("\n分布力循环读取中，按Enter键停止...")
            comm.stop_cycle_read()
        elif choice == "2":
            comm.start_module_cycle_read()
            input("\n模组合力循环读取中，按Enter键停止...")
            comm.stop_module_cycle_read()
        elif choice == "3":
            comm.stop_cycle_read()
            comm.stop_module_cycle_read()
            comm.disconnect()
            print("程序已退出！")
            exit(0)
        elif choice == "4":
            try:
                addr_str = input("请输入读取地址(16进制，如1000): ")
                reg_addr = int(addr_str, 16)
                read_len = int(input("请输入读取长度(字节): "))
                # 校验长度合法性
                if read_len <= 0:
                    print(" 读取长度必须为正整数！")
                    continue
                # 1. 设置自定义的读取参数
                comm.set_cycle_read_params(reg_addr, read_len, 0.001)
                # 2. 启动循环读取（和choice=1逻辑完全一致）
                comm.start_cycle_read()
                # 3. 交互逻辑：读取中等待，按回车停止
                input(
                    f"\n 已开始循环读取 | 地址=0x{reg_addr:04X} | 长度={read_len}字节，按Enter键停止..."
                )
                # 4. 停止循环
                comm.stop_cycle_read()

            except ValueError:
                print("输入格式错误！地址需为16进制数字，长度需为十进制整数")
        else:
            print("无效选择，请输入1、2、3或4！")
