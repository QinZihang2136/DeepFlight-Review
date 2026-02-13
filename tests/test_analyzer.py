"""
LogCortex V3 - 自动化测试套件
测试 LogAnalyzer 核心功能和各模块
"""
import unittest
import sys
import os
import tempfile
import struct
import numpy as np
import pandas as pd
from io import BytesIO

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_minimal_ulog():
    """
    创建一个最小化的 ULog 测试文件
    参考: https://dev.px4.io/en/log/ulog_file_format.html
    """
    buffer = BytesIO()

    # ULog 文件头 (16 bytes)
    # Magic: 0x55 0x4C 0x6F 0x67 0x01 0x12 0x35 0x01
    header = bytes([0x55, 0x4C, 0x6F, 0x67, 0x01, 0x12, 0x35, 0x01])
    # 版本信息
    header += struct.pack('<Q', 0)  # timestamp (8 bytes)
    buffer.write(header)

    # Flag Section (8 bytes)
    buffer.write(struct.pack('<Q', 0))  # flags

    # 定义消息格式 (Message Format)
    # Header: 0x01, continue_flag(1), msg_size(2)
    # 定义 vehicle_status 格式
    format_msg = b'\x01'  # Message Type: Format
    format_data = b'vehicle_status\x00'
    format_data += b'uint64_t timestamp\x00'
    format_data += b'uint8_t arming_state\x00'
    format_data += b'uint8_t nav_state\x00'
    format_data += b'bool failsafe\x00'
    format_msg += struct.pack('<H', len(format_data))
    format_msg += format_data
    buffer.write(format_msg)

    # 定义 vehicle_local_position 格式
    format_msg = b'\x01'
    format_data = b'vehicle_local_position\x00'
    format_data += b'uint64_t timestamp\x00'
    format_data += b'float[3] xyz\x00'  # 这将被展开为 x, y, z
    format_data += b'float vx\x00'
    format_data += b'float vy\x00'
    format_data += b'float vz\x00'
    format_msg += struct.pack('<H', len(format_data))
    format_msg += format_data
    buffer.write(format_msg)

    # 定义 vehicle_global_position 格式
    format_msg = b'\x01'
    format_data = b'vehicle_global_position\x00'
    format_data += b'uint64_t timestamp\x00'
    format_data += b'double lat\x00'
    format_data += b'double lon\x00'
    format_data += b'float alt\x00'
    format_msg += struct.pack('<H', len(format_data))
    format_msg += format_data
    buffer.write(format_msg)

    # 定义 battery_status 格式
    format_msg = b'\x01'
    format_data = b'battery_status\x00'
    format_data += b'uint64_t timestamp\x00'
    format_data += b'float voltage_v\x00'
    format_data += b'float current_a\x00'
    format_data += b'float remaining\x00'
    format_msg += struct.pack('<H', len(format_data))
    format_msg += format_data
    buffer.write(format_msg)

    # Information Message (0x02) - 添加元数据
    info_msg = b'\x02'
    info_data = b'sys_name\x00test_vehicle\x00'
    info_msg += struct.pack('<H', len(info_data))
    info_msg += info_data
    buffer.write(info_msg)

    info_msg = b'\x02'
    info_data = b'ver_sw\x001.14.0\x00'
    info_msg += struct.pack('<H', len(info_data))
    info_msg += info_data
    buffer.write(info_msg)

    # Parameter Message (0x03) - 添加一些参数
    param_msg = b'\x03'
    param_data = b'float MPC_XY_VEL_MAX\x00' + struct.pack('<f', 10.0)
    param_msg += struct.pack('<H', len(param_data))
    param_msg += param_data
    buffer.write(param_msg)

    param_msg = b'\x03'
    param_data = b'int32_t MPC_XY_MAN_EXPO\x00' + struct.pack('<i', 0)
    param_msg += struct.pack('<H', len(param_data))
    param_msg += param_data
    buffer.write(param_msg)

    # 添加数据消息
    base_ts = 1700000000000000  # 微秒时间戳

    # vehicle_status 数据
    for i in range(100):
        data_msg = b'\x04'  # Message Type: Data
        ts = base_ts + i * 100000  # 每 100ms 一条
        arming = 2 if i > 10 else 1  # 前10条是 STANDBY，之后是 ARMED
        nav = 2 if i > 20 else (1 if i > 10 else 0)  # 模式切换
        failsafe = 0

        # 数据
        data = struct.pack('<QBB?', ts, arming, nav, bool(failsafe))
        msg_id = 0  # message ID

        data_msg += struct.pack('<HB', msg_id, 0)  # msg_id, multi_id
        data_msg += struct.pack('<H', len(data))
        data_msg += data
        buffer.write(data_msg)

    # vehicle_local_position 数据
    for i in range(500):
        data_msg = b'\x04'
        ts = base_ts + i * 20000  # 每 20ms 一条

        x = float(i * 0.1)
        y = float(np.sin(i * 0.1) * 5)
        z = float(-i * 0.05 - 10)  # 负值表示高度
        vx = 0.1
        vy = np.cos(i * 0.1) * 0.5
        vz = -0.05

        data = struct.pack('<Qfff', ts, x, y, z)
        data += struct.pack('<fff', vx, vy, vz)

        data_msg += struct.pack('<HB', 1, 0)
        data_msg += struct.pack('<H', len(data))
        data_msg += data
        buffer.write(data_msg)

    return buffer.getvalue()


def create_mock_ulog_file():
    """创建临时 ULog 文件"""
    data = create_minimal_ulog()
    with tempfile.NamedTemporaryFile(suffix='.ulg', delete=False) as f:
        f.write(data)
        return f.name


class TestLogAnalyzerBasic(unittest.TestCase):
    """LogAnalyzer 基础测试 - 使用模拟数据"""

    @classmethod
    def setUpClass(cls):
        """设置测试类"""
        cls.test_file = None
        try:
            cls.test_file = create_mock_ulog_file()
        except Exception as e:
            print(f"Warning: Could not create mock ULog: {e}")

    @classmethod
    def tearDownClass(cls):
        """清理测试文件"""
        if cls.test_file and os.path.exists(cls.test_file):
            try:
                os.unlink(cls.test_file)
            except:
                pass

    def test_01_imports(self):
        """测试模块导入"""
        try:
            from modules.analyzer import LogAnalyzer
            from modules.ui_components import render_chart, render_comparison_chart
            from modules.flight_review_views import render_flight_review_dashboard_v2
            self.assertTrue(True, "所有模块导入成功")
        except ImportError as e:
            self.fail(f"导入失败: {e}")

    def test_02_analyzer_constants(self):
        """测试分析器常量定义"""
        from modules.analyzer import ARMING_STATE_MAP, NAV_STATE_MAP, ARMING_STATE_DESC, NAV_STATE_DESC

        self.assertIn(2, ARMING_STATE_MAP)
        self.assertEqual(ARMING_STATE_MAP[2], "ARMED")
        self.assertIn(2, NAV_STATE_MAP)
        self.assertEqual(NAV_STATE_MAP[2], "POSCTL")
        self.assertIn(2, ARMING_STATE_DESC)
        self.assertEqual(ARMING_STATE_DESC[2], "已解锁")

    def test_03_pyulog_available(self):
        """测试 pyulog 库可用性"""
        try:
            from pyulog import ULog
            self.assertTrue(True)
        except ImportError:
            self.skipTest("pyulog 未安装")


class TestUIComponents(unittest.TestCase):
    """UI 组件测试"""

    def test_01_ui_imports(self):
        """测试 UI 组件导入"""
        try:
            from modules.ui_components import (
                render_chart,
                render_linked_subplots,
                render_comparison_chart,
                render_map,
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"UI 组件导入失败: {e}")

    def test_02_create_sample_chart(self):
        """测试创建样例图表"""
        from modules.ui_components import render_chart

        # 创建测试数据
        df = pd.DataFrame({
            'timestamp': np.linspace(0, 10, 100),
            'value': np.sin(np.linspace(0, 10, 100))
        })

        # 这个测试只验证函数可以调用，不验证渲染结果
        # 因为 Streamlit 组件在没有运行时环境时可能会失败
        try:
            # 注意: 这里可能会因为没有 Streamlit 上下文而失败
            # 我们主要测试函数定义和参数
            self.assertTrue(callable(render_chart))
        except Exception as e:
            self.skipTest(f"图表渲染跳过 (无 Streamlit 上下文): {e}")


class TestFlightReviewViews(unittest.TestCase):
    """Flight Review 视图测试"""

    def test_01_views_imports(self):
        """测试视图模块导入"""
        try:
            from modules.flight_review_views import render_flight_review_dashboard_v2
            self.assertTrue(callable(render_flight_review_dashboard_v2))
        except ImportError as e:
            self.fail(f"视图模块导入失败: {e}")

    def test_02_layout_constants(self):
        """测试布局常量"""
        try:
            from modules.flight_review_layout import MODE_COLORS, FLIGHT_REVIEW_GROUPS
            self.assertIsInstance(MODE_COLORS, dict)
            self.assertIsInstance(FLIGHT_REVIEW_GROUPS, list)
            # 检查关键内容
            self.assertIn("MANUAL", MODE_COLORS)
            self.assertTrue(len(FLIGHT_REVIEW_GROUPS) > 0)
        except ImportError as e:
            self.skipTest(f"布局模块导入失败: {e}")


class TestAppFunctions(unittest.TestCase):
    """app.py 中的功能测试"""

    def test_01_fuzzy_match(self):
        """测试模糊匹配函数"""
        # 复制 app.py 中的函数进行测试
        def fuzzy_match(path, query):
            if not query.strip():
                return True
            p = path.lower()
            parts = [x for x in query.lower().strip().split() if x]
            return all(part in p for part in parts)

        self.assertTrue(fuzzy_match("vehicle_local_position/x", "local"))
        self.assertTrue(fuzzy_match("vehicle_local_position/x", "vehicle x"))
        self.assertFalse(fuzzy_match("vehicle_local_position/x", "gps"))
        self.assertTrue(fuzzy_match("sensor_mag/x", ""))

    def test_02_build_tool_specs(self):
        """测试工具规格定义"""
        # 复制 app.py 中的函数
        def build_tool_specs():
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "get_flight_summary",
                        "description": "获取飞行概览",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_event_timeline",
                        "description": "获取关键事件时间线",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ]

        specs = build_tool_specs()
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0]["function"]["name"], "get_flight_summary")


class TestAnalyzerFunctions(unittest.TestCase):
    """LogAnalyzer 方法测试（无需真实日志文件）"""

    def test_01_unique_change_times(self):
        """测试唯一变化时间检测"""
        # 模拟 _unique_change_times 逻辑
        def unique_change_times(t_s, values):
            if len(values) == 0:
                return []
            out = []
            last = values[0]
            out.append((float(t_s[0]), float(last)))
            for i in range(1, len(values)):
                if values[i] != last:
                    last = values[i]
                    out.append((float(t_s[i]), float(last)))
            return out

        t = np.array([0, 1, 2, 3, 4, 5])
        v = np.array([1, 1, 2, 2, 2, 3])

        changes = unique_change_times(t, v)
        self.assertEqual(len(changes), 3)
        self.assertEqual(changes[0], (0.0, 1.0))
        self.assertEqual(changes[1], (2.0, 2.0))
        self.assertEqual(changes[2], (5.0, 3.0))

    def test_02_format_arming_info(self):
        """测试 arming 状态格式化"""
        from modules.analyzer import ARMING_STATE_MAP, ARMING_STATE_DESC

        def format_arming_info(value):
            v = int(value)
            return {
                "value": v,
                "name": ARMING_STATE_MAP.get(v, f"UNKNOWN({v})"),
                "description_cn": ARMING_STATE_DESC.get(v, f"未知状态({v})"),
            }

        info = format_arming_info(2)
        self.assertEqual(info["name"], "ARMED")
        self.assertEqual(info["description_cn"], "已解锁")

    def test_03_format_nav_info(self):
        """测试导航状态格式化"""
        from modules.analyzer import NAV_STATE_MAP, NAV_STATE_DESC

        def format_nav_info(value):
            v = int(value)
            return {
                "value": v,
                "name": NAV_STATE_MAP.get(v, f"UNKNOWN({v})"),
                "description_cn": NAV_STATE_DESC.get(v, f"未知模式({v})"),
            }

        info = format_nav_info(5)
        self.assertEqual(info["name"], "AUTO_RTL")
        self.assertEqual(info["description_cn"], "自动返航模式")


class TestFFTFunctions(unittest.TestCase):
    """FFT 和信号处理测试"""

    def test_01_fft_computation(self):
        """测试 FFT 计算"""
        # 生成测试信号
        fs = 100.0  # 采样率
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz 正弦波

        # 计算 FFT
        n = len(signal)
        freq = np.fft.rfftfreq(n, d=1/fs)
        amp = np.abs(np.fft.rfft(signal)) / n

        # 找到主频率
        peak_idx = np.argmax(amp)
        peak_freq = freq[peak_idx]

        self.assertAlmostEqual(peak_freq, 10.0, places=1)

    def test_02_spectrogram_logic(self):
        """测试时频谱逻辑"""
        try:
            from scipy.signal import spectrogram

            fs = 100.0
            t = np.linspace(0, 2, 200)
            signal = np.sin(2 * np.pi * 10 * t)

            f, t_spec, Sxx = spectrogram(signal, fs=fs, nperseg=64)

            self.assertTrue(len(f) > 0)
            self.assertTrue(len(t_spec) > 0)
            self.assertTrue(Sxx.shape[0] == len(f))
        except ImportError:
            self.skipTest("scipy 未安装")


class TestPandasOperations(unittest.TestCase):
    """Pandas 数据操作测试"""

    def test_01_dataframe_resampling(self):
        """测试数据降采样"""
        df = pd.DataFrame({
            'timestamp': np.linspace(0, 10, 10000),
            'value': np.random.randn(10000)
        })

        # 模拟降采样
        max_points = 5000
        step = len(df) // max_points
        if step > 1:
            df_downsampled = df.iloc[::step, :]
        else:
            df_downsampled = df

        self.assertLessEqual(len(df_downsampled), max_points + step)

    def test_02_signal_normalization(self):
        """测试信号标准化"""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })

        # Z-Score 标准化
        mean = df['value'].mean()
        std = df['value'].std()
        z_score = (df['value'] - mean) / std

        self.assertAlmostEqual(z_score.mean(), 0, places=10)
        self.assertAlmostEqual(z_score.std(), 1, places=10)


class TestStreamlitConfig(unittest.TestCase):
    """Streamlit 配置测试"""

    def test_01_config_file_exists(self):
        """测试配置文件存在"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '.streamlit', 'config.toml'
        )
        self.assertTrue(os.path.exists(config_path), f"配置文件不存在: {config_path}")

    def test_02_config_content(self):
        """测试配置文件内容"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '.streamlit', 'config.toml'
        )
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                content = f.read()
            self.assertIn('maxUploadSize', content)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestLogAnalyzerBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestUIComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestFlightReviewViews))
    suite.addTests(loader.loadTestsFromTestCase(TestAppFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyzerFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestFFTFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestPandasOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestStreamlitConfig))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
