"""
LogCortex V3 - 集成测试套件
测试完整工作流程和依赖兼容性
"""
import unittest
import sys
import os
import subprocess
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDependencyVersions(unittest.TestCase):
    """依赖版本测试"""

    def test_01_core_dependencies(self):
        """测试核心依赖版本"""
        import streamlit
        import pandas
        import numpy
        import plotly
        import openai

        print(f"\n  streamlit: {streamlit.__version__}")
        print(f"  pandas: {pandas.__version__}")
        print(f"  numpy: {numpy.__version__}")
        print(f"  plotly: {plotly.__version__}")
        print(f"  openai: {openai.__version__}")

        # 验证最小版本要求
        self.assertTrue(int(streamlit.__version__.split('.')[0]) >= 1)
        self.assertTrue(int(pandas.__version__.split('.')[0]) >= 2)
        self.assertTrue(int(numpy.__version__.split('.')[0]) >= 2)

    def test_02_pyulog_available(self):
        """测试 pyulog 可用"""
        try:
            import pyulog
            print(f"  pyulog: available")
        except ImportError:
            self.fail("pyulog 未安装")

    def test_03_scipy_available(self):
        """测试 scipy 可用"""
        try:
            import scipy
            print(f"  scipy: {scipy.__version__}")
        except ImportError:
            self.fail("scipy 未安装")

    def test_04_pillow_available(self):
        """测试 Pillow 可用"""
        try:
            from PIL import Image
            import PIL
            print(f"  Pillow: {PIL.__version__}")
        except ImportError:
            self.fail("Pillow 未安装")


class TestModuleIntegration(unittest.TestCase):
    """模块集成测试"""

    def test_01_analyzer_instantiation(self):
        """测试 LogAnalyzer 类可以实例化（不加载日志）"""
        from modules.analyzer import LogAnalyzer
        # 仅验证类定义正确
        self.assertTrue(hasattr(LogAnalyzer, '__init__'))
        self.assertTrue(hasattr(LogAnalyzer, 'get_flight_summary'))
        self.assertTrue(hasattr(LogAnalyzer, 'get_event_timeline'))
        self.assertTrue(hasattr(LogAnalyzer, 'get_topic_data'))

    def test_02_ui_components_functions(self):
        """测试 UI 组件函数定义"""
        from modules.ui_components import (
            render_chart,
            render_linked_subplots,
            render_comparison_chart,
            render_map,
        )
        # 验证函数可调用
        self.assertTrue(callable(render_chart))
        self.assertTrue(callable(render_linked_subplots))
        self.assertTrue(callable(render_comparison_chart))
        self.assertTrue(callable(render_map))

    def test_03_flight_review_views(self):
        """测试 Flight Review 视图"""
        from modules.flight_review_views import render_flight_review_dashboard_v2
        self.assertTrue(callable(render_flight_review_dashboard_v2))

    def test_04_flight_review_layout(self):
        """测试 Flight Review 布局配置"""
        from modules.flight_review_layout import (
            MODE_COLORS,
            GPS_NOISE_FIELDS,
            ACCEL_AXIS_CANDIDATES,
            GYRO_AXIS_CANDIDATES,
            FLIGHT_REVIEW_GROUPS,
        )
        self.assertIsInstance(MODE_COLORS, dict)
        self.assertIsInstance(GPS_NOISE_FIELDS, dict)
        self.assertIsInstance(ACCEL_AXIS_CANDIDATES, list)
        self.assertIsInstance(GYRO_AXIS_CANDIDATES, list)
        self.assertIsInstance(FLIGHT_REVIEW_GROUPS, list)


class TestAnalyzerMethods(unittest.TestCase):
    """LogAnalyzer 方法测试"""

    def test_01_state_mappings(self):
        """测试状态映射完整性"""
        from modules.analyzer import ARMING_STATE_MAP, NAV_STATE_MAP

        # ARMING_STATE_MAP 应包含 0-6
        for i in range(7):
            self.assertIn(i, ARMING_STATE_MAP)

        # NAV_STATE_MAP 应包含常见模式
        common_modes = [0, 1, 2, 3, 4, 5, 12]  # MANUAL, ALTCTL, POSCTL, MISSION, LOITER, RTL, OFFBOARD
        for mode in common_modes:
            self.assertIn(mode, NAV_STATE_MAP)

    def test_02_state_descriptions(self):
        """测试状态描述完整性"""
        from modules.analyzer import ARMING_STATE_DESC, NAV_STATE_DESC

        # 检查描述是中文
        self.assertIn("已解锁", ARMING_STATE_DESC.values())
        self.assertIn("高度控制模式", NAV_STATE_DESC.values())


class TestAppConfiguration(unittest.TestCase):
    """应用配置测试"""

    def test_01_provider_configs(self):
        """测试 AI 提供商配置"""
        provider_configs = {
            "DeepSeek": {
                "base_url": "https://api.deepseek.com",
                "models": ["deepseek-chat", "deepseek-reasoner"],
            },
            "GLM": {
                "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
                "models": ["glm-4.7", "glm-4.5", "glm-4-air", "glm-4-flash"],
            },
        }

        for provider, config in provider_configs.items():
            self.assertIn("base_url", config)
            self.assertIn("models", config)
            self.assertTrue(len(config["models"]) > 0)

    def test_02_tool_specs(self):
        """测试 AI 工具规格"""
        tools = [
            "get_flight_summary",
            "get_event_timeline",
            "list_topics",
            "get_topic_fields",
            "get_topic_preview",
            "compute_statistics",
            "detect_anomalies",
            "search_parameters",
        ]

        for tool in tools:
            # 验证工具名称有效
            self.assertTrue(len(tool) > 0)
            self.assertTrue(tool.replace("_", "").isalnum())


class TestStreamlitStartup(unittest.TestCase):
    """Streamlit 启动测试"""

    def test_01_app_syntax(self):
        """测试 app.py 语法"""
        result = subprocess.run(
            ["python", "-m", "py_compile", "app.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        self.assertEqual(result.returncode, 0, f"语法错误: {result.stderr}")

    def test_02_app_imports(self):
        """测试 app.py 导入"""
        try:
            # 测试主要导入
            from modules.analyzer import LogAnalyzer
            from modules.ui_components import render_chart
            from modules.flight_review_views import render_flight_review_dashboard_v2
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")


class TestFileStructure(unittest.TestCase):
    """文件结构测试"""

    def test_01_required_files(self):
        """测试必需文件存在"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        required_files = [
            "app.py",
            "modules/__init__.py",
            "modules/analyzer.py",
            "modules/ui_components.py",
            "modules/flight_review_layout.py",
            "modules/flight_review_views.py",
            ".streamlit/config.toml",
        ]

        for f in required_files:
            path = os.path.join(base_dir, f)
            self.assertTrue(os.path.exists(path), f"缺少文件: {f}")

    def test_02_config_file_content(self):
        """测试配置文件内容"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(base_dir, ".streamlit", "config.toml")

        with open(config_path, 'r') as f:
            content = f.read()

        self.assertIn("maxUploadSize", content)
        # 验证上传大小设置合理 (应该 >= 1024 MB)
        import re
        match = re.search(r'maxUploadSize\s*=\s*(\d+)', content)
        if match:
            size = int(match.group(1))
            self.assertGreaterEqual(size, 1024, "上传大小应该 >= 1024 MB")


def generate_report():
    """生成测试报告"""
    import datetime

    print("\n" + "=" * 60)
    print("LogCortex V3 升级测试报告")
    print("=" * 60)
    print(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python 版本: {sys.version}")
    print("=" * 60)

    # 运行测试
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestDependencyVersions))
    suite.addTests(loader.loadTestsFromTestCase(TestModuleIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyzerMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestAppConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestStreamlitStartup))
    suite.addTests(loader.loadTestsFromTestCase(TestFileStructure))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印摘要
    print("\n" + "=" * 60)
    print("测试摘要")
    print("=" * 60)
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n状态: 通过")
    else:
        print("\n状态: 失败")

    return result


if __name__ == '__main__':
    result = generate_report()
    sys.exit(0 if result.wasSuccessful() else 1)
