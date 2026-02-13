"""
LogCortex V3 - AI Agent 模块测试
测试新的 AI 功能增强
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAIAgentImports(unittest.TestCase):
    """AI Agent 模块导入测试"""

    def test_01_import_tools(self):
        """测试 tools 模块导入"""
        from modules.ai_agent.tools import (
            build_tool_specs,
            execute_tool,
            get_quick_health_check,
            get_subsystem_summary,
            get_signal_stats,
        )
        self.assertTrue(callable(build_tool_specs))
        self.assertTrue(callable(execute_tool))
        self.assertTrue(callable(get_quick_health_check))
        self.assertTrue(callable(get_subsystem_summary))

    def test_02_import_prompts(self):
        """测试 prompts 模块导入"""
        from modules.ai_agent.prompts import (
            SYSTEM_PROMPT_TEMPLATE,
            DIAGNOSTIC_STAGE_PROMPTS,
            build_system_prompt,
        )
        self.assertIsInstance(SYSTEM_PROMPT_TEMPLATE, str)
        self.assertIsInstance(DIAGNOSTIC_STAGE_PROMPTS, dict)
        self.assertTrue(callable(build_system_prompt))

    def test_03_import_context_manager(self):
        """测试 context_manager 模块导入"""
        from modules.ai_agent.context_manager import ContextManager
        self.assertTrue(callable(ContextManager))

    def test_04_import_diagnostic_flow(self):
        """测试 diagnostic_flow 模块导入"""
        from modules.ai_agent.diagnostic_flow import (
            DiagnosticFlow,
            DIAGNOSTIC_STAGES,
        )
        self.assertTrue(callable(DiagnosticFlow))
        self.assertIsInstance(DIAGNOSTIC_STAGES, list)

    def test_05_import_presets(self):
        """测试 presets 模块导入"""
        from modules.ai_agent.presets import (
            DIAGNOSTIC_PRESETS,
            get_preset_names,
            get_preset,
            parse_slash_command,
            get_help_text,
        )
        self.assertIsInstance(DIAGNOSTIC_PRESETS, dict)
        self.assertTrue(callable(get_preset_names))
        self.assertTrue(callable(get_preset))
        self.assertTrue(callable(parse_slash_command))


class TestToolSpecs(unittest.TestCase):
    """工具规格测试"""

    def test_01_build_tool_specs(self):
        """测试工具规格构建"""
        from modules.ai_agent.tools import build_tool_specs
        specs = build_tool_specs()
        self.assertIsInstance(specs, list)
        self.assertTrue(len(specs) >= 8)

        # 检查工具名称
        tool_names = [s["function"]["name"] for s in specs]
        self.assertIn("get_quick_health_check", tool_names)
        self.assertIn("get_subsystem_summary", tool_names)
        self.assertIn("get_signal_stats", tool_names)
        self.assertIn("get_event_timeline", tool_names)

    def test_02_tool_has_required_fields(self):
        """测试工具规格包含必需字段"""
        from modules.ai_agent.tools import build_tool_specs
        specs = build_tool_specs()

        for spec in specs:
            self.assertEqual(spec["type"], "function")
            self.assertIn("name", spec["function"])
            self.assertIn("description", spec["function"])
            self.assertIn("parameters", spec["function"])


class TestContextManager(unittest.TestCase):
    """上下文管理器测试"""

    def test_01_create_context_manager(self):
        """测试创建上下文管理器"""
        from modules.ai_agent.context_manager import ContextManager
        ctx = ContextManager(max_tokens=10000)
        self.assertEqual(ctx.max_tokens, 10000)
        self.assertEqual(len(ctx.messages), 0)

    def test_02_add_messages(self):
        """测试添加消息"""
        from modules.ai_agent.context_manager import ContextManager
        ctx = ContextManager(max_tokens=10000)
        ctx.add_user_message("Hello")
        ctx.add_assistant_message("Hi there")
        self.assertEqual(len(ctx.messages), 2)

    def test_03_estimate_tokens(self):
        """测试 token 估算"""
        from modules.ai_agent.context_manager import ContextManager
        ctx = ContextManager()
        tokens = ctx.estimate_tokens("Hello World")
        self.assertGreater(tokens, 0)

    def test_04_tool_result_compression(self):
        """测试工具结果压缩"""
        from modules.ai_agent.context_manager import ContextManager
        ctx = ContextManager(max_tokens=10000, compression_threshold=500)

        # 创建一个大结果
        large_result = {
            "events": [{"t": i, "data": f"event_{i}"} for i in range(100)]
        }

        ctx.add_tool_result("tc_1", "get_event_timeline", large_result)

        # 应该被压缩
        self.assertIn("tc_1", ctx._tool_results_cache)
        self.assertEqual(len(ctx.messages), 1)

    def test_05_get_stats(self):
        """测试统计信息"""
        from modules.ai_agent.context_manager import ContextManager
        ctx = ContextManager(max_tokens=10000)
        ctx.add_user_message("Hello")
        stats = ctx.get_stats()
        self.assertIn("message_count", stats)
        self.assertIn("total_tokens", stats)
        self.assertEqual(stats["message_count"], 1)


class TestPresets(unittest.TestCase):
    """预设模板测试"""

    def test_01_get_preset_names(self):
        """测试获取预设名称"""
        from modules.ai_agent.presets import get_preset_names
        names = get_preset_names()
        self.assertIsInstance(names, list)
        self.assertTrue(len(names) >= 4)

        # 检查必需字段
        for preset in names:
            self.assertIn("id", preset)
            self.assertIn("name", preset)
            self.assertIn("icon", preset)

    def test_02_get_preset(self):
        """测试获取预设"""
        from modules.ai_agent.presets import get_preset
        preset = get_preset("quick_health")
        self.assertIsNotNone(preset)
        self.assertEqual(preset.id, "quick_health")

    def test_03_parse_slash_command(self):
        """测试斜杠命令解析"""
        from modules.ai_agent.presets import parse_slash_command

        self.assertEqual(parse_slash_command("/quick"), "quick_health")
        self.assertEqual(parse_slash_command("/full"), "full_diagnostic")
        self.assertEqual(parse_slash_command("/gps"), "gps_investigation")
        self.assertIsNone(parse_slash_command("/unknown"))

    def test_04_get_help_text(self):
        """测试帮助文本"""
        from modules.ai_agent.presets import get_help_text
        help_text = get_help_text()
        self.assertIn("/quick", help_text)
        self.assertIn("/full", help_text)


class TestDiagnosticFlow(unittest.TestCase):
    """诊断流程测试"""

    def test_01_create_flow(self):
        """测试创建诊断流程"""
        from modules.ai_agent.diagnostic_flow import DiagnosticFlow, DIAGNOSTIC_STAGES
        flow = DiagnosticFlow(stages=DIAGNOSTIC_STAGES)
        self.assertEqual(len(flow.stages), len(DIAGNOSTIC_STAGES))

    def test_02_flow_progress(self):
        """测试流程进度"""
        from modules.ai_agent.diagnostic_flow import DiagnosticFlow, DIAGNOSTIC_STAGES
        flow = DiagnosticFlow(stages=DIAGNOSTIC_STAGES[:3])
        flow.start()

        progress = flow.get_progress()
        self.assertIn("completed_stages", progress)
        self.assertIn("total_stages", progress)
        self.assertIn("progress_pct", progress)


class TestPrompts(unittest.TestCase):
    """Prompt 模板测试"""

    def test_01_system_prompt_template(self):
        """测试系统 prompt 模板"""
        from modules.ai_agent.prompts import SYSTEM_PROMPT_TEMPLATE
        self.assertIn("PX4", SYSTEM_PROMPT_TEMPLATE)
        self.assertIn("get_quick_health_check", SYSTEM_PROMPT_TEMPLATE)

    def test_02_diagnostic_stage_prompts(self):
        """测试诊断阶段 prompt"""
        from modules.ai_agent.prompts import DIAGNOSTIC_STAGE_PROMPTS
        self.assertIn("preflight", DIAGNOSTIC_STAGE_PROMPTS)
        self.assertIn("subsystem_check", DIAGNOSTIC_STAGE_PROMPTS)

    def test_03_preset_prompts(self):
        """测试预设 prompt"""
        from modules.ai_agent.prompts import PRESET_PROMPTS
        self.assertIn("quick_health", PRESET_PROMPTS)
        self.assertIn("full_diagnostic", PRESET_PROMPTS)


class TestSubSystemSummary(unittest.TestCase):
    """子系统摘要测试（无需实际日志）"""

    def test_01_subsystem_list(self):
        """测试子系统列表"""
        from modules.ai_agent.tools import get_subsystem_summary
        # 测试无效子系统
        result = get_subsystem_summary(None, "invalid_subsystem")
        self.assertIn("error", result)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestAIAgentImports))
    suite.addTests(loader.loadTestsFromTestCase(TestToolSpecs))
    suite.addTests(loader.loadTestsFromTestCase(TestContextManager))
    suite.addTests(loader.loadTestsFromTestCase(TestPresets))
    suite.addTests(loader.loadTestsFromTestCase(TestDiagnosticFlow))
    suite.addTests(loader.loadTestsFromTestCase(TestPrompts))
    suite.addTests(loader.loadTestsFromTestCase(TestSubSystemSummary))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
