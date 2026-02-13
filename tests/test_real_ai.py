#!/usr/bin/env python
"""
LogCortex V3 - 实际运行测试
使用真实的日志文件和 AI API 进行测试
"""
import os
import sys
import json

# 清除代理设置，避免连接问题
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

# 设置 API Key
os.environ["LOGCORTEX_API_KEY"] = "b00e23d740524abba55a3072d10bda47.Mno0AlrtfrkG18I8"

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from modules.analyzer import LogAnalyzer
from modules.ai_agent import (
    build_tool_specs,
    execute_tool,
    build_system_prompt,
    ContextManager,
    get_preset,
    get_quick_health_check,
    get_subsystem_summary,
)


def test_analyzer_with_real_log():
    """测试分析器加载真实日志"""
    log_path = "/home/qinzihang/Code/FlightLog/log_25_2025-10-17-17-20-28.ulg"

    print("=" * 60)
    print("测试 1: 加载真实日志文件")
    print("=" * 60)

    try:
        analyzer = LogAnalyzer(log_path)
        print(f"✅ 日志加载成功")
        print(f"   系统: {analyzer.sys_name}")
        print(f"   固件: {analyzer.ver_sw}")
        print(f"   飞行时长: {analyzer.duration:.1f} 秒")
        print(f"   Topic 数量: {len(analyzer.get_available_topics())}")
        return analyzer
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None


def test_l1_summary_tools(analyzer):
    """测试 L1 摘要层工具"""
    print("\n" + "=" * 60)
    print("测试 2: L1 摘要层工具")
    print("=" * 60)

    # 测试快速健康检查
    print("\n2.1 get_quick_health_check:")
    try:
        result = get_quick_health_check(analyzer)
        print(f"   飞行状态: {'✅ 正常' if result.get('flight_ok') else '⚠️ 有问题'}")
        print(f"   飞行时长: {result.get('duration_s')} 秒")
        print(f"   最大高度: {result.get('max_alt_m')} 米")
        print(f"   警告数: {len(result.get('warnings', []))}")
        if result.get('warnings'):
            for w in result.get('warnings', [])[:3]:
                print(f"      - [{w.get('type')}] {w.get('message')}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")

    # 测试子系统摘要
    print("\n2.2 get_subsystem_summary:")
    for subsystem in ['gps', 'battery', 'ekf', 'imu']:
        try:
            result = get_subsystem_summary(analyzer, subsystem)
            status = result.get('status', '?')
            status_icon = {'ok': '✅', 'warning': '⚠️', 'error': '❌'}.get(status, '❓')
            issues = len(result.get('issues', []))
            print(f"   {subsystem}: {status_icon} {status}, 问题: {issues}")
        except Exception as e:
            print(f"   {subsystem}: ❌ {e}")


def test_context_manager(analyzer):
    """测试上下文管理器"""
    print("\n" + "=" * 60)
    print("测试 3: 上下文管理器")
    print("=" * 60)

    ctx = ContextManager(max_tokens=32000)
    ctx.add_message("system", build_system_prompt(analyzer))
    ctx.add_user_message("请分析这次飞行")

    print(f"   初始 Token 数: {ctx.total_tokens()}")

    # 模拟添加工具结果
    health_check = get_quick_health_check(analyzer)
    ctx.add_tool_result("tc_1", "get_quick_health_check", health_check)

    print(f"   添加工具结果后 Token 数: {ctx.total_tokens()}")

    stats = ctx.get_stats()
    print(f"   统计: {stats['message_count']} 消息, {stats['utilization']}% 使用")


def test_glm_connection():
    """测试 GLM API 连接"""
    print("\n" + "=" * 60)
    print("测试 4: GLM API 连接")
    print("=" * 60)

    api_key = os.environ.get("LOGCORTEX_API_KEY")
    base_url = "https://open.bigmodel.cn/api/paas/v4"

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0
        )

        # 简单测试
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": "回复 OK 两个字母"}],
            max_tokens=10
        )
        content = response.choices[0].message.content
        print(f"   ✅ GLM 连接成功")
        print(f"   响应: {content}")
        return client
    except Exception as e:
        print(f"   ❌ 连接失败: {e}")
        return None


def test_ai_agent_with_tools(analyzer, client):
    """测试完整的 AI Agent 流程"""
    print("\n" + "=" * 60)
    print("测试 5: AI Agent 完整流程")
    print("=" * 60)

    if not client:
        print("   ⏭️ 跳过 (API 连接失败)")
        return

    tools = build_tool_specs()
    ctx = ContextManager(max_tokens=16000)
    ctx.add_message("system", build_system_prompt(analyzer))

    # 使用快速检查预设
    preset = get_preset("quick_health")
    user_prompt = preset.user_prompt if preset else "请快速检查这次飞行的健康状态"
    ctx.add_user_message(user_prompt)

    print(f"   用户输入: {user_prompt[:50]}...")

    max_steps = 5
    for step in range(max_steps):
        print(f"\n   步骤 {step + 1}/{max_steps}:")

        try:
            resp = client.chat.completions.create(
                model="glm-4-flash",
                messages=ctx.get_messages(),
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            print(f"      ❌ API 错误: {e}")
            break

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            # 完成
            content = msg.content or "未生成内容"
            print(f"      ✅ AI 响应完成")
            print(f"      Token 使用: {ctx.total_tokens()}")
            print(f"\n   --- AI 响应 ---")
            print(f"   {content[:500]}...")
            break

        # 处理工具调用
        tool_calls_data = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            }
            for tc in tool_calls
        ]
        ctx.add_assistant_message(tool_calls=tool_calls_data)

        for tc in tool_calls:
            tool_name = tc.function.name
            print(f"      🔧 调用工具: {tool_name}")

            try:
                tool_args = json.loads(tc.function.arguments or "{}")
            except:
                tool_args = {}

            result = execute_tool(analyzer, tool_name, tool_args)
            ctx.add_tool_result(tc.id, tool_name, result)

            # 显示结果摘要
            if "error" not in result:
                if tool_name == "get_quick_health_check":
                    status = "正常" if result.get("flight_ok") else "有问题"
                    print(f"         → 状态: {status}")
                elif tool_name == "get_subsystem_summary":
                    print(f"         → {result.get('subsystem')}: {result.get('status')}")
                else:
                    print(f"         → 完成")
            else:
                print(f"         → 错误: {result.get('error')}")


def main():
    print("\n" + "=" * 60)
    print("LogCortex V3 - 实际运行测试")
    print("=" * 60)

    # 测试 1: 加载日志
    analyzer = test_analyzer_with_real_log()
    if not analyzer:
        print("\n❌ 无法加载日志，测试终止")
        return

    # 测试 2: L1 工具
    test_l1_summary_tools(analyzer)

    # 测试 3: 上下文管理器
    test_context_manager(analyzer)

    # 测试 4: GLM 连接
    client = test_glm_connection()

    # 测试 5: 完整 AI Agent 流程
    test_ai_agent_with_tools(analyzer, client)

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
