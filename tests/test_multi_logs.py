#!/usr/bin/env python
"""
LogCortex V3 - 多日志测试
测试所有日志文件
"""
import os
import sys
import json

# 清除代理
for proxy in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(proxy, None)

os.environ["LOGCORTEX_API_KEY"] = "b00e23d740524abba55a3072d10bda47.Mno0AlrtfrkG18I8"

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
)


def test_all_logs():
    """测试所有日志文件"""
    log_dir = "/home/qinzihang/Code/FlightLog"
    log_files = [
        "log_25_2025-10-17-17-20-28.ulg",
        "log_26_2025-10-17-17-27-28.ulg",
        "log_27_2025-10-17-17-31-20.ulg",
    ]

    print("=" * 60)
    print("LogCortex V3 - 多日志 AI 分析测试")
    print("=" * 60)

    # 初始化 GLM 客户端
    client = OpenAI(
        api_key=os.environ["LOGCORTEX_API_KEY"],
        base_url="https://open.bigmodel.cn/api/paas/v4",
        timeout=60.0
    )

    for log_file in log_files:
        log_path = os.path.join(log_dir, log_file)
        print(f"\n{'='*60}")
        print(f"测试日志: {log_file}")
        print("=" * 60)

        try:
            # 加载日志
            analyzer = LogAnalyzer(log_path)
            print(f"✅ 加载成功 - 时长: {analyzer.duration:.1f}s, Topics: {len(analyzer.get_available_topics())}")

            # 快速健康检查
            health = get_quick_health_check(analyzer)
            status = "✅ 正常" if health.get("flight_ok") else "⚠️ 有问题"
            print(f"   健康状态: {status}")
            print(f"   最大高度: {health.get('max_alt_m', 0):.1f}m")
            print(f"   最大速度: {health.get('max_speed_mps', 0):.1f}m/s")

            if health.get("warnings"):
                print(f"   警告 ({len(health['warnings'])}):")
                for w in health["warnings"][:2]:
                    print(f"      - {w.get('type')}: {w.get('message')}")

            # AI 分析
            print(f"\n   🤖 AI 分析中...")
            ctx = ContextManager(max_tokens=16000)
            ctx.add_message("system", build_system_prompt(analyzer))

            preset = get_preset("quick_health")
            ctx.add_user_message(preset.user_prompt)

            tools = build_tool_specs()
            final_response = None

            for step in range(5):
                resp = client.chat.completions.create(
                    model="glm-4-flash",
                    messages=ctx.get_messages(),
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.2,
                )

                msg = resp.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None)

                if not tool_calls:
                    final_response = msg.content
                    break

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
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except:
                        args = {}
                    result = execute_tool(analyzer, tc.function.name, args)
                    ctx.add_tool_result(tc.id, tc.function.name, result)

            if final_response:
                # 显示 AI 响应的前几行
                lines = final_response.split('\n')[:5]
                print(f"   AI 响应预览:")
                for line in lines:
                    if line.strip():
                        print(f"      {line[:60]}...")
                print(f"   Token 使用: {ctx.total_tokens()}")

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("多日志测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_all_logs()
