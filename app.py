import json
import os
import tempfile
import threading
import time
from queue import Queue

# 禁用代理（避免 GLM/DeepSeek API 连接问题）
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(proxy_var, None)

import pandas as pd
import streamlit as st
from openai import OpenAI

from modules.analyzer import LogAnalyzer
from modules.ui_components import (
    render_chart,
    render_linked_subplots,
    render_comparison_chart,
)
from modules.flight_review_views import render_flight_review_dashboard_v2

# 新的 AI Agent 模块
from modules.ai_agent import (
    build_tool_specs,
    execute_tool,
    build_system_prompt,
    ContextManager,
    DIAGNOSTIC_PRESETS,
    get_preset_names,
    get_preset,
    get_preset_prompt,
    parse_slash_command,
    get_help_text,
)


st.set_page_config(
    layout="wide",
    page_title="LogCortex V3",
    page_icon="🚁",
    initial_sidebar_state="collapsed",
)
st.title("🚁 LogCortex V3: 本地日志查看 + AI 分析")


# =============================================================================
# AI Agent 运行器（带上下文管理和进度反馈）
# =============================================================================

def run_ai_agent(
    client,
    model_name: str,
    analyzer,
    user_prompt: str,
    context_manager: ContextManager,
    progress_callback=None,
    tool_callback=None,
    max_steps: int = 20,
):
    """
    运行 AI Agent，带上下文管理和进度反馈

    Args:
        client: OpenAI 客户端
        model_name: 模型名称
        analyzer: LogAnalyzer 实例
        user_prompt: 用户输入
        context_manager: 上下文管理器
        progress_callback: 进度回调函数
        tool_callback: 工具调用回调函数
        max_steps: 最大步骤数

    Returns:
        str: AI 响应内容
    """
    tools = build_tool_specs()
    tool_call_history = []  # 记录工具调用历史

    # 添加用户消息
    context_manager.add_user_message(user_prompt)

    for step in range(max_steps):
        # 获取当前消息
        messages = context_manager.get_messages()

        # 通知进度
        if progress_callback:
            progress_callback(step, max_steps, "thinking")

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            return f"AI 请求失败: {e}"

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        # 如果没有工具调用，返回结果
        if not tool_calls:
            content = msg.content or "未生成有效内容。"
            context_manager.add_assistant_message(content)
            return content

        # 添加助手消息（带工具调用）
        tool_calls_data = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            }
            for tc in tool_calls
        ]
        context_manager.add_assistant_message(tool_calls=tool_calls_data)

        # 执行工具调用
        for tc in tool_calls:
            try:
                tool_args = json.loads(tc.function.arguments or "{}")
            except Exception:
                tool_args = {}

            tool_name = tc.function.name
            tool_call_history.append(tool_name)

            # 通知工具调用
            if tool_callback:
                tool_callback(tool_name, tool_args, "calling")

            if progress_callback:
                progress_callback(step, max_steps, f"tool:{tool_name}")

            # 执行工具
            tool_result = execute_tool(analyzer, tool_name, tool_args)

            # 添加工具结果（自动压缩）
            context_manager.add_tool_result(tc.id, tool_name, tool_result)

            # 通知工具完成
            if tool_callback:
                tool_callback(tool_name, tool_args, "completed", tool_result)

        # 检查是否需要提前终止（重复工具调用）
        if len(tool_call_history) >= 3:
            recent_calls = tool_call_history[-3:]
            if len(set(recent_calls)) == 1:
                # 连续3次调用同一工具，可能陷入循环
                context_manager.add_user_message(
                    "你已经多次调用同一个工具，请根据已有信息给出分析结论。"
                )

    # 达到步数上限，尝试获取部分结论
    stats = context_manager.get_stats()
    tools_used = list(set(tool_call_history))
    return f"""分析已达到 {max_steps} 步的上限。

**已调用的工具**: {', '.join(tools_used)}

**上下文状态**: {stats['total_tokens']} tokens ({stats['utilization']}% 使用)

💡 **建议**: 可以尝试：
1. 使用更具体的预设（如 /quick 快速检查）
2. 清空对话历史后重新开始
3. 缩小问题范围，一次只问一个方面"""


def run_ai_stream(client, model_name: str, messages: list):
    """流式输出（用于普通对话）"""
    full_resp = ""
    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True,
        temperature=0.7,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_resp += delta
            yield delta
    return full_resp


# =============================================================================
# 提供商配置
# =============================================================================

provider_configs = {
    "GLM": {
        "key_label": "GLM API Key",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-5", "glm-4.7", "glm-4.5", "glm-4-air", "glm-4-flash"],
        "default_key": "",
        "default_model": "glm-5",
    },
    "DeepSeek": {
        "key_label": "DeepSeek API Key",
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_key": "",
        "default_model": "deepseek-chat",
    },
}

# =============================================================================
# UI: 连接与日志设置
# =============================================================================

with st.expander("连接与日志设置", expanded=False):
    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        provider = st.selectbox("AI 提供商", ["GLM", "DeepSeek"], index=0)
    provider_cfg = provider_configs[provider]
    with c2:
        env_key = os.getenv("LOGCORTEX_API_KEY", "")
        api_key = st.text_input(
            provider_cfg["key_label"],
            type="password",
            value=env_key or provider_cfg["default_key"],
        )
    with c3:
        use_custom_model = st.checkbox("自定义模型名", value=False)
        if use_custom_model:
            default_model = provider_cfg["default_model"]
            model_name = st.text_input("模型名", value=default_model)
        else:
            default_idx = (
                provider_cfg["models"].index(provider_cfg["default_model"])
                if provider_cfg["default_model"] in provider_cfg["models"]
                else 0
            )
            model_name = st.selectbox("AI 模型", provider_cfg["models"], index=default_idx)
    uploaded_file = st.file_uploader("上传 PX4 日志 (.ulg)", type=["ulg", "ulog"])

client = None
if api_key:
    try:
        client = OpenAI(api_key=api_key, base_url=provider_cfg["base_url"], timeout=60.0)
        st.success(f"🟢 {provider} / {model_name} 已连接")
    except Exception as e:
        st.error(f"🔴 连接失败: {e}")
else:
    st.info("可离线查看日志；展开「连接与日志设置」后填写 Key 可启用 AI。")


# =============================================================================
# Session State 初始化
# =============================================================================

if "analyzer" not in st.session_state:
    st.session_state.analyzer = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "curr_file" not in st.session_state:
    st.session_state.curr_file = None
if "context_manager" not in st.session_state:
    st.session_state.context_manager = None
if "raw_topic_selected" not in st.session_state:
    st.session_state.raw_topic_selected = None
if "favorite_signals" not in st.session_state:
    st.session_state.favorite_signals = {}
if "raw_workspace_pages" not in st.session_state:
    st.session_state.raw_workspace_pages = []
if "compare_basket" not in st.session_state:
    st.session_state.compare_basket = []
if "signal_index" not in st.session_state:
    st.session_state.signal_index = []
if "chart_tabs" not in st.session_state:
    st.session_state.chart_tabs = [{"name": "tab1", "signals": []}]
if "active_chart_tab" not in st.session_state:
    st.session_state.active_chart_tab = 0

# 后台分析相关状态
if "bg_analysis" not in st.session_state:
    st.session_state.bg_analysis = {
        "running": False,
        "status": "",
        "tool_logs": [],
        "result": None,
        "error": None,
        "user_prompt": None,
        "thread": None,
    }


# =============================================================================
# 后台分析函数
# =============================================================================

def run_background_analysis(client, model_name, analyzer, user_prompt, ctx_mgr, max_steps, bg):
    """在后台线程中运行AI分析"""
    import copy

    bg["status"] = "starting"
    bg["tool_logs"] = []
    bg["result"] = None
    bg["error"] = None

    tools = build_tool_specs()
    tool_call_history = []

    # 添加用户消息到上下文
    ctx_mgr.add_user_message(user_prompt)

    try:
        for step in range(max_steps):
            if not bg["running"]:  # 检查是否被取消
                bg["status"] = "cancelled"
                return

            bg["status"] = f"thinking:{step+1}/{max_steps}"
            messages = ctx_mgr.get_messages()

            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                # 完成
                content = msg.content or "未生成有效内容。"
                ctx_mgr.add_assistant_message(content)
                bg["result"] = content
                bg["status"] = "completed"
                return

            # 添加助手消息
            tool_calls_data = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                }
                for tc in tool_calls
            ]
            ctx_mgr.add_assistant_message(tool_calls=tool_calls_data)

            # 执行工具
            for tc in tool_calls:
                tool_name = tc.function.name
                tool_call_history.append(tool_name)
                bg["status"] = f"tool:{tool_name}"

                try:
                    tool_args = json.loads(tc.function.arguments or "{}")
                except:
                    tool_args = {}

                tool_result = execute_tool(analyzer, tool_name, tool_args)
                ctx_mgr.add_tool_result(tc.id, tool_name, tool_result)

                # 记录工具日志
                if "error" not in tool_result:
                    if tool_name == "get_quick_health_check":
                        status_text = "✅ 正常" if tool_result.get("flight_ok") else "⚠️ 有问题"
                        bg["tool_logs"].append(f"✅ `{tool_name}`: {status_text}")
                    elif tool_name == "get_subsystem_summary":
                        sub = tool_result.get("subsystem", "?")
                        st_text = tool_result.get("status", "?")
                        bg["tool_logs"].append(f"✅ `{tool_name}`({sub}): {st_text}")
                    else:
                        bg["tool_logs"].append(f"✅ `{tool_name}`: 完成")
                else:
                    bg["tool_logs"].append(f"❌ `{tool_name}`: 失败")

            # 循环检测
            if len(tool_call_history) >= 3:
                recent = tool_call_history[-3:]
                if len(set(recent)) == 1:
                    ctx_mgr.add_user_message("你已经多次调用同一个工具，请给出结论。")

        # 达到步数上限
        stats = ctx_mgr.get_stats()
        tools_used = list(set(tool_call_history))
        bg["result"] = f"""分析已达到 {max_steps} 步的上限。

**已调用的工具**: {', '.join(tools_used)}

**上下文状态**: {stats['total_tokens']} tokens ({stats['utilization']}% 使用)

💡 **建议**: 可以尝试：
1. 使用更具体的预设（如 /quick 快速检查）
2. 清空对话历史后重新开始
3. 缩小问题范围"""
        bg["status"] = "completed"

    except Exception as e:
        bg["error"] = str(e)
        bg["status"] = "error"


def start_background_analysis(client, model_name, analyzer, user_prompt, ctx_mgr, max_steps):
    """启动后台分析线程"""
    bg = st.session_state.bg_analysis
    bg["running"] = True
    bg["user_prompt"] = user_prompt
    bg["saved"] = False  # 重置保存标志
    bg["result"] = None  # 清除上次结果
    bg["error"] = None   # 清除上次错误

    def run():
        run_background_analysis(client, model_name, analyzer, user_prompt, ctx_mgr, max_steps, bg)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    bg["thread"] = thread


# =============================================================================
# 辅助函数
# =============================================================================

def fuzzy_match(path, query):
    if not query.strip():
        return True
    p = path.lower()
    parts = [x for x in query.lower().strip().split() if x]
    return all(part in p for part in parts)


def build_signal_index(analyzer):
    signals = []
    for topic in analyzer.get_available_topics():
        df = analyzer.get_topic_data(topic, downsample=True)
        if df is None:
            continue
        for field in df.columns:
            if field == "timestamp":
                continue
            if df[field].dtype.kind not in "iufb":
                continue
            signals.append({
                "path": f"{topic}/{field}",
                "topic": topic,
                "field": field,
            })
    signals.sort(key=lambda x: x["path"])
    return signals


# =============================================================================
# 日志上传与解析
# =============================================================================

if uploaded_file:
    if st.session_state.analyzer is None or st.session_state.curr_file != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ulg") as tmp:
            tmp.write(uploaded_file.getvalue())
            path = tmp.name

        try:
            with st.spinner("正在深度解析日志..."):
                analyzer = LogAnalyzer(path)
                st.session_state.analyzer = analyzer
                st.session_state.curr_file = uploaded_file.name
                st.session_state.signal_index = build_signal_index(analyzer)
                st.session_state.chart_tabs = [{"name": "tab1", "signals": []}]
                st.session_state.active_chart_tab = 0

                # 初始化上下文管理器
                ctx_mgr = ContextManager(max_tokens=32000)
                ctx_mgr.add_message("system", build_system_prompt(analyzer))
                st.session_state.context_manager = ctx_mgr

                # 简单的消息历史（用于显示）
                st.session_state.messages = []
        except Exception as e:
            st.error(f"解析严重错误: {e}")


# =============================================================================
# 主页面导航
# =============================================================================

if st.session_state.analyzer:
    analyzer = st.session_state.analyzer

    page = st.radio(
        "页面导航",
        ["📊 飞行概览", "💬 AI 智能分析", "🕒 事件时间线", "⚙️ 参数浏览", "📈 统计与异常", "🔎 原始数据"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # =========================================================================
    # 页面: 飞行概览
    # =========================================================================
    if page == "📊 飞行概览":
        render_flight_review_dashboard_v2(analyzer)

    # =========================================================================
    # 页面: AI 智能分析
    # =========================================================================
    elif page == "💬 AI 智能分析":
        if not client:
            st.error("⚠️ AI 功能不可用：请在上方展开「连接与日志设置」填写 API Key。")
        else:
            # 获取或初始化上下文管理器
            if st.session_state.context_manager is None:
                ctx_mgr = ContextManager(max_tokens=32000)
                ctx_mgr.add_message("system", build_system_prompt(analyzer))
                st.session_state.context_manager = ctx_mgr
            ctx_mgr = st.session_state.context_manager

            # --- 预设诊断按钮 ---
            st.markdown("### 🎯 一键诊断")
            preset_cols = st.columns(4)
            preset_names = get_preset_names()

            selected_preset = None
            for i, preset_info in enumerate(preset_names[:4]):
                with preset_cols[i]:
                    if st.button(
                        f"{preset_info['icon']} {preset_info['name']}",
                        key=f"preset_{preset_info['id']}",
                        use_container_width=True,
                    ):
                        selected_preset = preset_info['id']

            preset_cols2 = st.columns(4)
            for i, preset_info in enumerate(preset_names[4:8]):
                with preset_cols2[i]:
                    if st.button(
                        f"{preset_info['icon']} {preset_info['name']}",
                        key=f"preset_{preset_info['id']}",
                        use_container_width=True,
                    ):
                        selected_preset = preset_info['id']

            st.markdown("---")

            # --- 进度显示区 ---
            progress_placeholder = st.empty()
            tool_log_placeholder = st.empty()

            # --- 对话历史 ---
            chat_container = st.container(height=500)
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    chat_container.chat_message("user").markdown(msg["content"])
                elif msg["role"] == "assistant":
                    chat_container.chat_message("assistant").markdown(msg["content"])

            # --- 输入区 ---
            user_prompt = None
            chat_input = st.chat_input("输入问题，或使用 /quick /full /gps 等命令")

            # 处理预设选择
            if selected_preset:
                preset = get_preset(selected_preset)
                if preset:
                    user_prompt = f"[预设: {preset.name}]\n\n{preset.user_prompt}"

            # 处理聊天输入
            if chat_input:
                # 检查斜杠命令
                slash_preset = parse_slash_command(chat_input)
                if slash_preset == "help":
                    # 显示帮助
                    st.markdown(get_help_text())
                elif slash_preset:
                    preset = get_preset(slash_preset)
                    if preset:
                        user_prompt = f"[预设: {preset.name}]\n\n{preset.user_prompt}"
                else:
                    user_prompt = chat_input

            # --- 执行 AI 分析（后台模式）---
            bg = st.session_state.bg_analysis

            # 启动新的分析任务
            if user_prompt and not bg["running"]:
                # 添加用户消息到历史
                st.session_state.messages.append({"role": "user", "content": user_prompt})

                # 计算最大步数
                stats_before = ctx_mgr.get_stats()
                utilization = stats_before['utilization']
                if utilization < 30:
                    max_steps = 30
                elif utilization < 50:
                    max_steps = 20
                else:
                    max_steps = 15

                # 启动后台分析
                start_background_analysis(
                    client=client,
                    model_name=model_name,
                    analyzer=analyzer,
                    user_prompt=user_prompt,
                    ctx_mgr=ctx_mgr,
                    max_steps=max_steps,
                )
                st.rerun()

            # 显示后台分析状态
            if bg["running"]:
                status = bg.get("status", "")
                if status.startswith("thinking:"):
                    progress_placeholder.info(f"🧠 AI 思考中... ({status.split(':')[1]})")
                elif status.startswith("tool:"):
                    tool_name = status.split(":")[1]
                    progress_placeholder.info(f"🔧 调用工具: `{tool_name}`")
                else:
                    progress_placeholder.info("🔄 正在分析...")

                # 显示工具日志
                if bg.get("tool_logs"):
                    tool_log_placeholder.markdown("\n".join(bg["tool_logs"][-8:]))

                # 检查是否完成
                if bg["status"] == "completed":
                    # 保存结果到消息历史（仅当结果未被保存过）
                    if bg.get("result") and not bg.get("saved"):
                        st.session_state.messages.append({"role": "assistant", "content": bg["result"]})
                        bg["saved"] = True  # 标记已保存
                    bg["running"] = False
                    progress_placeholder.empty()
                    tool_log_placeholder.empty()
                    st.rerun()

                elif bg["status"] == "error":
                    error_msg = f"❌ 分析出错: {bg.get('error', '未知错误')}"
                    if not bg.get("saved"):
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        bg["saved"] = True
                    bg["running"] = False
                    progress_placeholder.empty()
                    tool_log_placeholder.empty()
                    st.error(error_msg)

                elif bg["status"] == "cancelled":
                    bg["running"] = False
                    progress_placeholder.warning("⚠️ 分析已取消")
                    progress_placeholder.empty()
                    tool_log_placeholder.empty()

                else:
                    # 仍在运行，自动刷新
                    time.sleep(0.5)
                    st.rerun()

            # 显示对话历史中的最新消息
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    chat_container.chat_message("user").markdown(msg["content"])
                elif msg["role"] == "assistant":
                    chat_container.chat_message("assistant").markdown(msg["content"])

            # --- 显示上下文信息 ---
            with st.expander("📊 上下文管理", expanded=False):
                stats = ctx_mgr.get_stats()
                col1, col2, col3 = st.columns(3)
                col1.metric("消息数", stats["message_count"])
                col2.metric("Token 数", f"{stats['total_tokens']:,}")
                col3.metric("使用率", f"{stats['utilization']}%")

                if st.button("清空对话历史", key="clear_chat"):
                    st.session_state.messages = []
                    ctx_mgr.clear()
                    ctx_mgr.add_message("system", build_system_prompt(analyzer))
                    st.rerun()

    # =========================================================================
    # 页面: 事件时间线
    # =========================================================================
    elif page == "🕒 事件时间线":
        st.markdown("### 🕒 关键事件时间线")
        max_events = st.slider("事件数量上限", min_value=20, max_value=300, value=120, step=20)
        timeline = analyzer.get_event_timeline(max_events=max_events)
        events = timeline.get("events", [])
        st.caption(f"共 {timeline.get('count', 0)} 条事件")
        if events:
            st.dataframe(events, width="stretch")
        else:
            st.info("未检测到事件")

    # =========================================================================
    # 页面: 参数浏览
    # =========================================================================
    elif page == "⚙️ 参数浏览":
        st.markdown("### ⚙️ 参数浏览")
        c1, c2 = st.columns(2)
        with c1:
            prefix = st.text_input("参数前缀过滤（可选）", value="")
        with c2:
            keyword = st.text_input("参数关键字搜索（可选）", value="")

        params = analyzer.list_parameters(
            prefix=prefix.strip() or None,
            keyword=keyword.strip() or None,
            max_results=2000,
        )
        st.caption(f"匹配参数数量: {len(params)}")
        st.dataframe(params, width="stretch", height=360)

        st.markdown("#### 参数变更记录")
        changes = analyzer.list_parameter_changes(limit=300)
        if changes:
            st.dataframe(changes, width="stretch", height=240)
        else:
            st.info("日志中无参数变更记录")

    # =========================================================================
    # 页面: 统计与异常
    # =========================================================================
    elif page == "📈 统计与异常":
        st.markdown("### 📈 单字段统计与异常检测")
        topic = st.selectbox("选择 Topic", analyzer.get_available_topics(), key="stats_topic")
        fields = analyzer.get_topic_numeric_fields(topic)
        if not fields:
            st.warning("该 Topic 没有可用的数值字段")
        else:
            field = st.selectbox("选择字段", fields)
            threshold = st.slider("异常阈值 (sigma)", min_value=1.5, max_value=5.0, value=3.0, step=0.5)

            c1, c2 = st.columns(2)
            with c1:
                stats = analyzer.compute_field_statistics(topic, field)
                st.markdown("#### 描述性统计")
                st.json(stats)
            with c2:
                anomalies = analyzer.detect_anomalies(topic, field, threshold_std=threshold)
                st.markdown("#### 异常检测")
                st.json(anomalies)

            df = analyzer.get_topic_data(topic, downsample=True)
            if df is not None and field in df.columns:
                render_chart(df, [field], f"{topic}.{field}", height=320)

    # =========================================================================
    # 页面: 原始数据
    # =========================================================================
    elif page == "🔎 原始数据":
        st.markdown("### 🔧 原始话题浏览器（PlotJuggler风格）")
        if not st.session_state.signal_index:
            with st.spinner("建立信号索引中..."):
                st.session_state.signal_index = build_signal_index(analyzer)

        signal_index = st.session_state.signal_index
        path_map = {s["path"]: s for s in signal_index}
        all_paths = [s["path"] for s in signal_index]

        if st.session_state.active_chart_tab >= len(st.session_state.chart_tabs):
            st.session_state.active_chart_tab = max(0, len(st.session_state.chart_tabs) - 1)

        left, right = st.columns([1, 3], gap="large")

        with left:
            st.markdown("#### 信号搜索（叶子级）")
            query = st.text_input("模糊搜索", value="", key="leaf_query")
            matched = [p for p in all_paths if fuzzy_match(p, query)]
            st.caption(f"匹配: {len(matched)} / {len(all_paths)}")

            list_height = st.slider("列表高度", min_value=220, max_value=900, value=520, step=20, key="leaf_list_height")
            show_all = st.checkbox("显示全部匹配信号", value=True, key="leaf_show_all")
            if show_all:
                show_paths = matched
            else:
                max_show = st.slider(
                    "最多显示条数",
                    min_value=200,
                    max_value=max(5000, len(matched) if matched else 5000),
                    value=min(2000, len(matched) if matched else 2000),
                    step=100,
                    key="leaf_max_show",
                )
                show_paths = matched[:max_show]
            leaf_df = pd.DataFrame({
                "选择": [False] * len(show_paths),
                "path": show_paths,
                "topic": [path_map[p]["topic"] for p in show_paths],
                "field": [path_map[p]["field"] for p in show_paths],
            })
            edited = st.data_editor(
                leaf_df,
                width="stretch",
                height=list_height,
                hide_index=True,
                disabled=["path", "topic", "field"],
                column_config={
                    "选择": st.column_config.CheckboxColumn(help="勾选后加入图表页"),
                    "path": st.column_config.TextColumn("信号路径"),
                    "topic": st.column_config.TextColumn("topic"),
                    "field": st.column_config.TextColumn("field"),
                },
                key="leaf_table_editor",
            )
            picked = edited[edited["选择"]]["path"].tolist()
            st.caption(f"已勾选: {len(picked)}")

            add_c1, add_c2 = st.columns(2)
            with add_c1:
                if st.button("加入当前页", width="stretch", key="leaf_add_curr") and picked:
                    tab = st.session_state.chart_tabs[st.session_state.active_chart_tab]
                    for p in picked:
                        if p not in tab["signals"]:
                            tab["signals"].append(p)
                    tab_signals_key = f"tab_signals_{st.session_state.active_chart_tab}"
                    st.session_state[tab_signals_key] = tab["signals"][:]
            with add_c2:
                if st.button("新建页并加入", width="stretch", key="leaf_add_new") and picked:
                    new_name = f"tab{len(st.session_state.chart_tabs)+1}"
                    st.session_state.chart_tabs.append({"name": new_name, "signals": picked[:]})
                    st.session_state.active_chart_tab = len(st.session_state.chart_tabs) - 1
                    st.rerun()

            st.markdown("#### 分级下拉（topic -> field）")
            topic_filter = st.text_input("按topic过滤", value="", key="topic_filter_leaf")
            topics = analyzer.get_available_topics()
            if topic_filter.strip():
                topics = [t for t in topics if topic_filter.lower() in t.lower()]
            if not topics:
                st.warning("没有匹配的 topic")
            else:
                groups = sorted({t.split("_")[0] for t in topics})
                group_selected = st.selectbox("一级分类", groups, key="topic_group_select")
                group_topics = [t for t in topics if t.startswith(group_selected + "_") or t == group_selected]
                topic_selected = st.selectbox("Topic", group_topics, key="topic_leaf_selected")

                topic_df = analyzer.get_topic_data(topic_selected, downsample=True)
                if topic_df is not None:
                    topic_fields = [c for c in topic_df.columns if c != "timestamp" and topic_df[c].dtype.kind in "iufb"]
                    field_df = pd.DataFrame({
                        "选择": [False] * len(topic_fields),
                        "field": topic_fields,
                        "signal": [f"{topic_selected}/{f}" for f in topic_fields],
                    })
                    edited_fields = st.data_editor(
                        field_df,
                        width="stretch",
                        height=260,
                        hide_index=True,
                        disabled=["field", "signal"],
                        column_config={
                            "选择": st.column_config.CheckboxColumn(help="勾选字段"),
                            "field": st.column_config.TextColumn("字段"),
                            "signal": st.column_config.TextColumn("信号路径"),
                        },
                        key=f"topic_field_editor_{topic_selected}",
                    )
                    topic_leaf_pick = edited_fields[edited_fields["选择"]]["signal"].tolist()
                    st.caption(f"该 topic 已勾选字段: {len(topic_leaf_pick)}")

                    tf_c1, tf_c2 = st.columns(2)
                    with tf_c1:
                        if st.button("加入当前页", width="stretch", key=f"leaf_topic_add_curr_{topic_selected}") and topic_leaf_pick:
                            tab = st.session_state.chart_tabs[st.session_state.active_chart_tab]
                            for p in topic_leaf_pick:
                                if p not in tab["signals"]:
                                    tab["signals"].append(p)
                            tab_signals_key = f"tab_signals_{st.session_state.active_chart_tab}"
                            st.session_state[tab_signals_key] = tab["signals"][:]
                    with tf_c2:
                        if st.button("新建页并加入", width="stretch", key=f"leaf_topic_add_new_{topic_selected}") and topic_leaf_pick:
                            new_name = f"tab{len(st.session_state.chart_tabs)+1}"
                            st.session_state.chart_tabs.append({"name": new_name, "signals": topic_leaf_pick[:]})
                            st.session_state.active_chart_tab = len(st.session_state.chart_tabs) - 1
                            st.rerun()

        with right:
            st.markdown("#### 图表页面（可新建多个）")
            tab_names = [f"{i+1}. {t['name']}" for i, t in enumerate(st.session_state.chart_tabs)]
            selected_tab_name = st.selectbox("当前页面", tab_names, index=st.session_state.active_chart_tab, key="chart_tab_select")
            st.session_state.active_chart_tab = tab_names.index(selected_tab_name)

            t_c1, t_c2, t_c3, t_c4 = st.columns([2, 1, 1, 1])
            with t_c1:
                new_tab_name = st.text_input("页面名", value=st.session_state.chart_tabs[st.session_state.active_chart_tab]["name"], key="tab_name_edit")
            with t_c2:
                if st.button("重命名", width="stretch", key="tab_rename_btn"):
                    st.session_state.chart_tabs[st.session_state.active_chart_tab]["name"] = new_tab_name
                    st.rerun()
            with t_c3:
                if st.button("新建页面", width="stretch", key="tab_new_btn"):
                    st.session_state.chart_tabs.append({"name": f"tab{len(st.session_state.chart_tabs)+1}", "signals": []})
                    st.session_state.active_chart_tab = len(st.session_state.chart_tabs) - 1
                    st.rerun()
            with t_c4:
                if st.button("删除页面", width="stretch", key="tab_del_btn") and len(st.session_state.chart_tabs) > 1:
                    st.session_state.chart_tabs.pop(st.session_state.active_chart_tab)
                    st.session_state.active_chart_tab = max(0, st.session_state.active_chart_tab - 1)
                    st.rerun()

            active_tab = st.session_state.chart_tabs[st.session_state.active_chart_tab]
            active_signals = st.multiselect(
                "当前页信号",
                options=all_paths,
                default=active_tab["signals"],
                key=f"tab_signals_{st.session_state.active_chart_tab}",
            )
            active_tab["signals"] = active_signals

            cfg_c1, cfg_c2, cfg_c3 = st.columns([1, 1, 2])
            with cfg_c1:
                downsample_raw = st.checkbox("降采样", value=True, key=f"tab_downsample_{st.session_state.active_chart_tab}")
            with cfg_c2:
                show_rangeslider = st.checkbox("时间滑条", value=True, key=f"tab_rangeslider_{st.session_state.active_chart_tab}")
            with cfg_c3:
                y_mode = st.selectbox(
                    "Y轴模式",
                    ["原始", "标准化(0-1)", "标准化(Z-Score)"],
                    key=f"tab_y_mode_{st.session_state.active_chart_tab}",
                )

            chart_height = st.slider(
                "图表高度",
                min_value=420,
                max_value=1000,
                value=680,
                step=20,
                key=f"tab_chart_height_{st.session_state.active_chart_tab}",
            )

            x_range = st.slider(
                "时间窗口 (s)",
                min_value=0.0,
                max_value=float(analyzer.duration),
                value=(0.0, float(analyzer.duration)),
                step=max(float(analyzer.duration) / 800.0, 0.01),
                key=f"tab_xrange_{st.session_state.active_chart_tab}",
            )

            series_list = []
            for p in active_signals:
                info = path_map.get(p)
                if not info:
                    continue
                df = analyzer.get_topic_data(info["topic"], downsample=downsample_raw)
                if df is None or info["field"] not in df.columns:
                    continue
                dfx = df[(df["timestamp"] >= x_range[0]) & (df["timestamp"] <= x_range[1])]
                series_list.append({"name": p, "x": dfx["timestamp"], "y": dfx[info["field"]]})

            if series_list:
                render_comparison_chart(
                    series_list,
                    title=f"Chart: {active_tab['name']}",
                    height=chart_height,
                    x_range=x_range,
                    show_rangeslider=show_rangeslider,
                    normalize_mode=y_mode,
                )
            else:
                st.info("当前页面还没有信号。请从左侧搜索叶子信号并加入。")

            st.markdown("#### 数据表（当前页）")
            table_pick = st.multiselect(
                "选择要展示数据表的信号（最多3个）",
                options=active_signals,
                default=active_signals[:1],
                key=f"tab_tables_{st.session_state.active_chart_tab}",
            )
            table_pick = table_pick[:3]
            for p in table_pick:
                info = path_map.get(p)
                if not info:
                    continue
                df = analyzer.get_topic_data(info["topic"], downsample=False)
                if df is None or info["field"] not in df.columns:
                    continue
                show_df = df[(df["timestamp"] >= x_range[0]) & (df["timestamp"] <= x_range[1])][["timestamp", info["field"]]]
                with st.expander(f"表格: {p}", expanded=False):
                    st.dataframe(show_df.head(1000), width="stretch")
else:
    st.info("请在上方上传日志文件以开始")
