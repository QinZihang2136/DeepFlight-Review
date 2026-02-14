"""
Flight Review 风格飞行概览页面渲染器。
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.flight_review_layout import (
    ACCEL_AXIS_CANDIDATES,
    ACTUATOR_FFT_CANDIDATES,
    ANGULAR_RATE_FFT_CANDIDATES,
    FLIGHT_REVIEW_GROUPS,
    GYRO_AXIS_CANDIDATES,
    IMU_CUTOFF_PARAMS,
    MODE_COLORS,
)
from modules.ui_components import render_map


def _find_first_topic(analyzer, candidates):
    for t in candidates:
        df = analyzer.get_topic(t, downsample=True)
        if df is not None:
            return t
    return None


def _add_mode_background(fig, mode_segments, t0, t1):
    for seg in mode_segments:
        s0 = max(float(seg["t0"]), float(t0))
        s1 = min(float(seg["t1"]), float(t1))
        if s1 <= s0:
            continue
        c = MODE_COLORS.get(seg.get("name", ""), "rgba(120,120,120,0.08)")
        fig.add_vrect(x0=s0, x1=s1, fillcolor=c, line_width=0)


def _resolve_topic_by_prefix(analyzer, prefix):
    topics = analyzer.get_available_topics()
    for t in topics:
        if t == prefix or t.startswith(prefix):
            return t
    return None


def _first_valid_candidate(analyzer, candidates):
    topics = analyzer.get_available_topics()
    for item in candidates:
        topic = item.get("topic")
        if not topic and item.get("topic_prefix"):
            topic = _resolve_topic_by_prefix(analyzer, item["topic_prefix"])
        if topic and topic in topics:
            return topic, item
    return None, None


def _plot_group(analyzer, group, t0, t1, use_downsample, show_rangeslider, mode_segments):
    topic = _find_first_topic(analyzer, group["topic_candidates"])
    if topic is None:
        st.info(f"缺失主题: 期望 {group['topic_candidates']}")
        return

    df = analyzer.get_topic(topic, downsample=use_downsample)
    if df is None or "timestamp" not in df.columns:
        st.info(f"主题不可用: {topic}")
        return

    df = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)]
    if df.empty:
        st.info(f"时间窗口内无数据: {topic}")
        return

    # Actuator / RC 组做动态字段聚合
    if group["key"] == "actuators":
        dynamic = [c for c in df.columns if c != "timestamp" and ("output" in c or "control" in c)]
        signals = [(c, c) for c in dynamic[:16]]
    elif group["key"] == "rc":
        # 扩展字段匹配，支持更多命名方式（Flight Review 兼容）
        dynamic = [c for c in df.columns if c != "timestamp" and (
            "channel" in c.lower() or
            c in ["x", "y", "z", "r"] or
            c in ["roll", "pitch", "yaw", "throttle"] or
            "values" in c.lower() or
            c.startswith("sticks") or
            c in ["aux1", "aux2", "aux3", "aux4", "aux5", "aux6"]
        )]
        signals = [(c, c) for c in dynamic[:12]]
        if not signals:
            # 如果没找到匹配字段，显示所有数值字段
            dynamic = [c for c in df.columns if c != "timestamp" and df[c].dtype.kind in "iufb"]
            signals = [(c, c) for c in dynamic[:8]]
    else:
        signals = group["signals"]

    valid = [(col, name) for col, name in signals if col in df.columns]
    if not valid:
        st.info(f"无可用字段: topic={topic}, expected={[s[0] for s in signals]}")
        st.caption(f"实际可用字段: {[c for c in df.columns if c != 'timestamp'][:10]}")
        return

    fig = go.Figure()
    _add_mode_background(fig, mode_segments, t0, t1)

    # 检查是否有 setpoint 配置
    setpoint_topic = group.get("setpoint_topic")
    setpoint_signals = group.get("setpoint_signals", [])
    sp_df = None

    if setpoint_topic:
        sp_df = analyzer.get_topic(setpoint_topic, downsample=use_downsample)
        if sp_df is not None and "timestamp" in sp_df.columns:
            sp_df = sp_df[(sp_df["timestamp"] >= t0) & (sp_df["timestamp"] <= t1)]

    # Flight Review 标准颜色
    # Setpoint = 绿色虚线, Estimated = 橙色实线
    ESTIMATED_COLOR = "#FF7F0E"  # 橙色 (Plotly 橙)
    SETPOINT_COLOR = "#2CA02C"   # 绿色 (Plotly 绿)

    has_setpoint = sp_df is not None and not sp_df.empty and len(setpoint_signals) > 0

    # 如果有 setpoint，分别绘制每个轴的 Setpoint 和 Estimated
    # 图例独立显示，可以单独显示/隐藏
    if has_setpoint:
        for idx, (col, name) in enumerate(valid):
            # 先绘制 Setpoint（绿色虚线）
            if idx < len(setpoint_signals):
                sp_col, sp_name = setpoint_signals[idx]
                if sp_col in sp_df.columns:
                    fig.add_trace(go.Scatter(
                        x=sp_df["timestamp"], y=sp_df[sp_col],
                        name=sp_name,
                        line=dict(width=1.5, color=SETPOINT_COLOR, dash="dash"),
                        showlegend=True,
                    ))

            # 再绘制 Estimated（橙色实线）
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df[col],
                name=name,
                line=dict(width=1.5, color=ESTIMATED_COLOR),
                showlegend=True,
            ))
    else:
        # 没有 setpoint，使用原来的多彩显示
        for idx, (col, name) in enumerate(valid):
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df[col],
                name=name,
                line=dict(width=1.5, color=px.colors.qualitative.Plotly[idx % 10]),
            ))

    # Power 图表特殊处理：添加 5V 和 3.3V 阈值参考线
    if group["key"] == "power":
        # 添加 5V 参考线（Flight Review 标准）
        fig.add_hline(
            y=5.0,
            line_dash="dot",
            line_color="rgba(255,165,0,0.6)",
            annotation_text="5V",
            annotation_position="right",
            annotation_font_size=9,
        )
        # 添加 3.3V 参考线（Flight Review 标准）
        fig.add_hline(
            y=3.3,
            line_dash="dot",
            line_color="rgba(255,165,0,0.6)",
            annotation_text="3.3V",
            annotation_position="right",
            annotation_font_size=9,
        )

    # Flight Review 标准图表样式
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=20, b=40),
        hovermode="x unified",
        xaxis=dict(
            showgrid=True,
            gridcolor="#e8e8e8",
            rangeslider=dict(visible=show_rangeslider),
            range=[float(t0), float(t1)],
        ),
        yaxis=dict(showgrid=True, gridcolor="#e8e8e8"),
        # 图例放在图表内部右上角（Flight Review 风格）
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ddd",
            borderwidth=1,
            font=dict(size=10, color="#333"),
        ),
        # 白色背景（Flight Review 风格）
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})

    # 如果有 setpoint，添加图例说明
    if has_setpoint:
        st.caption(f"🟢 --- Setpoint (目标值)  |  🟠 Estimated (实际值)")


def _render_status_cards(summary):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gps = summary.get("gps") or {}
        st.metric("GPS Fix", str(gps.get("best_fix_type", "N/A")))
    with c2:
        batt = summary.get("battery") or {}
        rem = batt.get("min_remaining")
        st.metric("Battery Min", f"{rem*100:.1f}%" if isinstance(rem, (int, float)) else "N/A")
    with c3:
        ekf = summary.get("ekf") or {}
        st.metric("EKF Reset (Vert)", str(ekf.get("pos_vert_reset_counter", "N/A")))
    with c4:
        failsafe_changes = summary.get("failsafe_changes") or []
        fs_count = sum(1 for x in failsafe_changes if bool(x.get("failsafe")))
        st.metric("Failsafe Events", str(fs_count))


def _render_events_and_messages(analyzer, t0, t1):
    st.markdown("### Messages / Events")
    c1, c2 = st.columns([1, 1])
    with c1:
        timeline = analyzer.get_event_timeline(max_events=400).get("events", [])
        if timeline:
            df = pd.DataFrame(timeline)
            df = df[(df["t_s"] >= t0) & (df["t_s"] <= t1)]
            st.dataframe(df, width="stretch", height=260)
        else:
            st.info("无事件时间线数据")
    with c2:
        msg_df = analyzer.get_messages(limit=2000)
        if not msg_df.empty:
            msg_df = msg_df[(msg_df["timestamp"] >= t0) & (msg_df["timestamp"] <= t1)]
            st.dataframe(msg_df, width="stretch", height=260)
        else:
            st.info("无日志消息数据")

    st.markdown("### Parameters Changed")
    p_df = analyzer.get_parameter_changes(limit=2000)
    if p_df.empty:
        st.info("无参数变化")
    else:
        p_df = p_df[(p_df["timestamp"] >= t0) & (p_df["timestamp"] <= t1)]
        st.dataframe(p_df, width="stretch", height=220)


def _build_cutoff_param_map(analyzer):
    params = analyzer.list_parameters(prefix="IMU_", max_results=1000)
    out = {}
    for p in params:
        name = str(p.get("name", ""))
        if name in IMU_CUTOFF_PARAMS:
            out[name] = p.get("value")
    return out


def _render_gps_noise_jamming_panel(analyzer, t0, t1, mode_segments):
    st.markdown("#### GPS Noise & Jamming")
    df = analyzer.get_gps_noise_jamming(t0=t0, t1=t1)

    if df.empty:
        st.info("GPS 噪声/干扰数据不可用")
        st.caption("可能原因：GPS 接收器不支持此功能或日志格式不兼容")
        # 显示 GPS topic 的可用字段
        gps_topics = [t for t in analyzer.get_available_topics() if "gps" in t.lower()]
        if gps_topics:
            st.caption(f"可用 GPS topics: {gps_topics[:3]}")
        return

    # 检查数据是否全为空或 0
    noise_valid = "noise_per_ms" in df.columns and df["noise_per_ms"].notna().any()
    jam_valid = "jamming_indicator" in df.columns and df["jamming_indicator"].notna().any()

    if not noise_valid and not jam_valid:
        st.warning("GPS 噪声/干扰数据存在但全为空值")
        st.caption(f"返回字段: {list(df.columns)}")
        return

    fig = go.Figure()
    _add_mode_background(fig, mode_segments, t0, t1)

    has_traces = False
    if "noise_per_ms" in df.columns and df["noise_per_ms"].notna().any():
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["noise_per_ms"], name="Noise per ms", line=dict(width=1.4)))
        has_traces = True
    if "jamming_indicator" in df.columns and df["jamming_indicator"].notna().any():
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["jamming_indicator"], name="Jamming Indicator", line=dict(width=1.4))
        )
        has_traces = True

    if not has_traces:
        st.info("GPS 噪声/干扰数据中没有有效值")
        return

    # Flight Review 标准图表样式
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=20, b=40),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#e8e8e8", range=[float(t0), float(t1)]),
        yaxis=dict(showgrid=True, gridcolor="#e8e8e8"),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#ddd",
            borderwidth=1,
            font=dict(size=10, color="#333"),
        ),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})
    # 添加 Flight Review 标准参考
    st.caption("参考: Jamming Indicator ≤40 正常, ≥80 需检查")


def _render_thrust_magnetic_panel(analyzer, t0, t1, mode_segments):
    """Render Thrust & Magnetic Field chart (Flight Review standard).

    Flight Review uses a single Y-axis (0-1 range) with both signals normalized.
    Colors: Green = Thrust, Red = Norm of Magnetic Field
    """
    st.markdown("#### Thrust & Magnetic Field")
    df = analyzer.get_thrust_and_magnetic(t0=t0, t1=t1)
    if df.empty:
        st.info("缺失推力或磁场数据: 期望 actuator_controls/actuator_motors + sensor_mag")
        return

    fig = go.Figure()
    _add_mode_background(fig, mode_segments, t0, t1)

    has_thrust = "thrust" in df.columns and df["thrust"].notna().any()
    has_mag = "mag_norm" in df.columns and df["mag_norm"].notna().any()

    # Flight Review standard colors
    THRUST_COLOR = "#2ca02c"  # Green (Flight Review standard)
    MAG_NORM_COLOR = "#d62728"  # Red (Flight Review standard)

    if has_thrust:
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["thrust"],
            name="Thrust",
            line=dict(width=1.5, color=THRUST_COLOR),
        ))

    if has_mag:
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["mag_norm"],
            name="Norm of Magnetic Field",
            line=dict(width=1.5, color=MAG_NORM_COLOR),
        ))

    if not has_thrust and not has_mag:
        st.info("推力和磁场数据中没有有效值")
        return

    # Flight Review 标准图表样式
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=20, b=40),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#e8e8e8", range=[float(t0), float(t1)]),
        yaxis=dict(showgrid=True, gridcolor="#e8e8e8"),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#ddd",
            borderwidth=1,
            font=dict(size=10, color="#333"),
        ),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})
    st.caption("参考: 磁场范数应保持恒定且与推力无关。若相关联则表示电机电流影响罗盘。")


def _render_fft_panel(analyzer, panel_title, candidates, t0, t1, cutoff_map=None):
    st.markdown(f"#### {panel_title}")
    topic, chosen = _first_valid_candidate(analyzer, candidates)
    if not topic or not chosen:
        st.info(f"缺失主题: 期望 {[x.get('topic') or x.get('topic_prefix') for x in candidates]}")
        return

    fields = chosen["fields"]
    labels = chosen["labels"]
    fft_map = analyzer.compute_fft_multi(topic, fields, t0=t0, t1=t1, window="hann")
    fig = go.Figure()
    has_trace = False
    scale = 57.29577951308232 if chosen.get("to_deg") else 1.0
    for field, label in zip(fields, labels):
        df = fft_map.get(field)
        if df is None or df.empty:
            continue
        y = df["amplitude"] * scale
        fig.add_trace(go.Scatter(x=df["freq_hz"], y=y, name=label, line=dict(width=1.4)))
        has_trace = True

    if not has_trace:
        st.info(f"数据不足: topic={topic}, fields={fields}")
        return

    if cutoff_map:
        for p in IMU_CUTOFF_PARAMS:
            value = cutoff_map.get(p)
            try:
                if value is None:
                    continue
                x = float(value)
            except Exception:
                continue
            fig.add_vline(x=x, line_dash="dash", line_color="rgba(20,20,20,0.65)")
            fig.add_annotation(x=x, y=1.0, xref="x", yref="paper", text=p, showarrow=False, yanchor="bottom")

    # Flight Review 标准图表样式
    fig.update_layout(
        height=320,
        margin=dict(l=50, r=20, t=20, b=40),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#e8e8e8", title="Hz"),
        yaxis=dict(showgrid=True, gridcolor="#e8e8e8", title="Amplitude"),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#ddd",
            borderwidth=1,
            font=dict(size=10, color="#333"),
        ),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})


def _render_single_spectrogram(analyzer, panel_title, topic, field, t0, t1, max_hz):
    st.markdown(f"#### {panel_title}")
    spec = analyzer.compute_spectrogram(topic, field, t0=t0, t1=t1, nperseg=512, noverlap=256)
    if spec.empty:
        st.info(f"时频数据不足: topic={topic}, field={field}")
        return
    if max_hz is not None:
        spec = spec[spec["freq_hz"] <= float(max_hz)]
    grid = spec.pivot_table(index="freq_hz", columns="time_s", values="power_db", aggfunc="mean").sort_index()
    if grid.empty:
        st.info(f"无法生成时频图: topic={topic}, field={field}")
        return

    fig = go.Figure(
        data=go.Heatmap(
            z=grid.values,
            x=grid.columns.to_numpy(),
            y=grid.index.to_numpy(),
            colorscale="Viridis",
            colorbar=dict(title="[dB]"),
        )
    )
    # Flight Review 标准图表样式
    fig.update_layout(
        height=340,
        margin=dict(l=50, r=20, t=20, b=40),
        xaxis=dict(title="Time [s]", range=[float(t0), float(t1)], showgrid=True, gridcolor="#e8e8e8"),
        yaxis=dict(title="Hz", showgrid=True, gridcolor="#e8e8e8"),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})


def _render_vibration_spectrogram_panel(analyzer, t0, t1):
    acc_topic, acc_cfg = _first_valid_candidate(analyzer, ACCEL_AXIS_CANDIDATES)
    gyro_topic, gyro_cfg = _first_valid_candidate(analyzer, GYRO_AXIS_CANDIDATES)

    if not acc_topic:
        st.info("缺失加速度主题: 期望 sensor_combined 或 sensor_accel")
    else:
        _render_single_spectrogram(
            analyzer=analyzer,
            panel_title="Acceleration Power Spectral Density",
            topic=acc_topic,
            field=acc_cfg["fields"][2],
            t0=t0,
            t1=t1,
            max_hz=100,
        )

    if not gyro_topic:
        st.info("缺失角速度主题: 期望 sensor_combined 或 vehicle_angular_velocity")
    else:
        _render_single_spectrogram(
            analyzer=analyzer,
            panel_title="Angular Velocity Power Spectral Density",
            topic=gyro_topic,
            field=gyro_cfg["fields"][2],
            t0=t0,
            t1=t1,
            max_hz=140,
        )


def _render_frequency(analyzer, t0, t1):
    st.markdown("### Frequency Analysis (FFT / PSD)")
    cutoff_map = _build_cutoff_param_map(analyzer)
    _render_fft_panel(analyzer, "Actuator Controls FFT", ACTUATOR_FFT_CANDIDATES, t0, t1, cutoff_map=cutoff_map)
    _render_fft_panel(analyzer, "Angular Velocity FFT", ANGULAR_RATE_FFT_CANDIDATES, t0, t1, cutoff_map=cutoff_map)
    _render_vibration_spectrogram_panel(analyzer, t0, t1)

    st.markdown("#### Custom Signal FFT / PSD")
    topics = analyzer.get_available_topics()
    f_c1, f_c2, f_c3 = st.columns([2, 2, 1])
    with f_c1:
        topic = st.selectbox("FFT/PSD Topic", topics, key="fr_fft_topic")
    with f_c2:
        fields = analyzer.get_topic_numeric_fields(topic)
        field = st.selectbox("Signal Field", fields, key="fr_fft_field") if fields else None
    with f_c3:
        st.caption("使用当前时间窗口")

    if not field:
        st.info("该 topic 无可用数值字段")
        return

    fft_df = analyzer.compute_fft(topic, field, t0=t0, t1=t1)
    psd_df = analyzer.compute_psd(topic, field, t0=t0, t1=t1)

    c1, c2 = st.columns(2)
    with c1:
        if fft_df.empty:
            st.info("FFT 数据不足")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fft_df["freq_hz"], y=fft_df["amplitude"], name="FFT amplitude"))
            fig.update_layout(
                height=260,
                margin=dict(l=50, r=20, t=25, b=40),
                xaxis_title="Hz",
                xaxis=dict(showgrid=True, gridcolor="#e8e8e8"),
                yaxis=dict(showgrid=True, gridcolor="#e8e8e8"),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})
    with c2:
        if psd_df.empty:
            st.info("PSD 数据不足")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=psd_df["freq_hz"], y=psd_df["psd"], name="PSD"))
            fig.update_layout(
                height=260,
                margin=dict(l=50, r=20, t=25, b=40),
                xaxis_title="Hz",
                xaxis=dict(showgrid=True, gridcolor="#e8e8e8"),
                yaxis=dict(showgrid=True, gridcolor="#e8e8e8"),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})


def render_flight_review_dashboard_v2(analyzer):
    summary = analyzer.get_flight_summary()
    mode_segments = analyzer.get_mode_segments()

    # 顶部信息区
    st.markdown("### Flight Overview")
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Start", str(analyzer.start_time.strftime("%Y-%m-%d %H:%M:%S")))
    h2.metric("Duration", f"{analyzer.duration:.1f}s")
    h3.metric("Airframe", str(analyzer.airframe))
    h4.metric("System", str(analyzer.sys_name))
    h5.metric("Firmware", str(analyzer.ver_sw))

    # 全局窗口
    c1, c2 = st.columns([4, 2])
    with c1:
        t0, t1 = st.slider(
            "Global Time Window (s)",
            min_value=0.0,
            max_value=float(max(analyzer.duration, 0.1)),
            value=(0.0, float(max(analyzer.duration, 0.1))),
            step=max(float(analyzer.duration) / 800.0, 0.01),
            key="fr_global_time",
        )
    with c2:
        use_downsample = st.checkbox("Use Downsample", value=True, key="fr_downsample")
        show_rangeslider = st.checkbox("Rangeslider", value=True, key="fr_rangeslider")

    # 地图 + 状态卡
    render_map(analyzer)
    _render_status_cards(summary)

    st.markdown("---")
    st.markdown("### Time Series Groups")
    for group in FLIGHT_REVIEW_GROUPS:
        with st.expander(group["title"], expanded=group.get("expanded", False)):
            _plot_group(
                analyzer=analyzer,
                group=group,
                t0=t0,
                t1=t1,
                use_downsample=use_downsample,
                show_rangeslider=show_rangeslider,
                mode_segments=mode_segments,
            )

    st.markdown("### GPS & Magnetic")
    _render_gps_noise_jamming_panel(analyzer, t0, t1, mode_segments)
    _render_thrust_magnetic_panel(analyzer, t0, t1, mode_segments)

    _render_frequency(analyzer, t0, t1)
    _render_events_and_messages(analyzer, t0, t1)
