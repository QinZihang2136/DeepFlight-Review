# modules/ui_components.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import pandas as pd

def render_map(analyzer):
    """绘制地图，支持自动缩放"""
    gps_df = analyzer.get_gps_tracks()
    
    if gps_df is not None:
        st.markdown("### 🗺️ 3D 飞行轨迹")
        
        mid_lat = gps_df['lat_deg'].mean()
        mid_lon = gps_df['lon_deg'].mean()

        # PathLayer 需要每条记录是一个 path 数组，而不是逐点 DataFrame 行
        path_points = gps_df[["lon_deg", "lat_deg", "alt_rel"]].values.tolist()
        path_data = [{"name": "flight_path", "path": path_points}]

        layer = pdk.Layer(
            "PathLayer",
            path_data,
            pickable=True,
            get_path="path",
            get_color=[255, 50, 50],
            width_scale=1,
            width_min_pixels=3,
            get_width=3,
        )

        # 点云兜底，便于确认轨迹数据确实存在
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            gps_df,
            pickable=False,
            get_position=["lon_deg", "lat_deg", "alt_rel"],
            get_radius=1.5,
            radius_min_pixels=1,
            radius_max_pixels=2,
            get_fill_color=[20, 20, 20, 160],
        )

        view_state = pdk.ViewState(
            latitude=mid_lat,
            longitude=mid_lon,
            zoom=16,
            pitch=45, # 倾斜视角看3D
        )

        st.pydeck_chart(
            pdk.Deck(
                # 使用 carto 底图，避免依赖 mapbox token 导致空白地图
                map_provider="carto",
                map_style="light",
                initial_view_state=view_state,
                layers=[layer, scatter_layer],
                tooltip={"text": "T: {timestamp}s"},
            )
        )
    else:
        st.warning("⚠️ 未找到有效的 GPS 轨迹数据 (尝试了 Global 和 GPS Position)")

def render_chart(
    df,
    fields,
    title="Chart",
    height=300,
    y_title=None,
    colors=None,
    x_range=None,
    show_rangeslider=False,
):
    if df is None: return

    fig = go.Figure()
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, field in enumerate(fields):
        col = field[0] if isinstance(field, tuple) else field
        name = field[1] if isinstance(field, tuple) else field
        
        if col in df.columns:
            line_style = dict(color=colors[i % len(colors)], width=1.5)
            # 自动识别 Setpoint 为虚线
            if 'setpoint' in name.lower() or 'sp' in name.lower() or '_d' in col:
                line_style['dash'] = 'dot'
                line_style['width'] = 2
            
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=name, line=line_style))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', rangeslider=dict(visible=show_rangeslider)),
        yaxis=dict(title=y_title, showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(250,250,250,1)' # 淡淡的背景色，像 Flight Review
    )
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})


def render_linked_subplots(df, fields, title="Linked Charts", height=640, x_range=None, show_rangeslider=False):
    """每个字段独立子图，共享时间轴，便于联动观察。"""
    if df is None or not fields:
        return

    valid_fields = [f for f in fields if f in df.columns]
    if not valid_fields:
        return

    fig = make_subplots(
        rows=len(valid_fields),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=valid_fields,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#17becf", "#9467bd"]
    for i, field in enumerate(valid_fields, start=1):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[field],
                name=field,
                line=dict(color=colors[(i - 1) % len(colors)], width=1.4),
                showlegend=True,
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text=field, row=i, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        hovermode="x unified",
        height=max(height, 220 * len(valid_fields)),
        margin=dict(l=0, r=0, t=35, b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        plot_bgcolor="rgba(250,250,250,1)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    if show_rangeslider:
        fig.update_xaxes(rangeslider=dict(visible=True), row=len(valid_fields), col=1)
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})


def render_comparison_chart(
    series_list,
    title="Series Comparison",
    height=420,
    x_range=None,
    show_rangeslider=True,
    normalize_mode="原始",
):
    """
    跨 topic 多曲线对比图。
    series_list: [{"name": str, "x": array-like, "y": array-like}, ...]
    """
    if not series_list:
        return

    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#17becf", "#9467bd", "#8c564b", "#e377c2"]

    for i, s in enumerate(series_list):
        y = pd.Series(s["y"]).astype(float)
        if normalize_mode == "标准化(0-1)":
            y_min, y_max = y.min(), y.max()
            if y_max > y_min:
                y = (y - y_min) / (y_max - y_min)
        elif normalize_mode == "标准化(Z-Score)":
            y_std = y.std()
            if y_std and y_std > 1e-12:
                y = (y - y.mean()) / y_std
        fig.add_trace(
            go.Scatter(
                x=s["x"],
                y=y,
                name=s["name"],
                line=dict(color=colors[i % len(colors)], width=1.5),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=height,
        margin=dict(l=0, r=0, t=35, b=0),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", rangeslider=dict(visible=show_rangeslider)),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(250,250,250,1)",
    )
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})

def render_flight_review_dashboard(analyzer):
    """
    Flight Review 风格仪表盘
    """
    # === Header Metrics ===
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("⏱️ 飞行时长", f"{analyzer.duration:.1f} s")
        c2.metric("🚀 最大速度", f"{analyzer.kpis['max_speed']:.1f} m/s")
        c3.metric("🏔️ 最大高度", f"{analyzer.kpis['max_alt']:.1f} m")
        c4.metric("💻 机型/固件", f"{analyzer.sys_name}")
    
    st.markdown("---")

    # === Map & Info ===
    render_map(analyzer)

    st.markdown("### 📈 核心数据分析")

    # === 1. Altitude Analysis (多源高度对比) ===
    with st.expander("🏔️ 高度数据分析 (Altitude)", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            # 尝试把 GPS 高度、气压高度、融合高度画在一起
            df_gps = analyzer.get_topic_data('vehicle_gps_position')
            df_baro = analyzer.get_topic_data('vehicle_air_data') # 或者是 sensor_baro
            df_local = analyzer.get_topic_data('vehicle_local_position')
            
            fig = go.Figure()
            if df_local is not None and 'altitude' in df_local:
                fig.add_trace(go.Scatter(x=df_local['timestamp'], y=df_local['altitude'], name='Fused (Est)', line=dict(width=2)))
            if df_gps is not None and 'alt' in df_gps:
                # GPS alt 这里的单位可能是 mm，需要判断
                alt_gps = df_gps['alt'] / 1000.0 if df_gps['alt'].mean() > 1000 else df_gps['alt']
                fig.add_trace(go.Scatter(x=df_gps['timestamp'], y=alt_gps, name='GPS Raw', line=dict(width=1, dash='dot')))
            
            fig.update_layout(title="Altitude Comparison (m)", height=300, hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.markdown("""
            **分析指南**:
            - **Fused**: EKF 融合后的高度，主控以此为准。
            - **GPS**: 原始 GPS 高度，如果与 Fused 偏差大，说明 GPS 信号不可靠。
            """)

    # === 2. Attitude & Rates (并排卡片) ===
    c_att, c_rate = st.columns(2)
    with c_att:
        with st.container(border=True):
            st.markdown("#### 📐 姿态 (Attitude)")
            df_att = analyzer.get_topic_data('vehicle_attitude')
            df_att_sp = analyzer.get_topic_data('vehicle_attitude_setpoint')
            
            if df_att is not None:
                # Roll
                render_chart(df_att, [('roll_deg', 'Roll Est')], "Roll Angle", height=200)
                # Pitch
                render_chart(df_att, [('pitch_deg', 'Pitch Est')], "Pitch Angle", height=200, colors=['#ff7f0e'])
    
    with c_rate:
        with st.container(border=True):
            st.markdown("#### 🔄 角速度 (Rates)")
            df_rates = analyzer.get_topic_data('vehicle_angular_velocity')
            if df_rates is not None:
                render_chart(df_rates, [('xyz[0]_deg', 'Roll Rate'), ('xyz[1]_deg', 'Pitch Rate'), ('xyz[2]_deg', 'Yaw Rate')], "Angular Rates (deg/s)", height=440)

    # === 3. Actuators (全量电机) ===
    with st.container(border=True):
        st.markdown("#### ⚙️ 电机/执行器输出 (Actuators)")
        df_act = analyzer.get_topic_data('actuator_outputs')
        if df_act is not None:
            # 动态查找所有 output 通道
            act_cols = sorted([c for c in df_act.columns if 'output' in c])
            render_chart(df_act, act_cols, "Motor Outputs (PWM or Normalized)", height=300)
        else:
            st.info("无电机数据 (actuator_outputs)")

    # === 4. Battery & Power ===
    c_bat, c_vib = st.columns(2)
    with c_bat:
        with st.container(border=True):
            st.markdown("#### 🔋 电池状态")
            df_bat = analyzer.get_topic_data('battery_status')
            if df_bat is not None:
                render_chart(df_bat, ['voltage_v', 'current_a'], "Voltage/Current", height=250)
    
    with c_vib:
        with st.container(border=True):
            st.markdown("#### 〰️ 震动 (Raw Accel)")
            df_imu = analyzer.get_topic_data('sensor_combined')
            if df_imu is not None:
                render_chart(df_imu, [
                    ('accelerometer_m_s2[0]', 'Acc X'),
                    ('accelerometer_m_s2[1]', 'Acc Y')
                ], "Acceleration X/Y (m/s²)", height=250)
