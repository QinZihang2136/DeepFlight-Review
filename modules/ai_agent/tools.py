"""
分层工具定义与执行
L1: 摘要层 - 返回精简关键信息
L2: 统计层 - 返回统计特征
L3: 原始层 - 返回原始数据（谨慎使用）
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any


# =============================================================================
# L1 摘要层工具 - 返回精简摘要（< 500 字符）
# =============================================================================

def get_quick_health_check(analyzer) -> Dict[str, Any]:
    """
    快速健康检查 - 返回精简摘要
    这是推荐首先调用的工具，用于快速了解飞行整体状态
    """
    summary = analyzer.get_flight_summary()

    # 判断总体状态
    warnings = []
    flight_ok = True

    # 检查 GPS
    gps = summary.get("gps") or {}
    if gps.get("best_fix_type", 0) < 3:
        warnings.append({"type": "gps", "message": f"GPS fix 等级为 {gps.get('best_fix_type', 'N/A')}，较低"})
        flight_ok = False

    # 检查电池
    battery = summary.get("battery") or {}
    if battery.get("min_remaining") is not None and battery.get("min_remaining") < 0.2:
        warnings.append({"type": "battery", "message": f"电池最低剩余 {battery.get('min_remaining')*100:.0f}%"})
        flight_ok = False

    # 检查 EKF 重置
    ekf = summary.get("ekf") or {}
    reset_count = sum([
        ekf.get("pos_horiz_reset_counter", 0) or 0,
        ekf.get("pos_vert_reset_counter", 0) or 0,
        ekf.get("yaw_reset_counter", 0) or 0,
    ])
    if reset_count > 5:
        warnings.append({"type": "ekf", "message": f"EKF 重置次数较多: {reset_count}"})
        flight_ok = False

    # 检查 Failsafe
    failsafe_events = [x for x in (summary.get("failsafe_changes") or []) if x.get("failsafe")]
    if failsafe_events:
        warnings.append({"type": "failsafe", "message": f"检测到 {len(failsafe_events)} 次 Failsafe 事件"})
        flight_ok = False

    # 提取关键事件
    timeline = analyzer.get_event_timeline(max_events=50)
    events = timeline.get("events", [])[:5]
    key_events = []
    for e in events:
        kind = e.get("kind", "")
        detail = e.get("detail", {})
        if kind == "arming_state":
            key_events.append({"t_s": round(e.get("t_s", 0), 1), "event": f"状态: {detail.get('state_cn', '?')}"})
        elif kind == "mode_change":
            key_events.append({"t_s": round(e.get("t_s", 0), 1), "event": f"模式: {detail.get('mode_cn', '?')}"})
        elif kind == "failsafe":
            key_events.append({"t_s": round(e.get("t_s", 0), 1), "event": "Failsafe 触发"})

    # 提取 KPI
    kpis = getattr(analyzer, "kpis", {})

    return {
        "flight_ok": flight_ok,
        "duration_s": round(analyzer.duration, 1),
        "max_alt_m": round(kpis.get("max_alt", 0), 1),
        "max_speed_mps": round(kpis.get("max_speed", 0), 1),
        "arming_successful": any(e.get("detail", {}).get("value") == 2 for e in events if e.get("kind") == "arming_state"),
        "warnings": warnings[:5],
        "key_events": key_events[:5],
        "sys_name": analyzer.sys_name,
        "ver_sw": analyzer.ver_sw,
    }


def get_subsystem_summary(analyzer, subsystem: str) -> Dict[str, Any]:
    """
    获取子系统状态摘要
    subsystem: gps, battery, ekf, imu, actuators, rc, position
    """
    subsystem = subsystem.lower().strip()

    if subsystem == "gps":
        return _get_gps_summary(analyzer)
    elif subsystem == "battery":
        return _get_battery_summary(analyzer)
    elif subsystem == "ekf":
        return _get_ekf_summary(analyzer)
    elif subsystem == "imu":
        return _get_imu_summary(analyzer)
    elif subsystem == "actuators":
        return _get_actuators_summary(analyzer)
    elif subsystem == "position":
        return _get_position_summary(analyzer)
    elif subsystem == "rc":
        return _get_rc_summary(analyzer)
    else:
        return {"error": f"未知子系统: {subsystem}", "available": ["gps", "battery", "ekf", "imu", "actuators", "position", "rc"]}


def _get_gps_summary(analyzer) -> Dict[str, Any]:
    """GPS 子系统摘要"""
    df = analyzer.get_topic_data("vehicle_gps_position", downsample=True)

    if df is None or df.empty:
        return {"subsystem": "gps", "status": "error", "message": "无 GPS 数据"}

    status = "ok"
    issues = []

    # Fix type 检查
    if "fix_type" in df.columns:
        min_fix = df["fix_type"].min()
        max_fix = df["fix_type"].max()
        if max_fix < 3:
            status = "error"
            issues.append(f"GPS fix 始终低于 3D (最高 {max_fix})")
        elif min_fix < 3:
            status = "warning"
            issues.append(f"GPS fix 部分时段低于 3D (最低 {min_fix})")

    # EPH/EPV 检查
    eph_ok = True
    if "eph" in df.columns:
        eph_mean = df["eph"].mean()
        if eph_mean > 3.0:
            status = "warning" if status == "ok" else status
            issues.append(f"水平精度较差 (EPH 均值 {eph_mean:.1f}m)")
            eph_ok = False

    return {
        "subsystem": "gps",
        "status": status,
        "metrics": {
            "fix_type_range": [int(df["fix_type"].min()), int(df["fix_type"].max())] if "fix_type" in df.columns else None,
            "eph_mean": round(float(df["eph"].mean()), 2) if "eph" in df.columns else None,
            "epv_mean": round(float(df["epv"].mean()), 2) if "epv" in df.columns else None,
            "satellites": int(df["satellites_used"].mean()) if "satellites_used" in df.columns else None,
        },
        "issues": issues,
        "related_params": ["GPS_UBX_DYNMODEL", "GPS_DUMP_COMM"],
    }


def _get_battery_summary(analyzer) -> Dict[str, Any]:
    """电池子系统摘要"""
    df = analyzer.get_topic_data("battery_status", downsample=True)

    if df is None or df.empty:
        return {"subsystem": "battery", "status": "unknown", "message": "无电池数据"}

    status = "ok"
    issues = []

    # 电压检查
    if "voltage_v" in df.columns:
        min_v = df["voltage_v"].min()
        if min_v < 10.5:  # 假设 3S 电池
            status = "error"
            issues.append(f"电压过低: {min_v:.1f}V")
        elif min_v < 11.1:
            status = "warning" if status == "ok" else status
            issues.append(f"电压偏低: {min_v:.1f}V")

    # 剩余电量检查
    if "remaining" in df.columns:
        min_rem = df["remaining"].min()
        if min_rem < 0.1:
            status = "error" if status != "error" else status
            issues.append(f"电量极低: {min_rem*100:.0f}%")

    return {
        "subsystem": "battery",
        "status": status,
        "metrics": {
            "voltage_range": [round(float(df["voltage_v"].min()), 2), round(float(df["voltage_v"].max()), 2)] if "voltage_v" in df.columns else None,
            "current_max": round(float(df["current_a"].max()), 2) if "current_a" in df.columns else None,
            "remaining_min": round(float(df["remaining"].min()), 2) if "remaining" in df.columns else None,
        },
        "issues": issues,
        "related_params": ["BAT1_V_EMPTY", "BAT1_V_CHARGED", "BAT1_N_CELLS"],
    }


def _get_ekf_summary(analyzer) -> Dict[str, Any]:
    """EKF 子系统摘要"""
    summary = analyzer.get_flight_summary()
    ekf = summary.get("ekf") or {}

    status = "ok"
    issues = []

    # 检查重置计数
    pos_h = ekf.get("pos_horiz_reset_counter", 0) or 0
    pos_v = ekf.get("pos_vert_reset_counter", 0) or 0
    yaw_r = ekf.get("yaw_reset_counter", 0) or 0

    if pos_h > 3 or pos_v > 3:
        status = "warning"
        issues.append(f"位置重置较多: 水平{pos_h}次, 垂直{pos_v}次")
    if yaw_r > 3:
        status = "warning" if status == "ok" else status
        issues.append(f"航向重置较多: {yaw_r}次")

    # 检查故障标志
    if ekf.get("any_filter_fault_flags_nonzero"):
        status = "error"
        issues.append("检测到 EKF 故障标志")

    return {
        "subsystem": "ekf",
        "status": status,
        "metrics": {
            "pos_horiz_reset": pos_h,
            "pos_vert_reset": pos_v,
            "yaw_reset": yaw_r,
            "has_fault": ekf.get("any_filter_fault_flags_nonzero", False),
        },
        "issues": issues,
        "related_params": ["EKF2_MAG_TYPE", "EKF2_GPS_CHECK", "EKF2_EV_DELAY"],
    }


def _get_imu_summary(analyzer) -> Dict[str, Any]:
    """IMU 子系统摘要"""
    df = analyzer.get_topic_data("sensor_combined", downsample=True)

    if df is None or df.empty:
        return {"subsystem": "imu", "status": "unknown", "message": "无 IMU 数据"}

    status = "ok"
    issues = []

    # 检查加速度计震动
    if "accelerometer_m_s2[2]" in df.columns:
        acc_z = df["accelerometer_m_s2[2]"]
        # 震动指标: 高频分量的标准差
        acc_std = acc_z.std()
        if acc_std > 2.0:
            status = "warning"
            issues.append(f"Z轴震动较大 (std={acc_std:.2f} m/s²)")
        elif acc_std > 4.0:
            status = "error"
            issues.append(f"Z轴震动严重 (std={acc_std:.2f} m/s²)")

    return {
        "subsystem": "imu",
        "status": status,
        "metrics": {
            "accel_z_std": round(float(df["accelerometer_m_s2[2]"].std()), 3) if "accelerometer_m_s2[2]" in df.columns else None,
            "gyro_x_range": [round(float(df["gyro_rad[0]"].min()), 3), round(float(df["gyro_rad[0]"].max()), 3)] if "gyro_rad[0]" in df.columns else None,
        },
        "issues": issues,
        "related_params": ["IMU_GYRO_CUTOFF", "IMU_ACCEL_CUTOFF", "IMU_DGYRO_CUTOFF"],
    }


def _get_actuators_summary(analyzer) -> Dict[str, Any]:
    """执行器摘要"""
    df = analyzer.get_topic_data("actuator_outputs", downsample=True)
    if df is None:
        df = analyzer.get_topic_data("actuator_motors", downsample=True)

    if df is None or df.empty:
        return {"subsystem": "actuators", "status": "unknown", "message": "无执行器数据"}

    # 找到所有输出通道
    output_cols = [c for c in df.columns if "output" in c or "control" in c]

    status = "ok"
    issues = []

    # 检查是否有电机输出饱和
    for col in output_cols[:8]:
        if df[col].max() > 0.95:
            status = "warning"
            issues.append(f"{col} 接近饱和 ({df[col].max():.2f})")

    return {
        "subsystem": "actuators",
        "status": status,
        "metrics": {
            "motor_count": len(output_cols),
            "output_range": [round(float(df[output_cols].min().min()), 3), round(float(df[output_cols].max().max()), 3)] if output_cols else None,
        },
        "issues": issues[:5],
        "related_params": ["MC_ROLL_P", "MC_PITCH_P", "MC_YAW_P"],
    }


def _get_position_summary(analyzer) -> Dict[str, Any]:
    """位置控制摘要"""
    df = analyzer.get_topic_data("vehicle_local_position", downsample=True)

    if df is None or df.empty:
        return {"subsystem": "position", "status": "unknown", "message": "无位置数据"}

    status = "ok"
    issues = []

    # 检查高度异常
    if "z" in df.columns:
        z_range = df["z"].max() - df["z"].min()
        if z_range > 100:
            issues.append(f"高度变化大: {z_range:.1f}m")

    # 检查速度
    if "vx" in df.columns and "vy" in df.columns:
        speed = np.sqrt(df["vx"]**2 + df["vy"]**2)
        max_speed = speed.max()
        if max_speed > 20:
            status = "warning"
            issues.append(f"水平速度较高: {max_speed:.1f} m/s")

    return {
        "subsystem": "position",
        "status": status,
        "metrics": {
            "alt_range_m": round(float(-df["z"].min()), 1) if "z" in df.columns else None,
            "max_speed_mps": round(float(np.sqrt((df["vx"]**2 + df["vy"]**2).max())), 1) if "vx" in df.columns and "vy" in df.columns else None,
        },
        "issues": issues,
        "related_params": ["MPC_XY_VEL_MAX", "MPC_Z_VEL_MAX_UP", "MPC_Z_VEL_MAX_DN"],
    }


def _get_rc_summary(analyzer) -> Dict[str, Any]:
    """RC 遥控器摘要"""
    df = analyzer.get_topic_data("manual_control_setpoint", downsample=True)
    if df is None:
        df = analyzer.get_topic_data("input_rc", downsample=True)

    if df is None or df.empty:
        return {"subsystem": "rc", "status": "unknown", "message": "无 RC 数据"}

    return {
        "subsystem": "rc",
        "status": "ok",
        "metrics": {
            "has_data": True,
            "channels": len([c for c in df.columns if "channel" in c.lower() or c in ["x", "y", "z", "r"]]),
        },
        "issues": [],
        "related_params": ["RC_MAP_ROLL", "RC_MAP_PITCH", "RC_MAP_YAW", "RC_MAP_THROTTLE"],
    }


def get_signal_stats(analyzer, topic: str, field: str, time_range: List[float] = None) -> Dict[str, Any]:
    """
    获取信号的统计特征（L2 统计层）
    返回精简的统计信息，而不是原始数据
    """
    df = analyzer.get_topic_data(topic, downsample=False)
    if df is None or field not in df.columns:
        return {"error": f"Topic 或字段不存在: {topic}/{field}"}

    # 时间范围过滤
    if time_range and len(time_range) == 2:
        df = df[(df["timestamp"] >= time_range[0]) & (df["timestamp"] <= time_range[1])]

    if df.empty:
        return {"error": "时间范围内无数据"}

    data = df[field].dropna()
    if data.empty:
        return {"error": "无有效数据"}

    # 计算统计特征
    stats = {
        "topic": topic,
        "field": field,
        "count": len(data),
        "time_range": [round(float(df["timestamp"].min()), 2), round(float(df["timestamp"].max()), 2)],
        "mean": round(float(data.mean()), 4),
        "std": round(float(data.std()), 4),
        "min": round(float(data.min()), 4),
        "max": round(float(data.max()), 4),
        "range": round(float(data.max() - data.min()), 4),
        "median": round(float(data.median()), 4),
        "p5": round(float(data.quantile(0.05)), 4),
        "p95": round(float(data.quantile(0.95)), 4),
    }

    # 检测异常值数量
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    outliers = ((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).sum()
    stats["outliers_count"] = int(outliers)
    stats["outliers_pct"] = round(float(outliers / len(data) * 100), 2)

    return stats


# =============================================================================
# 工具规格定义（供 OpenAI Function Calling 使用）
# =============================================================================

def build_tool_specs() -> List[Dict]:
    """构建工具规格列表"""
    return [
        # === L1 摘要层工具（推荐首先使用）===
        {
            "type": "function",
            "function": {
                "name": "get_quick_health_check",
                "description": "【推荐首先调用】快速获取飞行健康状态摘要。返回精简的整体状态评估，包括关键指标、警告和事件。",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_subsystem_summary",
                "description": "获取指定子系统的状态摘要。可用子系统: gps, battery, ekf, imu, actuators, position, rc",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subsystem": {
                            "type": "string",
                            "enum": ["gps", "battery", "ekf", "imu", "actuators", "position", "rc"],
                            "description": "要检查的子系统名称",
                        }
                    },
                    "required": ["subsystem"],
                },
            },
        },

        # === L2 统计层工具 ===
        {
            "type": "function",
            "function": {
                "name": "get_signal_stats",
                "description": "获取指定信号的统计特征（均值、标准差、范围等），比原始数据更精简。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic 名称"},
                        "field": {"type": "string", "description": "字段名称"},
                        "time_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "可选的时间范围 [start, end]",
                        },
                    },
                    "required": ["topic", "field"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_event_timeline",
                "description": "获取关键事件时间线（arming/mode/failsafe/EKF reset）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_events": {"type": "integer", "minimum": 10, "maximum": 100, "default": 30}
                    },
                },
            },
        },

        # === L3 原始层工具（谨慎使用）===
        {
            "type": "function",
            "function": {
                "name": "get_topic_fields",
                "description": "获取 topic 的可用字段列表。",
                "parameters": {
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_topics",
                "description": "列出可用的 topic 名称。",
                "parameters": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "minimum": 10, "maximum": 100, "default": 50}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_parameters",
                "description": "搜索参数配置（支持前缀和关键字）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prefix": {"type": "string", "description": "参数前缀"},
                        "keyword": {"type": "string", "description": "搜索关键字"},
                        "max_results": {"type": "integer", "minimum": 10, "maximum": 100, "default": 30},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "detect_anomalies",
                "description": "检测指定信号的异常值。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "field": {"type": "string"},
                        "threshold_std": {"type": "number", "default": 3.0},
                    },
                    "required": ["topic", "field"],
                },
            },
        },

        # === L4 频谱分析工具 ===
        {
            "type": "function",
            "function": {
                "name": "get_signal_raw",
                "description": "获取指定时间段的原始时序数据（降采样版本）。返回 timestamp + value 数组，用于时域分析。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic 名称，如 sensor_combined"},
                        "field": {"type": "string", "description": "字段名称，如 accelerometer_m_s2[2]"},
                        "time_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "时间范围 [start_s, end_s]，可选",
                        },
                        "max_samples": {
                            "type": "integer",
                            "default": 500,
                            "description": "最大返回样本数（自动降采样）",
                        },
                    },
                    "required": ["topic", "field"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compute_psd",
                "description": "计算功率谱密度（PSD）。返回频率轴和功率密度，用于振动分析、噪声识别。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic 名称"},
                        "field": {"type": "string", "description": "字段名称"},
                        "time_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "时间范围 [start_s, end_s]，可选",
                        },
                    },
                    "required": ["topic", "field"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compute_fft",
                "description": "计算快速傅里叶变换（FFT）。返回频率轴和幅度，用于识别主频和谐波。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic 名称"},
                        "field": {"type": "string", "description": "字段名称"},
                        "time_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "时间范围 [start_s, end_s]，可选",
                        },
                    },
                    "required": ["topic", "field"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_signal_segments",
                "description": "对比多个时间段的信号特征。返回各段的时域统计和频域特征对比，用于分析不同飞行阶段的振动/性能差异。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic 名称"},
                        "field": {"type": "string", "description": "字段名称"},
                        "segments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "number", "description": "开始时间（秒）"},
                                    "end": {"type": "number", "description": "结束时间（秒）"},
                                    "label": {"type": "string", "description": "段标签，如 '悬停'、'高速飞行'"},
                                },
                                "required": ["start", "end"],
                            },
                            "description": "要对比的时间段列表",
                        },
                    },
                    "required": ["topic", "field", "segments"],
                },
            },
        },

        # === L5 图表理解工具 ===
        {
            "type": "function",
            "function": {
                "name": "get_available_charts",
                "description": "获取当前日志可用的图表列表及其含义。返回每个图表的名称、包含的信号、用途说明，帮助理解可以分析哪些数据。",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },

        # === L6 高级分析工具 ===
        {
            "type": "function",
            "function": {
                "name": "compute_spectrogram",
                "description": "【推荐用于振动分析】计算时频谱图（Spectrogram），与绘图区域的 'Acceleration/Angular Velocity Power Spectral Density' 完全一致。返回频率随时间变化的功率分布，自动使用三轴求和（Flight Review 标准）。用于分析不同飞行阶段的振动频率变化。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic 名称，如 sensor_combined"},
                        "field": {"type": "string", "description": "字段名称，如 accelerometer_m_s2[2]（会自动扩展为三轴求和）"},
                        "time_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "时间范围 [start_s, end_s]，可选",
                        },
                    },
                    "required": ["topic", "field"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_correlation_analysis",
                "description": "分析多个信号之间的相关性。用于分析速度与振动、姿态与振动等的关联关系。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "signals": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "topic": {"type": "string"},
                                    "field": {"type": "string"},
                                    "label": {"type": "string"},
                                },
                                "required": ["topic", "field"],
                            },
                            "description": "要分析的信号列表（最多3个）",
                        },
                        "time_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "时间范围 [start_s, end_s]，可选",
                        },
                    },
                    "required": ["signals"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_flight_phases",
                "description": "自动检测飞行阶段。返回起飞、悬停、前飞、降落等阶段的时间区间和统计信息。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "时间范围 [start_s, end_s]，可选",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_control_performance",
                "description": "分析控制回路性能。计算 setpoint（目标值）与 actual（实际值）之间的跟踪误差，评估姿态控制精度。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "axis": {
                            "type": "string",
                            "enum": ["roll", "pitch", "yaw"],
                            "description": "要分析的控制轴",
                        },
                        "time_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "时间范围 [start_s, end_s]，可选",
                        },
                    },
                    "required": ["axis"],
                },
            },
        },
    ]


# =============================================================================
# 工具执行器
# =============================================================================

def execute_tool(analyzer, name: str, args: Dict) -> Dict:
    """执行工具调用"""
    args = args or {}

    # L1 摘要层
    if name == "get_quick_health_check":
        return get_quick_health_check(analyzer)

    if name == "get_subsystem_summary":
        return get_subsystem_summary(analyzer, args.get("subsystem", "gps"))

    # L2 统计层
    if name == "get_signal_stats":
        return get_signal_stats(
            analyzer,
            args.get("topic"),
            args.get("field"),
            args.get("time_range")
        )

    if name == "get_event_timeline":
        max_events = int(args.get("max_events", 30))
        timeline = analyzer.get_event_timeline(max_events=min(max_events, 100))
        # 压缩返回
        events = timeline.get("events", [])
        return {
            "count": len(events),
            "events": [
                {"t_s": round(e.get("t_s", 0), 1), "kind": e.get("kind"), "summary": _event_summary(e)}
                for e in events[:50]
            ]
        }

    # L3 原始层
    if name == "get_topic_fields":
        topic = args.get("topic")
        if not topic:
            return {"error": "缺少 topic 参数"}
        df = analyzer.get_topic_data(topic, downsample=True)
        if df is None:
            return {"error": f"topic 不存在: {topic}"}
        fields = list(df.columns)
        # 区分数值字段和其他字段
        numeric = [f for f in fields if df[f].dtype.kind in "iufb"]
        return {"topic": topic, "fields": fields, "numeric_fields": numeric}

    if name == "list_topics":
        limit = int(args.get("limit", 50))
        topics = analyzer.get_available_topics()[:limit]
        return {"topics": topics, "total": len(analyzer.get_available_topics())}

    if name == "search_parameters":
        params = analyzer.list_parameters(
            prefix=args.get("prefix") or None,
            keyword=args.get("keyword") or None,
            max_results=int(args.get("max_results", 30)),
        )
        return {"parameters": params, "count": len(params)}

    if name == "detect_anomalies":
        result = analyzer.detect_anomalies(
            args.get("topic"),
            args.get("field"),
            threshold_std=float(args.get("threshold_std", 3.0))
        )
        # 压缩返回，只保留关键信息
        if "error" not in result:
            result["anomalies"] = result.get("anomalies", [])[:10]
        return result

    # L4 频谱分析层
    if name == "get_signal_raw":
        return _get_signal_raw(
            analyzer,
            args.get("topic"),
            args.get("field"),
            args.get("time_range"),
            int(args.get("max_samples", 500))
        )

    if name == "compute_psd":
        return _compute_psd_tool(
            analyzer,
            args.get("topic"),
            args.get("field"),
            args.get("time_range")
        )

    if name == "compute_fft":
        return _compute_fft_tool(
            analyzer,
            args.get("topic"),
            args.get("field"),
            args.get("time_range")
        )

    if name == "compare_signal_segments":
        return _compare_signal_segments(
            analyzer,
            args.get("topic"),
            args.get("field"),
            args.get("segments", [])
        )

    # L5 图表理解层
    if name == "get_available_charts":
        return _get_available_charts(analyzer)

    # L6 高级分析层
    if name == "compute_spectrogram":
        return _compute_spectrogram_tool(
            analyzer,
            args.get("topic"),
            args.get("field"),
            args.get("time_range")
        )

    if name == "get_correlation_analysis":
        return _get_correlation_analysis(
            analyzer,
            args.get("signals", []),
            args.get("time_range")
        )

    if name == "get_flight_phases":
        return _get_flight_phases(analyzer, args.get("time_range"))

    if name == "get_control_performance":
        return _get_control_performance(
            analyzer,
            args.get("axis"),
            args.get("time_range")
        )

    return {"error": f"未知工具: {name}"}


def _event_summary(event: Dict) -> str:
    """生成事件的简短摘要"""
    kind = event.get("kind", "")
    detail = event.get("detail", {})

    if kind == "arming_state":
        return detail.get("state_cn", detail.get("state_en", "?"))
    elif kind == "mode_change":
        return detail.get("mode_cn", detail.get("mode_en", "?"))
    elif kind == "failsafe":
        return "Failsafe" + ("触发" if detail.get("failsafe") else "解除")
    elif kind == "ekf_reset":
        return f"EKF重置: {detail.get('counter', '?')}"
    else:
        return kind


# =============================================================================
# L4 频谱分析辅助函数
# =============================================================================

def _get_signal_raw(analyzer, topic: str, field: str, time_range: List[float] = None, max_samples: int = 500) -> Dict:
    """获取原始信号数据（降采样版本）"""
    t0, t1 = None, None
    if time_range and len(time_range) >= 2:
        t0, t1 = time_range[0], time_range[1]

    df = analyzer.get_signal(topic, field, t0=t0, t1=t1)
    if df is None or df.empty:
        return {"error": f"无法获取数据: {topic}/{field}"}

    # 降采样
    if len(df) > max_samples:
        step = len(df) // max_samples
        df = df.iloc[::step].copy()

    # 返回精简格式
    timestamps = df["timestamp"].tolist()
    values = df["value"].tolist()

    return {
        "topic": topic,
        "field": field,
        "samples": len(timestamps),
        "time_range": [round(float(timestamps[0]), 2), round(float(timestamps[-1]), 2)] if timestamps else None,
        "data": {
            "timestamp": [round(t, 3) for t in timestamps],
            "value": [round(v, 4) if isinstance(v, float) else v for v in values]
        }
    }


def _compute_psd_tool(analyzer, topic: str, field: str, time_range: List[float] = None) -> Dict:
    """计算功率谱密度"""
    t0, t1 = None, None
    if time_range and len(time_range) >= 2:
        t0, t1 = time_range[0], time_range[1]

    df = analyzer.compute_psd(topic, field, t0=t0, t1=t1)
    if df is None or df.empty:
        return {"error": f"无法计算 PSD: {topic}/{field}"}

    # 找到峰值频率
    freqs = df["freq_hz"].tolist()
    psd = df["psd"].tolist()

    # 找前5个峰值
    peak_indices = np.argsort(psd)[-5:][::-1]
    peaks = [
        {"freq_hz": round(freqs[i], 2), "psd": round(psd[i], 6)}
        for i in peak_indices if i < len(freqs) and freqs[i] > 0
    ][:3]

    # 降采样返回（最多100个点）
    step = max(1, len(freqs) // 100)
    freqs_sampled = freqs[::step]
    psd_sampled = psd[::step]

    return {
        "topic": topic,
        "field": field,
        "freq_range_hz": [round(min(freqs), 2), round(max(freqs), 2)] if freqs else None,
        "peak_frequencies": peaks,
        "data": {
            "freq_hz": [round(f, 2) for f in freqs_sampled],
            "psd": [round(p, 6) for p in psd_sampled]
        }
    }


def _compute_fft_tool(analyzer, topic: str, field: str, time_range: List[float] = None) -> Dict:
    """计算 FFT 频谱"""
    t0, t1 = None, None
    if time_range and len(time_range) >= 2:
        t0, t1 = time_range[0], time_range[1]

    df = analyzer.compute_fft(topic, field, t0=t0, t1=t1)
    if df is None or df.empty:
        return {"error": f"无法计算 FFT: {topic}/{field}"}

    freqs = df["freq_hz"].tolist()
    amp = df["amplitude"].tolist()

    # 找前5个峰值（忽略直流分量）
    valid_indices = [i for i, f in enumerate(freqs) if f > 0.5]
    if valid_indices:
        peak_indices = sorted(valid_indices, key=lambda i: amp[i], reverse=True)[:5]
        peaks = [
            {"freq_hz": round(freqs[i], 2), "amplitude": round(amp[i], 4)}
            for i in peak_indices
        ]
    else:
        peaks = []

    # 降采样返回（最多100个点）
    step = max(1, len(freqs) // 100)
    freqs_sampled = freqs[::step]
    amp_sampled = amp[::step]

    return {
        "topic": topic,
        "field": field,
        "freq_range_hz": [round(min(freqs), 2), round(max(freqs), 2)] if freqs else None,
        "peak_frequencies": peaks,
        "data": {
            "freq_hz": [round(f, 2) for f in freqs_sampled],
            "amplitude": [round(a, 4) for a in amp_sampled]
        }
    }


def _compare_signal_segments(analyzer, topic: str, field: str, segments: List[Dict]) -> Dict:
    """对比多个时间段的信号特征"""
    if not segments:
        return {"error": "segments 参数为空"}

    results = []

    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")
        label = seg.get("label", f"{start}s-{end}s")

        if start is None or end is None:
            continue

        segment_result = {"label": label, "time_range": [start, end]}

        # 获取时域统计
        df = analyzer.get_signal(topic, field, t0=start, t1=end)
        if df is not None and not df.empty:
            values = df["value"].dropna()
            if len(values) > 0:
                segment_result["time_domain"] = {
                    "samples": len(values),
                    "mean": round(float(values.mean()), 4),
                    "std": round(float(values.std()), 4),
                    "min": round(float(values.min()), 4),
                    "max": round(float(values.max()), 4),
                    "range": round(float(values.max() - values.min()), 4),
                }

        # 获取频域特征（FFT 峰值）
        fft_df = analyzer.compute_fft(topic, field, t0=start, t1=end)
        if fft_df is not None and not fft_df.empty:
            freqs = fft_df["freq_hz"].tolist()
            amps = fft_df["amplitude"].tolist()

            # 找主频（忽略直流）
            valid_indices = [i for i, f in enumerate(freqs) if f > 0.5]
            if valid_indices:
                peak_idx = max(valid_indices, key=lambda i: amps[i])
                segment_result["frequency_domain"] = {
                    "dominant_freq_hz": round(freqs[peak_idx], 2),
                    "peak_amplitude": round(amps[peak_idx], 4),
                }

        results.append(segment_result)

    return {
        "topic": topic,
        "field": field,
        "segments_compared": len(results),
        "comparison": results
    }


def _get_available_charts(analyzer) -> Dict:
    """获取可用的图表列表及其含义"""
    # 图表组定义（与 flight_review_layout.py 同步）
    chart_definitions = [
        {
            "key": "attitude",
            "title": "姿态角 (Attitude)",
            "description": "显示飞行器的横滚、俯仰、偏航角度及其目标值，用于评估姿态控制性能",
            "signals": ["Roll (横滚)", "Pitch (俯仰)", "Yaw (偏航)"],
            "usage": "检查姿态跟踪精度、震荡、超调等问题",
            "topic": "vehicle_attitude",
        },
        {
            "key": "rates",
            "title": "角速度 (Angular Rates)",
            "description": "显示飞行器的三轴角速度及其目标值，是姿态控制的核心指标",
            "signals": ["Roll Rate (横滚速率)", "Pitch Rate (俯仰速率)", "Yaw Rate (偏航速率)"],
            "usage": "检查控制响应速度、阻尼特性、振动问题",
            "topic": "vehicle_angular_velocity",
        },
        {
            "key": "position_velocity",
            "title": "位置与速度 (Position & Velocity)",
            "description": "显示飞行器的本地位置和速度，用于评估位置控制性能",
            "signals": ["x, y (水平位置)", "altitude (高度)", "vx, vy, vz (三轴速度)"],
            "usage": "检查位置保持精度、飞行轨迹、速度控制",
            "topic": "vehicle_local_position",
        },
        {
            "key": "actuators",
            "title": "执行器 (Actuators)",
            "description": "显示电机/舵机的输出值，反映控制指令的执行情况",
            "signals": ["Motor 1-8 outputs"],
            "usage": "检查电机饱和、不平衡、控制分配",
            "topic": "actuator_outputs / actuator_motors",
        },
        {
            "key": "power",
            "title": "电源 (Power)",
            "description": "显示电池电压、电流和剩余电量",
            "signals": ["voltage_v (电压)", "current_a (电流)", "remaining (剩余电量)"],
            "usage": "监控电池状态、检测电压跌落、评估续航",
            "topic": "battery_status",
        },
        {
            "key": "gps",
            "title": "GPS 定位",
            "description": "显示 GPS 定位质量指标",
            "signals": ["fix_type (定位类型)", "eph (水平精度)", "epv (垂直精度)"],
            "usage": "检查 GPS 信号质量、定位精度",
            "topic": "vehicle_gps_position",
        },
        {
            "key": "ekf",
            "title": "状态估计 (EKF)",
            "description": "显示 EKF 滤波器状态，反映位置估计的可靠性",
            "signals": ["reset counters (重置计数)", "health flags (健康标志)"],
            "usage": "检测 EKF 问题、位置跳变、估计发散",
            "topic": "estimator_status / ekf2_estimator_status",
        },
        {
            "key": "sensor_combined",
            "title": "IMU 传感器",
            "description": "显示加速度计和陀螺仪原始数据",
            "signals": ["accelerometer (加速度)", "gyro (角速度)"],
            "usage": "检查 IMU 数据质量、振动水平、传感器故障",
            "topic": "sensor_combined",
        },
    ]

    # 检查每个图表的数据可用性
    available_topics = set(analyzer.get_available_topics())
    charts = []

    for chart in chart_definitions:
        # 检查 topic 是否可用
        topic_available = any(
            t in available_topics for t in chart["topic"].split(" / ")
        )
        chart["available"] = topic_available
        charts.append(chart)

    return {
        "total_charts": len(charts),
        "available_charts": sum(1 for c in charts if c["available"]),
        "charts": charts,
        "analysis_tools": [
            {
                "tool": "get_signal_raw",
                "description": "获取原始时序数据进行分析",
            },
            {
                "tool": "compute_fft",
                "description": "计算 FFT 频谱，识别主频和谐波",
            },
            {
                "tool": "compute_psd",
                "description": "计算功率谱密度，分析振动能量分布",
            },
            {
                "tool": "compare_signal_segments",
                "description": "对比不同飞行阶段的信号特征",
            },
        ],
    }


# =============================================================================
# L6 高级分析辅助函数
# =============================================================================

def _compute_spectrogram_tool(analyzer, topic: str, field: str, time_range: List[float] = None) -> Dict:
    """计算时频谱图（与绘图区域一致的标准）

    注意：绘图区域的 "Acceleration/Angular Velocity Power Spectral Density"
    实际上是时频谱（Spectrogram），显示的是频率随时间变化的功率分布。
    Flight Review 标准是对三轴数据求和后再计算。
    """
    t0, t1 = None, None
    if time_range and len(time_range) >= 2:
        t0, t1 = time_range[0], time_range[1]

    # 判断是否是加速度或角速度，使用三轴求和（Flight Review 标准）
    fields = [field]
    is_accel = "accel" in field.lower() or "accelerometer" in field.lower()
    is_gyro = "gyro" in field.lower() or "angular" in topic.lower()

    if is_accel and "[" in field:
        # 加速度：使用三轴求和
        fields = ["accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"]
    elif is_gyro and "[" in field:
        # 角速度：使用三轴求和
        fields = ["gyro_rad[0]", "gyro_rad[1]", "gyro_rad[2]"]

    df = analyzer.compute_spectrogram(topic, fields, t0=t0, t1=t1)
    if df is None or df.empty:
        return {"error": f"无法计算时频谱: {topic}/{fields}"}

    # 提取关键信息
    time_axis = df["time_s"].tolist()
    freq_axis = df["freq_hz"].tolist()
    power_db = df["power_db"].tolist()

    if not time_axis:
        return {"error": "时频谱数据为空"}

    # 找到主频随时间的变化
    unique_times = sorted(set(time_axis))
    dominant_freq_over_time = []

    for t in unique_times[:100]:  # 最多100个时间点
        mask = [abs(ti - t) < 0.01 for ti in time_axis]
        indices = [i for i, m in enumerate(mask) if m]
        if indices:
            max_idx = max(indices, key=lambda i: power_db[i])
            if freq_axis[max_idx] > 1:  # 忽略直流分量
                dominant_freq_over_time.append({
                    "time_s": round(t, 2),
                    "dominant_freq_hz": round(freq_axis[max_idx], 2),
                    "power_db": round(power_db[max_idx], 2)
                })

    # 计算统计信息
    power_values = [p for p in power_db if p > -100]  # 过滤极低值
    avg_power = np.mean(power_values) if power_values else 0

    # 按时间段分析（分成4段）
    if unique_times:
        t_min, t_max = min(unique_times), max(unique_times)
        t_span = t_max - t_min
        segments = []
        for i in range(4):
            seg_start = t_min + i * t_span / 4
            seg_end = t_min + (i + 1) * t_span / 4
            seg_mask = [seg_start <= t <= seg_end for t in time_axis]
            seg_indices = [i for i, m in enumerate(seg_mask) if m]
            if seg_indices:
                seg_powers = [power_db[i] for i in seg_indices]
                seg_freqs = [freq_axis[i] for i in seg_indices]
                max_idx = seg_powers.index(max(seg_powers))
                segments.append({
                    "time_range": [round(seg_start, 1), round(seg_end, 1)],
                    "peak_freq_hz": round(seg_freqs[max_idx], 1),
                    "peak_power_db": round(seg_powers[max_idx], 1),
                    "avg_power_db": round(np.mean(seg_powers), 1)
                })

    return {
        "topic": topic,
        "fields": fields,
        "time_range": [round(min(time_axis), 2), round(max(time_axis), 2)],
        "freq_range_hz": [round(min(freq_axis), 2), round(max(freq_axis), 2)],
        "power_range_db": [round(min(power_values), 2), round(max(power_values), 2)] if power_values else None,
        "avg_power_db": round(avg_power, 2),
        "dominant_freq_over_time": dominant_freq_over_time[:50],
        "time_segments": segments,
        "note": "此分析使用三轴 PSD 求和（Flight Review 标准），与绘图区域一致"
    }


def _get_correlation_analysis(analyzer, signals: List[Dict], time_range: List[float] = None) -> Dict:
    """分析多个信号之间的相关性"""
    if not signals or len(signals) < 2:
        return {"error": "至少需要 2 个信号进行相关性分析"}

    if len(signals) > 3:
        signals = signals[:3]  # 最多 3 个信号

    t0, t1 = None, None
    if time_range and len(time_range) >= 2:
        t0, t1 = time_range[0], time_range[1]

    # 获取各信号数据
    signal_data = {}
    for sig in signals:
        topic = sig.get("topic")
        field = sig.get("field")
        label = sig.get("label", f"{topic}.{field}")

        df = analyzer.get_signal(topic, field, t0=t0, t1=t1)
        if df is None or df.empty:
            return {"error": f"无法获取信号: {label}"}

        signal_data[label] = df

    # 对齐时间轴（使用交集）
    common_times = None
    for label, df in signal_data.items():
        times = set(round(t, 3) for t in df["timestamp"].tolist())
        if common_times is None:
            common_times = times
        else:
            common_times = common_times & times

    if not common_times or len(common_times) < 10:
        return {"error": "信号时间轴重叠太少，无法进行相关性分析"}

    common_times = sorted(common_times)

    # 提取对齐后的值
    aligned_values = {}
    for label, df in signal_data.items():
        values = []
        time_to_value = {round(t, 3): v for t, v in zip(df["timestamp"], df["value"])}
        for t in common_times:
            values.append(time_to_value.get(t, np.nan))
        aligned_values[label] = values

    # 计算相关系数矩阵
    labels = list(aligned_values.keys())
    n = len(labels)
    correlation_matrix = [[1.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            x = np.array(aligned_values[labels[i]])
            y = np.array(aligned_values[labels[j]])

            # 移除 NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 10:
                corr = np.corrcoef(x[mask], y[mask])[0, 1]
                correlation_matrix[i][j] = round(corr, 4) if not np.isnan(corr) else 0
                correlation_matrix[j][i] = correlation_matrix[i][j]

    # 解释相关性
    interpretations = []
    for i in range(n):
        for j in range(i + 1, n):
            corr = correlation_matrix[i][j]
            if abs(corr) > 0.7:
                strength = "强"
            elif abs(corr) > 0.4:
                strength = "中等"
            else:
                strength = "弱"
            direction = "正" if corr > 0 else "负"
            interpretations.append(
                f"{labels[i]} 与 {labels[j]}: {direction}相关 ({strength}, r={corr:.2f})"
            )

    return {
        "signals": labels,
        "samples": len(common_times),
        "time_range": [round(min(common_times), 2), round(max(common_times), 2)],
        "correlation_matrix": {
            "labels": labels,
            "values": correlation_matrix
        },
        "interpretations": interpretations,
    }


def _get_flight_phases(analyzer, time_range: List[float] = None) -> Dict:
    """自动检测飞行阶段"""
    t0, t1 = None, None
    if time_range and len(time_range) >= 2:
        t0, t1 = time_range[0], time_range[1]

    # 获取事件时间线
    timeline = analyzer.get_event_timeline(max_events=100)
    events = timeline.get("events", [])

    # 获取位置数据用于判断飞行状态
    pos_df = analyzer.get_topic_data("vehicle_local_position", downsample=True)

    phases = []
    phase_stats = {}

    # 基于 arming_state 和 nav_state 划分阶段
    current_phase = {"type": "unknown", "start": 0}

    for event in events:
        t = event.get("t_s", 0)
        kind = event.get("kind", "")
        detail = event.get("detail", {})

        if t0 is not None and t < t0:
            continue
        if t1 is not None and t > t1:
            continue

        if kind == "arming_state":
            state = detail.get("value", 0)
            if state == 2:  # ARMED
                if current_phase["type"] != "armed":
                    if current_phase["type"] != "unknown":
                        phases.append({**current_phase, "end": t})
                    current_phase = {"type": "armed", "start": t}
            elif state == 1:  # STANDBY
                if current_phase["type"] == "armed" or current_phase["type"] == "flying":
                    phases.append({**current_phase, "end": t})
                    current_phase = {"type": "standby", "start": t}

        elif kind == "mode_change":
            mode = detail.get("mode_en", "").lower()
            phase_type = "unknown"
            if "land" in mode:
                phase_type = "landing"
            elif "rtl" in mode:
                phase_type = "rtl"
            elif "loiter" in mode or "hover" in mode:
                phase_type = "hover"
            elif "mission" in mode:
                phase_type = "mission"
            elif "takeoff" in mode:
                phase_type = "takeoff"
            elif "offboard" in mode:
                phase_type = "offboard"
            elif "posctl" in mode or "altctl" in mode:
                phase_type = "manual_flight"

            if phase_type != "unknown" and phase_type != current_phase["type"]:
                if current_phase["type"] != "unknown":
                    phases.append({**current_phase, "end": t})
                current_phase = {"type": phase_type, "start": t}

    # 添加最后一个阶段
    end_time = t1 if t1 is not None else analyzer.duration
    phases.append({**current_phase, "end": end_time})

    # 计算各阶段统计
    for phase in phases:
        ptype = phase["type"]
        duration = phase["end"] - phase["start"]
        if ptype not in phase_stats:
            phase_stats[ptype] = {"count": 0, "total_duration": 0}
        phase_stats[ptype]["count"] += 1
        phase_stats[ptype]["total_duration"] += duration

    # 简化输出
    phase_list = [
        {
            "type": p["type"],
            "start": round(p["start"], 1),
            "end": round(p["end"], 1),
            "duration": round(p["end"] - p["start"], 1)
        }
        for p in phases
        if p["end"] - p["start"] > 0.5  # 过滤太短的阶段
    ]

    return {
        "phases": phase_list,
        "phase_statistics": {
            k: {"count": v["count"], "total_duration": round(v["total_duration"], 1)}
            for k, v in phase_stats.items()
        },
        "flight_duration": round(analyzer.duration, 1)
    }


def _get_control_performance(analyzer, axis: str, time_range: List[float] = None) -> Dict:
    """分析控制回路性能"""
    t0, t1 = None, None
    if time_range and len(time_range) >= 2:
        t0, t1 = time_range[0], time_range[1]

    # 根据轴选择对应的信号
    axis_config = {
        "roll": {
            "actual_topic": "vehicle_attitude",
            "actual_field": "roll_deg",
            "setpoint_topic": "vehicle_attitude_setpoint",
            "setpoint_field": "roll_body_deg",
        },
        "pitch": {
            "actual_topic": "vehicle_attitude",
            "actual_field": "pitch_deg",
            "setpoint_topic": "vehicle_attitude_setpoint",
            "setpoint_field": "pitch_body_deg",
        },
        "yaw": {
            "actual_topic": "vehicle_attitude",
            "actual_field": "yaw_deg",
            "setpoint_topic": "vehicle_attitude_setpoint",
            "setpoint_field": "yaw_body_deg",
        },
    }

    if axis not in axis_config:
        return {"error": f"不支持的轴: {axis}"}

    config = axis_config[axis]

    # 获取实际值和目标值
    actual_df = analyzer.get_signal(config["actual_topic"], config["actual_field"], t0=t0, t1=t1)
    setpoint_df = analyzer.get_signal(config["setpoint_topic"], config["setpoint_field"], t0=t0, t1=t1)

    if actual_df is None or actual_df.empty:
        return {"error": f"无法获取实际值: {config['actual_topic']}.{config['actual_field']}"}

    if setpoint_df is None or setpoint_df.empty:
        return {
            "axis": axis,
            "actual_available": True,
            "setpoint_available": False,
            "actual_stats": {
                "mean": round(float(actual_df["value"].mean()), 4),
                "std": round(float(actual_df["value"].std()), 4),
                "min": round(float(actual_df["value"].min()), 4),
                "max": round(float(actual_df["value"].max()), 4),
            },
            "note": "Setpoint 数据不可用，无法计算跟踪误差"
        }

    # 对齐时间轴计算误差
    # 简化：使用插值对齐
    from scipy import interpolate

    sp_times = setpoint_df["timestamp"].values
    sp_values = setpoint_df["value"].values
    actual_times = actual_df["timestamp"].values
    actual_values = actual_df["value"].values

    # 创建插值函数
    try:
        sp_interp = interpolate.interp1d(sp_times, sp_values, bounds_error=False, fill_value="extrapolate")
        sp_at_actual = sp_interp(actual_times)
        errors = actual_values - sp_at_actual
        errors = errors[~np.isnan(errors)]
    except Exception:
        errors = np.array([])

    if len(errors) < 10:
        return {"error": "对齐后数据点太少"}

    # 计算误差统计
    error_stats = {
        "mean_error": round(float(np.mean(errors)), 4),
        "std_error": round(float(np.std(errors)), 4),
        "max_error": round(float(np.max(np.abs(errors))), 4),
        "rms_error": round(float(np.sqrt(np.mean(errors ** 2))), 4),
    }

    # 评估性能
    if error_stats["rms_error"] < 1:
        performance = "优秀"
    elif error_stats["rms_error"] < 3:
        performance = "良好"
    elif error_stats["rms_error"] < 5:
        performance = "一般"
    else:
        performance = "需改进"

    return {
        "axis": axis,
        "setpoint": {"topic": config["setpoint_topic"], "field": config["setpoint_field"]},
        "actual": {"topic": config["actual_topic"], "field": config["actual_field"]},
        "error_stats": error_stats,
        "tracking_performance": {
            "samples": len(errors),
            "performance": performance,
        },
        "interpretation": f"{axis}轴控制: RMS误差 {error_stats['rms_error']:.2f}°，性能{performance}"
    }
