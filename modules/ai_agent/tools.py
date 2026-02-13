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
