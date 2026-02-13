"""
预设诊断模板 - 一键诊断功能
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DiagnosticPreset:
    """诊断预设模板"""
    id: str
    name: str
    description: str
    icon: str
    estimated_time: str
    system_prompt_addition: str
    user_prompt: str
    stages: List[str]  # 要执行的阶段 ID
    focus_subsystems: List[str]  # 重点关注的子系统
    output_template: str


# 预设诊断模板定义
DIAGNOSTIC_PRESETS: Dict[str, DiagnosticPreset] = {

    "quick_health": DiagnosticPreset(
        id="quick_health",
        name="快速健康检查",
        description="30秒快速评估飞行状态，适合初步筛查",
        icon="🏥",
        estimated_time="~30秒",
        system_prompt_addition="""
用户请求快速健康检查。请高效完成：
1. 调用 get_quick_health_check 获取整体状态
2. 如有警告，简要说明
3. 给出一句总体评价

回复要简洁，不超过 200 字。
""",
        user_prompt="请执行快速健康检查，告诉我这次飞行是否正常。",
        stages=["preflight"],
        focus_subsystems=[],
        output_template="""
## 🏥 快速健康检查

**状态**: {status_emoji} {status}

**飞行信息**:
- 时长: {duration}
- 最大高度: {max_alt}
- 最大速度: {max_speed}

**{warning_icon} 警告项** ({warning_count}):
{warnings}

**建议**: {recommendation}
"""
    ),

    "full_diagnostic": DiagnosticPreset(
        id="full_diagnostic",
        name="完整诊断报告",
        description="全面分析所有子系统，生成详细报告",
        icon="🔍",
        estimated_time="~2分钟",
        system_prompt_addition="""
用户请求完整诊断报告。请按步骤进行全面分析：

【阶段 1: 预检】
调用 get_quick_health_check 获取整体状态。

【阶段 2: 子系统检查】
依次调用 get_subsystem_summary 检查:
- GPS 状态
- 电池状态
- EKF 状态
- IMU 状态
- 执行器状态

【阶段 3: 事件分析】
调用 get_event_timeline 分析事件序列。

【阶段 4: 信号分析】
对发现的问题信号，调用 get_signal_stats 深入分析。

【阶段 5: 根因诊断】
综合分析问题根因。

【阶段 6: 建议】
生成复飞前检查清单。

每个阶段完成后简要说明发现。
""",
        user_prompt="请执行完整诊断分析，生成详细报告。",
        stages=["preflight", "subsystem_check", "event_analysis", "signal_analysis", "root_cause", "recommendation"],
        focus_subsystems=["gps", "battery", "ekf", "imu", "actuators"],
        output_template="""
# 🔍 完整诊断报告

## 📊 飞行概览
{flight_overview}

## ✅ 系统状态检查
{subsystem_status}

## ⚠️ 发现的问题
{issues_found}

## 🔬 详细分析
{detailed_analysis}

## 💡 改进建议
{recommendations}

---
*诊断完成*
"""
    ),

    "vibration_analysis": DiagnosticPreset(
        id="vibration_analysis",
        name="震动分析",
        description="深度分析震动问题，适合飞控调参",
        icon="📊",
        estimated_time="~1分钟",
        system_prompt_addition="""
用户请求震动问题分析。请专注分析震动相关内容：

1. 调用 get_subsystem_summary("imu") 获取 IMU 状态
2. 调用 get_signal_stats 分析加速度计数据
3. 调用 detect_anomalies 检测异常震动点
4. 分析震动频率特征
5. 给出减震建议

重点关注：
- 加速度计标准差
- 震动频率分布
- 可能的震动源（电机、桨叶、机架共振）
""",
        user_prompt="请分析这次飞行的震动情况，判断是否有震动问题。",
        stages=["preflight", "signal_analysis", "recommendation"],
        focus_subsystems=["imu"],
        output_template="""
## 📊 震动分析报告

**震动状态**: {vib_status}

### 加速度计分析
{accel_analysis}

### 震动特征
{vib_characteristics}

### 可能原因
{possible_causes}

### 减震建议
{damping_recommendations}
"""
    ),

    "gps_investigation": DiagnosticPreset(
        id="gps_investigation",
        name="GPS 问题排查",
        description="排查 GPS 信号和定位问题",
        icon="📡",
        estimated_time="~1分钟",
        system_prompt_addition="""
用户请求 GPS 问题排查。请专注分析 GPS 相关内容：

1. 调用 get_subsystem_summary("gps") 获取 GPS 状态
2. 调用 get_subsystem_summary("ekf") 检查 EKF 状态
3. 分析 fix_type、EPH、EPV 变化
4. 检查卫星数量变化
5. 判断是否存在干扰
6. 给出 GPS 改善建议

重点关注：
- GPS fix 等级是否稳定
- 精度指标 (EPH/EPV)
- EKF 重置与 GPS 问题的关联
""",
        user_prompt="请排查这次飞行的 GPS 问题，分析定位表现。",
        stages=["preflight", "subsystem_check", "event_analysis", "recommendation"],
        focus_subsystems=["gps", "ekf"],
        output_template="""
## 📡 GPS 问题排查报告

**GPS 状态**: {gps_status}

### 信号质量
{signal_quality}

### 精度分析
{accuracy_analysis}

### 问题诊断
{problem_diagnosis}

### 改善建议
{gps_recommendations}
"""
    ),

    "battery_analysis": DiagnosticPreset(
        id="battery_analysis",
        name="电池分析",
        description="分析电池使用和健康状态",
        icon="🔋",
        estimated_time="~30秒",
        system_prompt_addition="""
用户请求电池分析。请专注分析电池相关内容：

1. 调用 get_subsystem_summary("battery") 获取电池状态
2. 分析电压曲线变化
3. 分析电流消耗
4. 计算平均功耗
5. 评估电池健康状态

重点关注：
- 电压压降情况
- 剩余电量变化
- 电流峰值
""",
        user_prompt="请分析这次飞行的电池使用情况。",
        stages=["preflight", "subsystem_check", "recommendation"],
        focus_subsystems=["battery"],
        output_template="""
## 🔋 电池分析报告

**电池状态**: {battery_status}

### 电压分析
{voltage_analysis}

### 电流分析
{current_analysis}

### 健康评估
{health_assessment}

### 使用建议
{battery_recommendations}
"""
    ),

    "ekf_check": DiagnosticPreset(
        id="ekf_check",
        name="EKF 状态检查",
        description="检查状态估计器表现",
        icon="🧭",
        estimated_time="~1分钟",
        system_prompt_addition="""
用户请求 EKF 状态检查。请专注分析 EKF 相关内容：

1. 调用 get_subsystem_summary("ekf") 获取 EKF 状态
2. 检查各类重置计数
3. 分析重置发生的时间点
4. 调用 get_event_timeline 查找相关事件
5. 判断 EKF 表现是否正常

重点关注：
- 位置重置次数
- 航向重置次数
- 故障标志
""",
        user_prompt="请检查这次飞行的 EKF 状态估计表现。",
        stages=["preflight", "subsystem_check", "event_analysis", "recommendation"],
        focus_subsystems=["ekf", "gps"],
        output_template="""
## 🧭 EKF 状态检查报告

**EKF 状态**: {ekf_status}

### 重置统计
{reset_statistics}

### 故障检查
{fault_check}

### 相关事件
{related_events}

### 参数建议
{ekf_recommendations}
"""
    ),

    "control_tuning": DiagnosticPreset(
        id="control_tuning",
        name="控制调参分析",
        description="分析控制回路表现，辅助 PID 调参",
        icon="🎛️",
        estimated_time="~2分钟",
        system_prompt_addition="""
用户请求控制调参分析。请专注分析控制相关内容：

1. 调用 get_subsystem_summary("actuators") 检查执行器状态
2. 调用 get_signal_stats 分析角速度跟踪
3. 检查是否有振荡或超调
4. 分析电机输出饱和情况
5. 给出调参建议

重点关注：
- 角速度跟踪误差
- 电机输出范围
- 是否有持续振荡
""",
        user_prompt="请分析这次飞行的控制表现，帮助我判断是否需要调参。",
        stages=["preflight", "subsystem_check", "signal_analysis", "recommendation"],
        focus_subsystems=["imu", "actuators"],
        output_template="""
## 🎛️ 控制调参分析

**控制状态**: {control_status}

### 跟踪性能
{tracking_performance}

### 振荡检测
{oscillation_check}

### 执行器分析
{actuator_analysis}

### 调参建议
{tuning_recommendations}
"""
    ),
}


def get_preset_names() -> List[Dict[str, str]]:
    """获取所有预设的名称和描述"""
    return [
        {
            "id": preset.id,
            "name": preset.name,
            "description": preset.description,
            "icon": preset.icon,
            "estimated_time": preset.estimated_time,
        }
        for preset in DIAGNOSTIC_PRESETS.values()
    ]


def get_preset(preset_id: str) -> Optional[DiagnosticPreset]:
    """获取指定的预设"""
    return DIAGNOSTIC_PRESETS.get(preset_id)


def get_preset_prompt(preset_id: str) -> str:
    """获取预设的完整 prompt"""
    preset = DIAGNOSTIC_PRESETS.get(preset_id)
    if not preset:
        return ""

    return f"{preset.system_prompt_addition}\n\n{preset.user_prompt}"


def list_presets_for_display() -> str:
    """生成用于显示的预设列表"""
    lines = ["可用的一键诊断模板：\n"]
    for preset in DIAGNOSTIC_PRESETS.values():
        lines.append(f"{preset.icon} **{preset.name}** ({preset.estimated_time})")
        lines.append(f"   {preset.description}")
        lines.append("")
    return "\n".join(lines)


# 斜杠命令映射
SLASH_COMMANDS = {
    "/quick": "quick_health",
    "/full": "full_diagnostic",
    "/vibration": "vibration_analysis",
    "/gps": "gps_investigation",
    "/battery": "battery_analysis",
    "/ekf": "ekf_check",
    "/tuning": "control_tuning",
    "/help": None,  # 特殊命令
}


def parse_slash_command(text: str) -> Optional[str]:
    """解析斜杠命令，返回对应的预设 ID"""
    text = text.strip().lower()
    if text in SLASH_COMMANDS:
        return SLASH_COMMANDS[text]

    # 检查是否以斜杠开头
    if text.startswith("/"):
        cmd = text.split()[0]
        return SLASH_COMMANDS.get(cmd)

    return None


def get_help_text() -> str:
    """获取帮助文本"""
    return """
**可用的斜杠命令：**

| 命令 | 功能 | 预计时间 |
|------|------|---------|
| `/quick` | 快速健康检查 | ~30秒 |
| `/full` | 完整诊断报告 | ~2分钟 |
| `/vibration` | 震动分析 | ~1分钟 |
| `/gps` | GPS 问题排查 | ~1分钟 |
| `/battery` | 电池分析 | ~30秒 |
| `/ekf` | EKF 状态检查 | ~1分钟 |
| `/tuning` | 控制调参分析 | ~2分钟 |
| `/help` | 显示此帮助 | - |

**使用方式：** 直接输入命令，或输入问题让 AI 自由分析。
"""
