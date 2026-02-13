"""
诊断流程控制器 - 管理结构化的诊断过程
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum


class StageStatus(Enum):
    """阶段状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class DiagnosticStage:
    """诊断阶段定义"""
    id: str
    name: str
    description: str
    tools: List[str]
    prompt: str
    required: bool = True
    depends_on: List[str] = field(default_factory=list)


@dataclass
class StageResult:
    """阶段执行结果"""
    stage_id: str
    status: StageStatus
    tool_calls: List[Dict] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    error: Optional[str] = None


# 预定义的诊断阶段
DIAGNOSTIC_STAGES = [
    DiagnosticStage(
        id="preflight",
        name="预检阶段",
        description="快速评估飞行整体健康状态",
        tools=["get_quick_health_check"],
        prompt="首先调用 get_quick_health_check 获取飞行整体状态。",
        required=True,
    ),
    DiagnosticStage(
        id="subsystem_check",
        name="子系统检查",
        description="检查各子系统状态",
        tools=["get_subsystem_summary"],
        prompt="根据预检发现的警告，调用 get_subsystem_summary 检查相关子系统。",
        required=True,
        depends_on=["preflight"],
    ),
    DiagnosticStage(
        id="event_analysis",
        name="事件分析",
        description="分析关键事件时间线",
        tools=["get_event_timeline"],
        prompt="调用 get_event_timeline 分析关键事件序列。",
        required=False,
        depends_on=["preflight"],
    ),
    DiagnosticStage(
        id="signal_analysis",
        name="信号分析",
        description="深入分析可疑信号",
        tools=["get_signal_stats", "detect_anomalies"],
        prompt="对发现的问题信号进行深入分析。",
        required=False,
        depends_on=["subsystem_check"],
    ),
    DiagnosticStage(
        id="root_cause",
        name="根因诊断",
        description="综合分析问题根因",
        tools=[],
        prompt="综合前述发现，分析问题根因。",
        required=True,
        depends_on=["subsystem_check"],
    ),
    DiagnosticStage(
        id="recommendation",
        name="建议生成",
        description="生成改进建议",
        tools=["search_parameters"],
        prompt="基于诊断结果，生成复飞前检查清单和参数建议。",
        required=True,
        depends_on=["root_cause"],
    ),
]


class DiagnosticFlow:
    """
    诊断流程控制器

    管理结构化的诊断过程，包括:
    - 阶段进度跟踪
    - 工具调用记录
    - 发现汇总
    """

    def __init__(self, stages: List[DiagnosticStage] = None):
        self.stages = stages or DIAGNOSTIC_STAGES
        self.stage_results: Dict[str, StageResult] = {}
        self.current_stage_id: Optional[str] = None
        self.findings: List[str] = []
        self._stage_index = 0
        self._on_progress: Optional[Callable] = None
        self._on_tool_call: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable):
        """设置进度回调函数"""
        self._on_progress = callback

    def set_tool_call_callback(self, callback: Callable):
        """设置工具调用回调函数"""
        self._on_tool_call = callback

    def start(self):
        """开始诊断流程"""
        self._stage_index = 0
        self.stage_results = {}
        self.findings = []
        self._move_to_next_stage()

    def _move_to_next_stage(self):
        """移动到下一个可执行的阶段"""
        while self._stage_index < len(self.stages):
            stage = self.stages[self._stage_index]

            # 检查依赖
            deps_met = all(
                self.stage_results.get(dep, StageResult("", StageStatus.PENDING)).status == StageStatus.COMPLETED
                for dep in stage.depends_on
            )

            if deps_met:
                self.current_stage_id = stage.id
                self.stage_results[stage.id] = StageResult(
                    stage_id=stage.id,
                    status=StageStatus.RUNNING
                )
                self._notify_progress()
                return stage
            else:
                # 跳过依赖未满足的阶段
                self.stage_results[stage.id] = StageResult(
                    stage_id=stage.id,
                    status=StageStatus.SKIPPED
                )
                self._stage_index += 1

        self.current_stage_id = None
        return None

    def get_current_stage(self) -> Optional[DiagnosticStage]:
        """获取当前阶段"""
        if self.current_stage_id:
            for stage in self.stages:
                if stage.id == self.current_stage_id:
                    return stage
        return None

    def record_tool_call(self, tool_name: str, args: Dict, result: Dict):
        """记录工具调用"""
        if self.current_stage_id:
            result_obj = self.stage_results.get(self.current_stage_id)
            if result_obj:
                result_obj.tool_calls.append({
                    "tool": tool_name,
                    "args": args,
                    "result_summary": self._summarize_result(tool_name, result)
                })

                # 通知回调
                if self._on_tool_call:
                    self._on_tool_call(self.current_stage_id, tool_name, result)

    def _summarize_result(self, tool_name: str, result: Dict) -> str:
        """生成结果摘要"""
        if tool_name == "get_quick_health_check":
            status = "正常" if result.get("flight_ok") else "异常"
            warnings = len(result.get("warnings", []))
            return f"状态: {status}, 警告: {warnings} 项"
        elif tool_name == "get_subsystem_summary":
            subsystem = result.get("subsystem", "?")
            status = result.get("status", "?")
            issues = len(result.get("issues", []))
            return f"{subsystem}: {status}, 问题: {issues} 项"
        elif tool_name == "get_event_timeline":
            count = result.get("count", 0)
            return f"共 {count} 个事件"
        elif tool_name == "get_signal_stats":
            field = result.get("field", "?")
            mean = result.get("mean", "?")
            std = result.get("std", "?")
            return f"{field}: mean={mean}, std={std}"
        else:
            return "已获取"

    def add_finding(self, finding: str):
        """添加发现"""
        self.findings.append(finding)
        if self.current_stage_id:
            result = self.stage_results.get(self.current_stage_id)
            if result:
                result.findings.append(finding)

    def complete_stage(self, success: bool = True, error: str = None):
        """完成当前阶段"""
        if self.current_stage_id:
            result = self.stage_results.get(self.current_stage_id)
            if result:
                result.status = StageStatus.COMPLETED if success else StageStatus.ERROR
                result.error = error

            self._stage_index += 1
            self._move_to_next_stage()

    def is_complete(self) -> bool:
        """检查流程是否完成"""
        return self.current_stage_id is None

    def get_progress(self) -> Dict:
        """获取进度信息"""
        completed = sum(1 for r in self.stage_results.values() if r.status == StageStatus.COMPLETED)
        total = len(self.stages)

        stages_status = []
        for stage in self.stages:
            result = self.stage_results.get(stage.id)
            status = result.status.value if result else StageStatus.PENDING.value
            stages_status.append({
                "id": stage.id,
                "name": stage.name,
                "status": status,
                "tool_calls": len(result.tool_calls) if result else 0,
                "findings": len(result.findings) if result else 0,
            })

        return {
            "current_stage": self.current_stage_id,
            "completed_stages": completed,
            "total_stages": total,
            "progress_pct": round(completed / total * 100, 1) if total > 0 else 0,
            "stages": stages_status,
            "is_complete": self.is_complete(),
        }

    def _notify_progress(self):
        """通知进度更新"""
        if self._on_progress:
            self._on_progress(self.get_progress())

    def get_next_prompt(self) -> Optional[str]:
        """获取下一个阶段的 prompt"""
        stage = self.get_current_stage()
        if stage:
            return stage.prompt
        return None

    def get_summary(self) -> str:
        """生成诊断摘要"""
        lines = ["## 诊断摘要\n"]

        for stage in self.stages:
            result = self.stage_results.get(stage.id)
            if result:
                status_icon = {
                    StageStatus.COMPLETED: "✅",
                    StageStatus.SKIPPED: "⏭️",
                    StageStatus.ERROR: "❌",
                    StageStatus.RUNNING: "🔄",
                    StageStatus.PENDING: "⏸️",
                }.get(result.status, "❓")

                lines.append(f"{status_icon} **{stage.name}**")

                if result.tool_calls:
                    lines.append(f"   - 工具调用: {len(result.tool_calls)} 次")
                    for tc in result.tool_calls[:3]:
                        lines.append(f"     - {tc['tool']}: {tc['result_summary']}")

                if result.findings:
                    lines.append(f"   - 发现: {len(result.findings)} 项")
                    for f in result.findings[:2]:
                        lines.append(f"     - {f[:50]}...")

                lines.append("")

        if self.findings:
            lines.append("### 主要发现")
            for i, f in enumerate(self.findings[:5], 1):
                lines.append(f"{i}. {f}")

        return "\n".join(lines)


def create_quick_flow() -> DiagnosticFlow:
    """创建快速诊断流程"""
    return DiagnosticFlow(stages=[
        DIAGNOSTIC_STAGES[0],  # preflight
        DIAGNOSTIC_STAGES[1],  # subsystem_check
        DIAGNOSTIC_STAGES[4],  # root_cause
        DIAGNOSTIC_STAGES[5],  # recommendation
    ])


def create_gps_flow() -> DiagnosticFlow:
    """创建 GPS 问题诊断流程"""
    return DiagnosticFlow(stages=[
        DiagnosticStage(
            id="gps_preflight",
            name="GPS 预检",
            description="检查 GPS 整体状态",
            tools=["get_subsystem_summary"],
            prompt="调用 get_subsystem_summary('gps') 获取 GPS 状态。",
            required=True,
        ),
        DiagnosticStage(
            id="gps_signal",
            name="GPS 信号分析",
            description="分析 GPS 信号质量",
            tools=["get_signal_stats"],
            prompt="调用 get_signal_stats 分析 GPS 相关信号的统计特征。",
            required=True,
            depends_on=["gps_preflight"],
        ),
        DiagnosticStage(
            id="gps_events",
            name="GPS 相关事件",
            description="分析与 GPS 相关的事件",
            tools=["get_event_timeline"],
            prompt="调用 get_event_timeline 查找与位置/EKF 相关的事件。",
            required=False,
            depends_on=["gps_preflight"],
        ),
        DiagnosticStage(
            id="gps_recommendation",
            name="GPS 改进建议",
            description="给出 GPS 问题改进建议",
            tools=["search_parameters"],
            prompt="基于分析结果，搜索 GPS 相关参数并给出改进建议。",
            required=True,
            depends_on=["gps_signal"],
        ),
    ])


def create_vibration_flow() -> DiagnosticFlow:
    """创建震动分析流程"""
    return DiagnosticFlow(stages=[
        DiagnosticStage(
            id="vib_preflight",
            name="震动预检",
            description="检查 IMU 震动状态",
            tools=["get_subsystem_summary"],
            prompt="调用 get_subsystem_summary('imu') 获取 IMU 状态。",
            required=True,
        ),
        DiagnosticStage(
            id="vib_analysis",
            name="震动信号分析",
            description="分析加速度计信号",
            tools=["get_signal_stats", "detect_anomalies"],
            prompt="调用 get_signal_stats 和 detect_anomalies 分析震动特征。",
            required=True,
            depends_on=["vib_preflight"],
        ),
        DiagnosticStage(
            id="vib_recommendation",
            name="减震建议",
            description="给出减震改进建议",
            tools=["search_parameters"],
            prompt="基于分析结果，给出减震相关的参数和硬件建议。",
            required=True,
            depends_on=["vib_analysis"],
        ),
    ])
