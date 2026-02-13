# AI 智能分析增强 - 需求分析文档

## 一、当前实现分析

### 1.1 现有架构

```
app.py
├── build_ai_system_prompt()      # 构建系统提示词
├── build_tool_specs()            # 定义 8 个工具
├── execute_tool()                # 执行工具调用
└── run_tool_agent_report()       # 工具代理循环
```

### 1.2 当前问题诊断

#### 问题 1: 上下文爆炸
```python
# 当前实现：每次工具调用的完整结果都加入 messages
messages.append({
    "role": "tool",
    "content": json.dumps(tool_result, ensure_ascii=False),  # 可能非常大
})
```
- `get_flight_summary()` 返回完整 JSON (~2KB)
- `get_event_timeline()` 120 条事件 (~10KB)
- `get_topic_preview()` 120 行数据 (~50KB+)
- 多轮对话后，上下文轻松超过 100KB

#### 问题 2: 工具返回数据冗余
```python
# get_topic_preview 返回完整字典
{
    "topic": "...",
    "rows": 120,
    "columns": [...],
    "data": [{"timestamp": 0.1, "x": 1.2, ...}, ...]  # 120 条完整记录
}
```
- AI 实际需要的往往只是统计摘要，不是原始数据

#### 问题 3: 没有结构化诊断流程
```python
# 当前的 system prompt 过于简单
"你是 PX4 日志诊断代理。你可以按需调用工具，不要一次性拉取过多数据。"
```
- 没有定义诊断步骤
- 没有检查清单
- AI 需要自己决定分析什么，效率低

#### 问题 4: 触发机制单一
```python
# 只在包含 "诊断报告" 时才用工具代理
if use_tool_agent and "诊断报告" in user_prompt:
    run_tool_agent_report(...)
```
- 用户问具体问题时，不使用工具
- 流式对话模式下，AI 没有任何日志数据

#### 问题 5: 缺少进度反馈
```python
msg_box.markdown("正在调用本地工具分析日志，请稍候...")
# 然后就没有任何更新，直到完成
```
- 用户不知道 AI 在分析什么
- 等待体验差

---

## 二、改进方案设计

### 2.1 架构重构

```
modules/
├── ai_agent/
│   ├── __init__.py
│   ├── tools.py              # 工具定义与执行
│   ├── prompts.py            # 系统 prompt 模板
│   ├── context_manager.py    # 上下文管理（摘要、截断）
│   ├── diagnostic_flow.py    # 诊断流程控制器
│   └── presets.py            # 预设诊断模板
└── analyzer.py               # 现有分析器
```

### 2.2 核心改进项

#### 改进 1: 分层工具设计

将工具分为三个层次，返回不同粒度的数据：

| 层次 | 工具类型 | 返回内容 | 大小控制 |
|------|---------|---------|---------|
| L1 摘要层 | `get_xxx_summary` | 精简的关键指标 | < 500 字符 |
| L2 统计层 | `get_xxx_stats` | 统计特征 | < 2KB |
| L3 原始层 | `get_xxx_data` | 原始数据（谨慎使用） | 限制行列 |

**新增 L1 摘要层工具**：

```python
# 新工具：返回精简摘要
{
    "name": "get_quick_health_check",
    "description": "快速获取飞行健康状态摘要（推荐首先调用）",
    "returns": {
        "flight_ok": True/False,
        "warnings": ["GPS 信号弱", "震动偏高"],
        "key_events": ["12.3s 解锁", "45.6s 自动模式"],
        "recommendation": "建议进一步检查..."
    }
}
```

#### 改进 2: 上下文管理器

```python
class ContextManager:
    """管理对话上下文，防止爆炸"""

    def __init__(self, max_tokens=32000):
        self.max_tokens = max_tokens
        self.messages = []

    def add_tool_result(self, tool_name, result):
        """添加工具结果，自动压缩"""
        if self._estimate_tokens(result) > 2000:
            result = self._compress_result(tool_name, result)
        self.messages.append(...)

    def _compress_result(self, tool_name, result):
        """压缩大结果"""
        # 1. 提取关键信息
        # 2. 生成摘要
        # 3. 保留原始数据的引用
        pass

    def maybe_summarize(self):
        """上下文过长时生成摘要"""
        if self._total_tokens() > self.max_tokens:
            summary = self._generate_summary()
            self.messages = [summary] + self.messages[-5:]
```

#### 改进 3: 结构化诊断流程

```python
DIAGNOSTIC_STAGES = [
    {
        "stage": "preflight_check",
        "name": "预检阶段",
        "tools": ["get_quick_health_check"],
        "prompt": "快速评估本次飞行的整体健康状态..."
    },
    {
        "stage": "event_analysis",
        "name": "事件分析",
        "tools": ["get_event_timeline", "get_mode_changes"],
        "prompt": "分析关键事件序列..."
    },
    {
        "stage": "signal_inspection",
        "name": "信号检查",
        "tools": ["get_gps_summary", "get_battery_summary", "get_ekf_summary"],
        "prompt": "检查各子系统状态..."
    },
    {
        "stage": "root_cause",
        "name": "根因诊断",
        "tools": ["get_anomaly_report", "get_topic_stats"],
        "prompt": "根据前述发现，定位问题根因..."
    },
    {
        "stage": "recommendation",
        "name": "建议生成",
        "tools": ["search_parameters"],
        "prompt": "生成复飞前检查清单..."
    }
]
```

#### 改进 4: 预设诊断模板

```python
DIAGNOSTIC_PRESETS = {
    "quick_health": {
        "name": "快速健康检查",
        "description": "30秒快速评估飞行状态",
        "stages": ["preflight_check"],
        "output_template": """
## 飞行健康报告

**状态**: {status_emoji} {status}

**关键指标**:
- 飞行时长: {duration}
- 最大高度: {max_alt}
- GPS 状态: {gps_status}
- 电池状态: {battery_status}

**警告项** ({warning_count}):
{warnings}

**建议**: {recommendation}
"""
    },

    "full_diagnostic": {
        "name": "完整诊断报告",
        "description": "全面分析所有子系统",
        "stages": DIAGNOSTIC_STAGES,
        "output_template": "..."
    },

    "vibration_analysis": {
        "name": "震动分析",
        "description": "深度分析震动问题",
        "focus": ["imu", "actuators", "control"],
        "stages": [...]
    },

    "gps_investigation": {
        "name": "GPS 问题排查",
        "description": "排查 GPS 信号/干扰问题",
        "focus": ["gps", "ekf", "position"],
        "stages": [...]
    }
}
```

#### 改进 5: 实时进度反馈

```python
class DiagnosticProgress:
    """诊断进度跟踪"""

    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.stages = []
        self.current_stage = None

    def start_stage(self, stage_name, tools):
        """开始新阶段"""
        self.current_stage = {
            "name": stage_name,
            "tools": tools,
            "results": [],
            "status": "running"
        }
        self._render()

    def tool_called(self, tool_name, summary):
        """工具被调用"""
        self.current_stage["results"].append({
            "tool": tool_name,
            "summary": summary
        })
        self._render()

    def _render(self):
        """渲染进度 UI"""
        # 显示：
        # ✅ 预检阶段 - 完成
        # 🔄 事件分析 - 进行中
        #    ├─ ✅ get_event_timeline: 发现 5 个事件
        #    └─ ⏳ get_mode_changes...
        # ⏸️ 信号检查 - 等待中
        pass
```

---

## 三、新工具设计

### 3.1 L1 摘要层工具

```python
def get_quick_health_check(analyzer):
    """快速健康检查 - 返回精简摘要"""
    return {
        "flight_ok": bool,           # 总体是否正常
        "duration_s": float,         # 飞行时长
        "max_alt_m": float,          # 最大高度
        "max_speed_mps": float,      # 最大速度
        "arming_successful": bool,   # 解锁是否成功
        "warnings": [                # 警告列表（最多5条）
            {"type": "gps", "message": "GPS fix 等级为 2，较低"},
            {"type": "vibration", "message": "震动水平偏高"}
        ],
        "key_events": [              # 关键事件（最多5条）
            {"t_s": 12.3, "event": "ARMED"},
            {"t_s": 45.6, "event": "模式切换至 POSCTL"}
        ]
    }

def get_subsystem_summary(analyzer, subsystem):
    """获取子系统摘要"""
    # subsystem: gps, battery, ekf, imu, actuators, rc
    return {
        "subsystem": "gps",
        "status": "warning",         # ok, warning, error
        "metrics": {
            "fix_type": 3,
            "satellites": 8,
            "eph_m": 1.2
        },
        "issues": ["GPS fix 在 30-45s 期间下降"],
        "related_params": ["GPS_UBX_DYNMODEL"]
    }
```

### 3.2 改进现有工具

```python
def get_topic_preview_v2(analyzer, topic, fields=None, mode="summary"):
    """
    mode:
    - "summary": 返回统计摘要（推荐）
    - "sample": 返回采样数据（head/tail/min/max)
    - "full": 返回完整数据（谨慎）
    """
    if mode == "summary":
        return {
            "topic": topic,
            "row_count": 1000,
            "time_range": [0.0, 120.5],
            "fields": {
                "x": {"min": -10, "max": 50, "mean": 15.2, "std": 8.3},
                "y": {"min": -5, "max": 30, "mean": 10.1, "std": 5.2}
            },
            "anomalies": [
                {"field": "x", "count": 3, "time_range": [45.2, 46.1]}
            ]
        }
```

---

## 四、用户界面改进

### 4.1 诊断入口改进

```
┌─────────────────────────────────────────────────────────┐
│  💬 AI 智能分析                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  快速诊断 (点击开始):                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ 🏥 快速检查 │ │ 🔍 完整诊断 │ │ 📡 GPS排查  │       │
│  │   ~30秒     │ │   ~2分钟    │ │   ~1分钟    │       │
│  └─────────────┘ └─────────────┘ └─────────────┘       │
│                                                         │
│  ─────────────────────────────────────────────────      │
│                                                         │
│  诊断进度:                                              │
│  ✅ 预检阶段 - 完成                                     │
│  🔄 事件分析 - 进行中                                   │
│     ├─ ✅ 事件时间线: 5 个事件                          │
│     └─ ⏳ 模式变化分析...                               │
│  ⏸️ 信号检查 - 等待中                                   │
│                                                         │
│  ─────────────────────────────────────────────────      │
│                                                         │
│  [对话历史...]                                          │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 输入问题或使用 / 命令...                         │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 4.2 斜杠命令支持

```
/quick      - 快速健康检查
/full       - 完整诊断报告
/gps        - GPS 问题排查
/vibration  - 震动分析
/battery    - 电池分析
/ekf        - EKF 状态检查
/compare    - 对比分析（多日志时）
/help       - 显示帮助
```

---

## 五、实现优先级

| 优先级 | 功能 | 工作量 | 影响 |
|--------|------|--------|------|
| P0 | L1 摘要层工具 | 2天 | 解决上下文爆炸 |
| P0 | 上下文管理器 | 1天 | 防止 token 超限 |
| P1 | 结构化诊断流程 | 2天 | 提升分析效率 |
| P1 | 预设诊断模板 | 1天 | 改善用户体验 |
| P2 | 实时进度反馈 | 1天 | 改善等待体验 |
| P2 | 斜杠命令 | 0.5天 | 便捷操作 |
| P3 | 改进现有工具 | 1天 | 数据优化 |

---

## 六、技术细节

### 6.1 Token 估算

```python
def estimate_tokens(text):
    """估算文本的 token 数量"""
    # 中文: ~1.5 字符/token
    # 英文: ~4 字符/token
    # JSON: 更高密度
    if isinstance(text, dict):
        text = json.dumps(text, ensure_ascii=False)
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)
```

### 6.2 结果压缩策略

```python
def compress_tool_result(tool_name, result, max_tokens=1500):
    """压缩工具结果"""
    if estimate_tokens(result) <= max_tokens:
        return result

    if tool_name == "get_event_timeline":
        # 只保留关键事件类型
        events = result.get("events", [])
        key_types = ["arming_state", "failsafe", "ekf_reset"]
        compressed = [e for e in events if e.get("kind") in key_types]
        return {"events": compressed[:20], "note": "已压缩，仅显示关键事件"}

    if tool_name == "get_topic_preview":
        # 返回统计摘要替代原始数据
        return convert_to_summary(result)
```

---

*文档版本: 1.0*
*创建日期: 2026-02-14*
