"""
LogCortex V3 - AI Agent 模块
提供智能日志分析、诊断流程管理、上下文压缩等功能
"""

from .tools import (
    build_tool_specs,
    execute_tool,
    get_quick_health_check,
    get_subsystem_summary,
    get_signal_stats,
)
from .prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    DIAGNOSTIC_STAGE_PROMPTS,
    build_system_prompt,
)
from .context_manager import ContextManager
from .diagnostic_flow import DiagnosticFlow, DIAGNOSTIC_STAGES
from .presets import DIAGNOSTIC_PRESETS, get_preset_names, get_preset, parse_slash_command, get_help_text

__all__ = [
    # Tools
    "build_tool_specs",
    "execute_tool",
    "get_quick_health_check",
    "get_subsystem_summary",
    "get_signal_stats",
    # Prompts
    "SYSTEM_PROMPT_TEMPLATE",
    "DIAGNOSTIC_STAGE_PROMPTS",
    "build_system_prompt",
    # Context
    "ContextManager",
    # Diagnostic Flow
    "DiagnosticFlow",
    "DIAGNOSTIC_STAGES",
    # Presets
    "DIAGNOSTIC_PRESETS",
    "get_preset_names",
    "get_preset",
    "parse_slash_command",
    "get_help_text",
]
