"""
上下文管理器 - 管理 AI 对话上下文，防止 token 爆炸
增强版：更智能的压缩策略，支持增量压缩
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ContextManager:
    """
    AI 对话上下文管理器

    功能:
    - 自动估算 token 数量
    - 压缩过大的工具结果
    - 智能压缩策略：提前压缩，保持上下文在安全范围
    - 支持增量压缩和摘要生成
    """

    max_tokens: int = 32000
    compression_threshold: int = 1500  # 单条消息超过此值时压缩（降低阈值）
    safe_utilization: float = 0.6  # 保持在60%以下更安全
    messages: List[Dict] = field(default_factory=list)
    _tool_results_cache: Dict[str, Any] = field(default_factory=dict)
    _compression_count: int = 0  # 压缩次数统计

    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self._tool_results_cache is None:
            self._tool_results_cache = {}

    def estimate_tokens(self, content: Any) -> int:
        """估算内容的 token 数量"""
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False, default=str)
        elif not isinstance(content, str):
            content = str(content)

        # 中文约 1.5 字符/token，英文约 4 字符/token
        chinese_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
        other_chars = len(content) - chinese_chars
        return max(1, int(chinese_chars / 1.5 + other_chars / 4))

    def total_tokens(self) -> int:
        """计算当前总 token 数"""
        total = 0
        for msg in self.messages:
            if isinstance(msg.get("content"), str):
                total += self.estimate_tokens(msg["content"])
            elif isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if isinstance(part, dict) and "text" in part:
                        total += self.estimate_tokens(part["text"])
            # tool_calls
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    total += self.estimate_tokens(tc.get("function", {}).get("arguments", ""))
        return total

    def add_message(self, role: str, content: Any, **kwargs):
        """添加消息到上下文"""
        msg = {"role": role, "content": content, **kwargs}
        self.messages.append(msg)
        self._check_and_compress()

    def add_user_message(self, content: str):
        """添加用户消息"""
        self.add_message("user", content)

    def add_assistant_message(self, content: str = None, tool_calls: List[Dict] = None):
        """添加助手消息"""
        msg = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)
        self._check_and_compress()

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: Dict):
        """添加工具结果，自动压缩大结果"""
        # 缓存原始结果
        self._tool_results_cache[tool_call_id] = {
            "tool_name": tool_name,
            "full_result": result
        }

        # 激进压缩结果
        compressed = self._compress_result(tool_name, result)

        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(compressed, ensure_ascii=False, default=str)
        })

        self._check_and_compress()

    def _compress_result(self, tool_name: str, result: Dict) -> Dict:
        """压缩工具结果 - 更激进的策略"""
        tokens = self.estimate_tokens(result)

        # 根据当前上下文使用率调整压缩力度
        utilization = self.total_tokens() / self.max_tokens

        if utilization > 0.5:
            # 上下文较满，更激进压缩
            threshold = 800
        elif utilization > 0.3:
            threshold = 1200
        else:
            threshold = self.compression_threshold

        if tokens <= threshold:
            return result

        # 根据工具类型进行针对性压缩
        if tool_name == "get_event_timeline":
            return self._compress_timeline(result, aggressive=utilization > 0.4)
        elif tool_name == "get_topic_fields":
            return self._compress_topic_fields(result, aggressive=utilization > 0.4)
        elif tool_name == "list_topics":
            return self._compress_list_topics(result, aggressive=utilization > 0.4)
        elif tool_name == "search_parameters":
            return self._compress_parameters(result, aggressive=utilization > 0.4)
        elif tool_name == "detect_anomalies":
            return self._compress_anomalies(result, aggressive=utilization > 0.4)
        elif tool_name == "get_signal_stats":
            return result  # 已经很精简
        elif tool_name == "get_quick_health_check":
            return self._compress_health_check(result)
        elif tool_name == "get_subsystem_summary":
            return self._compress_subsystem_summary(result)
        else:
            # 通用压缩
            return self._generic_compress(result, max_items=5 if utilization > 0.5 else 10)

    def _compress_health_check(self, result: Dict) -> Dict:
        """压缩健康检查结果"""
        warnings = result.get("warnings", [])
        if len(warnings) <= 5:
            return result

        return {
            "flight_ok": result.get("flight_ok"),
            "duration_s": result.get("duration_s"),
            "max_alt_m": result.get("max_alt_m"),
            "max_speed_mps": result.get("max_speed_mps"),
            "warning_count": len(warnings),
            "warnings": warnings[:5],
            "_note": f"已压缩警告，显示前5条（共{len(warnings)}条）"
        }

    def _compress_subsystem_summary(self, result: Dict) -> Dict:
        """压缩子系统摘要"""
        issues = result.get("issues", [])
        if len(issues) <= 3:
            return result

        return {
            "subsystem": result.get("subsystem"),
            "status": result.get("status"),
            "issue_count": len(issues),
            "issues": issues[:3],
            "summary": result.get("summary", "")[:200],
            "_note": f"已压缩问题列表（共{len(issues)}个）"
        }

    def _compress_timeline(self, result: Dict, aggressive: bool = False) -> Dict:
        """压缩事件时间线"""
        events = result.get("events", [])
        max_events = 10 if aggressive else 20
        if len(events) <= max_events:
            return result

        # 按重要性筛选
        important_kinds = ["arming_state", "failsafe", "ekf_reset", "error"]
        important = [e for e in events if e.get("kind") in important_kinds]
        others = [e for e in events if e.get("kind") not in important_kinds]

        # 保留所有重要事件 + 部分其他事件
        kept = important[:max_events//2] + others[:max_events//2]

        return {
            "count": result.get("count", len(events)),
            "events": kept,
            "_note": f"已压缩，从 {len(events)} 条事件中保留 {len(kept)} 条关键事件"
        }

    def _compress_topic_fields(self, result: Dict, aggressive: bool = False) -> Dict:
        """压缩字段列表"""
        fields = result.get("fields", [])
        numeric = result.get("numeric_fields", [])
        max_fields = 15 if aggressive else 30

        if len(fields) <= max_fields:
            return result

        return {
            "topic": result.get("topic"),
            "total_fields": len(fields),
            "numeric_fields": numeric[:max_fields],
            "other_fields_count": len(fields) - len(numeric),
            "_note": f"已压缩，共 {len(fields)} 个字段"
        }

    def _compress_list_topics(self, result: Dict, aggressive: bool = False) -> Dict:
        """压缩 topic 列表"""
        topics = result.get("topics", [])
        max_topics = 15 if aggressive else 30
        if len(topics) <= max_topics:
            return result

        # 按前缀分组
        prefixes = {}
        for t in topics:
            prefix = t.split("_")[0] if "_" in t else t
            prefixes[prefix] = prefixes.get(prefix, 0) + 1

        return {
            "total": result.get("total", len(topics)),
            "sample_topics": topics[:max_topics],
            "prefixes": prefixes,
            "_note": f"已压缩，共 {len(topics)} 个 topic"
        }

    def _compress_parameters(self, result: Dict, aggressive: bool = False) -> Dict:
        """压缩参数列表"""
        params = result.get("parameters", [])
        max_params = 10 if aggressive else 20
        if len(params) <= max_params:
            return result

        return {
            "count": len(params),
            "parameters": params[:max_params],
            "_note": f"已压缩，共 {len(params)} 个参数"
        }

    def _compress_anomalies(self, result: Dict, aggressive: bool = False) -> Dict:
        """压缩异常检测结果"""
        anomalies = result.get("anomalies", [])
        max_anomalies = 5 if aggressive else 10
        if len(anomalies) <= max_anomalies:
            return result

        return {
            "field": result.get("field"),
            "threshold_std": result.get("threshold_std"),
            "anomaly_count": result.get("anomaly_count"),
            "anomaly_percentage": result.get("anomaly_percentage"),
            "bounds": result.get("bounds"),
            "anomalies": anomalies[:max_anomalies],
            "_note": f"已压缩，显示前 {max_anomalies} 个异常（共 {len(anomalies)} 个）"
        }

    def _generic_compress(self, result: Dict, max_items: int = 10, max_depth: int = 2) -> Dict:
        """通用压缩：截断长列表和字符串"""
        def compress_value(v, depth=0):
            if depth > max_depth:
                return "..."

            if isinstance(v, str) and len(v) > 150:
                return v[:150] + f"... (截断，共 {len(v)} 字符)"
            elif isinstance(v, list):
                if len(v) > max_items:
                    return [compress_value(x, depth+1) for x in v[:max_items]] + [f"... (共 {len(v)} 项)"]
                return [compress_value(x, depth+1) for x in v]
            elif isinstance(v, dict):
                return {k: compress_value(val, depth+1) for k, val in v.items()}
            return v

        return compress_value(result)

    def _check_and_compress(self):
        """检查并在需要时压缩上下文 - 智能策略"""
        current_tokens = self.total_tokens()
        utilization = current_tokens / self.max_tokens

        # 多级压缩触发条件
        if utilization > 0.8:
            # 紧急压缩：保留最少消息
            self._emergency_compress()
        elif utilization > self.safe_utilization:
            # 常规压缩
            self._compress_old_messages()
        elif utilization > 0.4 and len(self.messages) > 20:
            # 轻度压缩：压缩工具结果
            self._compress_tool_results_only()

    def _emergency_compress(self):
        """紧急压缩：只保留最关键的消息"""
        if len(self.messages) <= 4:
            return

        self._compression_count += 1

        # 保留 system 消息和最近的消息
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        recent_msgs = self.messages[-3:]  # 只保留最近3条

        # 生成精简摘要
        middle_msgs = [m for m in self.messages if m not in system_msgs and m not in recent_msgs]
        if middle_msgs:
            summary = self._generate_compact_summary(middle_msgs)
            self.messages = system_msgs + [
                {"role": "system", "content": f"[压缩摘要 #{self._compression_count}]\n{summary}"}
            ] + recent_msgs

    def _compress_old_messages(self):
        """压缩旧消息"""
        if len(self.messages) <= 6:
            return

        self._compression_count += 1

        # 保留 system 消息和最近的消息
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        recent_msgs = self.messages[-5:]

        # 对中间消息生成摘要
        middle_msgs = [m for m in self.messages if m not in system_msgs and m not in recent_msgs]
        if middle_msgs:
            summary = self._generate_summary(middle_msgs)
            self.messages = system_msgs + [
                {"role": "system", "content": f"[历史摘要 #{self._compression_count}]\n{summary}"}
            ] + recent_msgs

    def _compress_tool_results_only(self):
        """只压缩工具结果，不改变消息结构"""
        for i, msg in enumerate(self.messages):
            if msg.get("role") == "tool":
                try:
                    content = json.loads(msg["content"])
                    if self.estimate_tokens(content) > 800:
                        compressed = self._generic_compress(content, max_items=3)
                        self.messages[i]["content"] = json.dumps(compressed, ensure_ascii=False, default=str)
                except:
                    pass

    def _generate_compact_summary(self, messages: List[Dict]) -> str:
        """生成精简摘要"""
        parts = []
        tool_calls_made = []

        for msg in messages[-8:]:  # 只看最近8条
            role = msg.get("role", "")
            if role == "user":
                content = msg.get("content", "")[:50]
                parts.append(f"用户: {content}")
            elif role == "assistant":
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        tool_name = tc.get("function", {}).get("name", "")
                        if tool_name:
                            tool_calls_made.append(tool_name)
            elif role == "tool":
                pass  # 跳过工具结果详情

        summary = ""
        if parts:
            summary += "最近对话: " + " | ".join(parts[-2:]) + "\n"
        if tool_calls_made:
            summary += f"已调用工具: {', '.join(tool_calls_made[-5:])}"

        return summary or "历史对话已压缩"

    def _generate_summary(self, messages: List[Dict]) -> str:
        """生成消息摘要"""
        summary_parts = []
        tool_calls_made = []

        for msg in messages:
            role = msg.get("role", "")
            if role == "user":
                content = msg.get("content", "")[:80]
                summary_parts.append(f"用户: {content}")
            elif role == "assistant":
                content = msg.get("content", "")
                if content:
                    summary_parts.append(f"助手: {content[:80]}")
                elif "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        tool_name = tc.get("function", {}).get("name", "")
                        if tool_name:
                            tool_calls_made.append(tool_name)
            elif role == "tool":
                tool_id = msg.get("tool_call_id", "")
                cached = self._tool_results_cache.get(tool_id, {})
                tool_name = cached.get("tool_name", "工具")
                if tool_name not in tool_calls_made:
                    tool_calls_made.append(tool_name)

        result = ""
        if summary_parts:
            result += "\n".join(summary_parts[-6:]) + "\n"
        if tool_calls_made:
            result += f"已调用: {', '.join(tool_calls_made[-8:])}"

        return result or "历史已压缩"

    def get_messages(self) -> List[Dict]:
        """获取当前消息列表（供 API 调用）"""
        return self.messages.copy()

    def get_full_tool_result(self, tool_call_id: str) -> Optional[Dict]:
        """获取工具的完整结果（从缓存）"""
        cached = self._tool_results_cache.get(tool_call_id)
        if cached:
            return cached.get("full_result")
        return None

    def clear(self):
        """清空上下文"""
        self.messages = []
        self._tool_results_cache = {}
        self._compression_count = 0

    def get_stats(self) -> Dict:
        """获取上下文统计信息"""
        return {
            "message_count": len(self.messages),
            "total_tokens": self.total_tokens(),
            "max_tokens": self.max_tokens,
            "cached_results": len(self._tool_results_cache),
            "utilization": round(self.total_tokens() / self.max_tokens * 100, 1),
            "compression_count": self._compression_count,
        }


# 便捷函数
def create_context_manager(max_tokens: int = 32000) -> ContextManager:
    """创建上下文管理器"""
    return ContextManager(max_tokens=max_tokens)
