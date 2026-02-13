"""
上下文管理器 - 管理 AI 对话上下文，防止 token 爆炸
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
    - 在上下文过长时自动摘要
    """

    max_tokens: int = 32000
    compression_threshold: int = 2000  # 单条消息超过此值时压缩
    messages: List[Dict] = field(default_factory=list)
    _tool_results_cache: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self._tool_results_cache is None:
            self._tool_results_cache = {}

    def estimate_tokens(self, content: Any) -> int:
        """估算内容的 token 数量"""
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
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
        self._maybe_compress()

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
        self._maybe_compress()

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: Dict):
        """添加工具结果，自动压缩大结果"""
        # 缓存原始结果
        self._tool_results_cache[tool_call_id] = {
            "tool_name": tool_name,
            "full_result": result
        }

        # 压缩结果
        compressed = self._compress_result(tool_name, result)

        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(compressed, ensure_ascii=False)
        })

        self._maybe_compress()

    def _compress_result(self, tool_name: str, result: Dict) -> Dict:
        """压缩工具结果"""
        tokens = self.estimate_tokens(result)

        if tokens <= self.compression_threshold:
            return result

        # 根据工具类型进行针对性压缩
        if tool_name == "get_event_timeline":
            return self._compress_timeline(result)
        elif tool_name == "get_topic_fields":
            return self._compress_topic_fields(result)
        elif tool_name == "list_topics":
            return self._compress_list_topics(result)
        elif tool_name == "search_parameters":
            return self._compress_parameters(result)
        elif tool_name == "detect_anomalies":
            return self._compress_anomalies(result)
        elif tool_name == "get_signal_stats":
            return result  # 已经很精简
        else:
            # 通用压缩：只保留关键字段
            return self._generic_compress(result)

    def _compress_timeline(self, result: Dict) -> Dict:
        """压缩事件时间线"""
        events = result.get("events", [])
        if len(events) <= 20:
            return result

        # 按重要性筛选
        important_kinds = ["arming_state", "failsafe", "ekf_reset"]
        important = [e for e in events if e.get("kind") in important_kinds]
        others = [e for e in events if e.get("kind") not in important_kinds]

        # 保留所有重要事件 + 部分其他事件
        kept = important[:15] + others[:10]

        return {
            "count": result.get("count", len(events)),
            "events": kept,
            "_note": f"已压缩，从 {len(events)} 条事件中保留 {len(kept)} 条关键事件"
        }

    def _compress_topic_fields(self, result: Dict) -> Dict:
        """压缩字段列表"""
        fields = result.get("fields", [])
        numeric = result.get("numeric_fields", [])

        if len(fields) <= 30:
            return result

        return {
            "topic": result.get("topic"),
            "total_fields": len(fields),
            "numeric_fields": numeric[:30],
            "other_fields_count": len(fields) - len(numeric),
            "_note": f"已压缩，共 {len(fields)} 个字段"
        }

    def _compress_list_topics(self, result: Dict) -> Dict:
        """压缩 topic 列表"""
        topics = result.get("topics", [])
        if len(topics) <= 30:
            return result

        # 按前缀分组
        prefixes = {}
        for t in topics:
            prefix = t.split("_")[0] if "_" in t else t
            prefixes[prefix] = prefixes.get(prefix, 0) + 1

        return {
            "total": result.get("total", len(topics)),
            "sample_topics": topics[:20],
            "prefixes": prefixes,
            "_note": f"已压缩，共 {len(topics)} 个 topic"
        }

    def _compress_parameters(self, result: Dict) -> Dict:
        """压缩参数列表"""
        params = result.get("parameters", [])
        if len(params) <= 20:
            return result

        return {
            "count": len(params),
            "parameters": params[:20],
            "_note": f"已压缩，共 {len(params)} 个参数"
        }

    def _compress_anomalies(self, result: Dict) -> Dict:
        """压缩异常检测结果"""
        anomalies = result.get("anomalies", [])
        if len(anomalies) <= 10:
            return result

        return {
            "field": result.get("field"),
            "threshold_std": result.get("threshold_std"),
            "anomaly_count": result.get("anomaly_count"),
            "anomaly_percentage": result.get("anomaly_percentage"),
            "bounds": result.get("bounds"),
            "anomalies": anomalies[:10],
            "_note": f"已压缩，显示前 10 个异常（共 {len(anomalies)} 个）"
        }

    def _generic_compress(self, result: Dict, max_depth: int = 2) -> Dict:
        """通用压缩：截断长列表和字符串"""
        def compress_value(v, depth=0):
            if depth > max_depth:
                return "..."

            if isinstance(v, str) and len(v) > 200:
                return v[:200] + f"... (截断，共 {len(v)} 字符)"
            elif isinstance(v, list):
                if len(v) > 10:
                    return [compress_value(x, depth+1) for x in v[:10]] + [f"... (共 {len(v)} 项)"]
                return [compress_value(x, depth+1) for x in v]
            elif isinstance(v, dict):
                return {k: compress_value(val, depth+1) for k, val in v.items()}
            return v

        return compress_value(result)

    def _maybe_compress(self):
        """检查并在需要时压缩上下文"""
        if self.total_tokens() > self.max_tokens:
            self._compress_old_messages()

    def _compress_old_messages(self):
        """压缩旧消息"""
        if len(self.messages) <= 6:
            return

        # 保留 system 消息和最近的消息
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        recent_msgs = self.messages[-5:]

        # 对中间消息生成摘要
        middle_msgs = [m for m in self.messages if m not in system_msgs and m not in recent_msgs]
        if middle_msgs:
            summary = self._generate_summary(middle_msgs)
            self.messages = system_msgs + [
                {"role": "system", "content": f"[历史对话摘要]\n{summary}"}
            ] + recent_msgs

    def _generate_summary(self, messages: List[Dict]) -> str:
        """生成消息摘要"""
        summary_parts = []

        for msg in messages:
            role = msg.get("role", "")
            if role == "user":
                content = msg.get("content", "")[:100]
                summary_parts.append(f"用户: {content}...")
            elif role == "assistant":
                content = msg.get("content", "")
                if content:
                    summary_parts.append(f"助手: {content[:100]}...")
                elif "tool_calls" in msg:
                    tools = [tc["function"]["name"] for tc in msg["tool_calls"]]
                    summary_parts.append(f"调用工具: {', '.join(tools)}")
            elif role == "tool":
                # 从缓存获取工具名
                tool_id = msg.get("tool_call_id", "")
                cached = self._tool_results_cache.get(tool_id, {})
                tool_name = cached.get("tool_name", "未知工具")
                summary_parts.append(f"工具结果({tool_name}): 已获取")

        return "\n".join(summary_parts[-10:])  # 最多保留10条摘要

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

    def get_stats(self) -> Dict:
        """获取上下文统计信息"""
        return {
            "message_count": len(self.messages),
            "total_tokens": self.total_tokens(),
            "max_tokens": self.max_tokens,
            "cached_results": len(self._tool_results_cache),
            "utilization": round(self.total_tokens() / self.max_tokens * 100, 1),
        }


# 便捷函数
def create_context_manager(max_tokens: int = 32000) -> ContextManager:
    """创建上下文管理器"""
    return ContextManager(max_tokens=max_tokens)
