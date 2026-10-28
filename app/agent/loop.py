"""Agent 编排循环：让模型自己决定查什么、查几次。

和 W5 的 answer() 有什么区别：

    answer()  固定流程：问题 → 检索 → 生成。适合单一事实问答。
    run()     模型自主决定调用哪些工具、调用几次，直到它认为信息够了。

为什么需要后者——看一个真实问题：

    「我在吃华法林和阿司匹林，最近牙龈出血，报告里 HGB 108 偏低，要注意什么？」

固定流程做不到，因为它不知道要查几个药、查哪一章、要不要核对指标。
Agent 会自己拆成：查华法林出血相关 → 查阿司匹林 → 核对 HGB 与两药的关系 → 综合。

三个必须有的保护：

    轮数上限    模型可能反复调工具不停下来，max_turns 兜底
    空答案兜底  超轮数时模型可能什么都没说，要给出可解释的结果而不是空字符串
    工具异常    工具内部报错要喂回给模型，让它自己纠正，而不是中断整个循环
"""

import json
import logging
from dataclasses import dataclass, field

from app.agent.tools import TOOLS, ToolContext, execute
from app.llm.client import ChatClient, build_chat_client
from app.llm.prompts import REFUSAL_PHRASE, extract_cited_numbers
from app.rag.citation import Citation

logger = logging.getLogger(__name__)

# Agent 模式下「拒答」的判定不能只看关键词。
#
# 踩过两次坑：
#   1. 只认「资料中未提及」→ 漏判「知识库未收录 X」（工具转述的措辞）
#   2. 放宽成关键词包含 → 误伤「会。……诱发消化道溃疡出血[1]。
#      资料中未提及短期规范用药是否会损伤胃。」这种「作答 + 补充边界」的答案
#
# 所以判据改成结构性的，而不是找词：
_OFF_TOPIC_HINTS = ("无法回答", "不能回答", "无法提供", "不属于我的")


def _is_refusal(text: str, hits: list[dict], steps: list["AgentStep"]) -> bool:
    """判断这次答复算不算拒答。

    四条规则，按可靠性从高到低：

        1. 调了工具却一条资料都没拿到  → 拒答（不管措辞）
        2. 拿到了资料且答案引用了编号  → 明确是作答
        3. 整段就只有一句拒答短语      → 拒答
        4. 没调工具（寒暄/跑题）        → 看有没有明确的能力外声明
    """
    if steps and not hits:
        return True
    if hits and extract_cited_numbers(text):
        return False

    core = text.strip().strip("。.！! ")
    if core == REFUSAL_PHRASE:
        return True

    if not steps and not hits:
        return any(hint in text for hint in _OFF_TOPIC_HINTS)
    return False

SYSTEM_PROMPT = f"""你是体检报告解读助手，可以调用工具查阅药品说明书。

【工作方式】
1. 先判断问题需不需要查资料。需要就调用工具，不需要就直接回答。
2. 一次只查一件事。要对比两种药，就分别调用两次。
3. 工具返回的资料带 [1][2] 编号，回答时用编号标注来源。
4. 如果工具返回「未收录」，如实告诉用户，**不要用其他药的资料代替**。
5. 信息足够回答时立刻作答，不要为了「查得更全」而反复调用工具。

【必须遵守】
- 只能依据工具返回的资料回答，不得使用你自己的医学知识补充或推测。
- 不下诊断结论，不建议增减剂量或停药。
- 资料不足以回答时，只输出这一句：{REFUSAL_PHRASE}
- 用简洁中文回答，控制在 300 字以内。"""


@dataclass(frozen=True)
class AgentStep:
    """一次工具调用记录。展示给用户就是「它查了什么」。"""

    tool: str
    args: dict
    output: str
    error: bool = False


@dataclass
class AgentResult:
    question: str
    answer: str = ""
    refused: bool = False
    steps: list[AgentStep] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    # 与 citations 一一对应的原始编号。
    #
    # 为什么必须留着：citations 按「文件+章节」去重了，而答案里的编号是
    # 工具返回的全局序号。三个肝功能指标可能都引用同一条「注意事项」，
    # 答案里写 [2][4][6]，去重后 citations 只有一条。
    # 前端若按顺序重新编号，就会出现「正文说 [6]、出处只有 [1]」的情况。
    cited_numbers: list[int] = field(default_factory=list)
    cited: bool = False
    turns: int = 0
    stopped: str = "done"  # done / max_turns


def _format_report(report) -> str:
    """把报告压缩成模型能一眼看懂的形式。

    只放异常项——21 项全列会让模型把注意力分散到正常指标上，
    而且正常指标对「要注意什么」没有信息量。
    """
    abnormal = [i for i in report.items if i.flag in ("high", "low")]
    if not abnormal:
        return f"【体检报告】编号 {report.report_id}，未发现异常指标"

    lines = [f"【体检报告】编号 {report.report_id}，采样日期 {report.exam_date or '未识别'}",
             f"共 {len(report.items)} 项，其中 {len(abnormal)} 项异常："]
    for item in abnormal:
        unit = f" {item.unit}" if item.unit else ""
        ref = ""
        if item.ref_low is not None and item.ref_high is not None:
            ref = f"（参考 {item.ref_low:g}–{item.ref_high:g}）"
        elif item.ref_high is not None:
            ref = f"（参考 <{item.ref_high:g}）"
        elif item.ref_low is not None:
            ref = f"（参考 >{item.ref_low:g}）"
        flag = "偏高" if item.flag == "high" else "偏低"
        lines.append(f"- {item.name_raw} {item.value:g}{unit}{ref}{flag}")
    return "\n".join(lines)


def build_user_message(question: str, ctx: ToolContext) -> str:
    parts: list[str] = []
    if ctx.report is not None:
        parts.append(_format_report(ctx.report))
    if ctx.medications:
        parts.append(f"【在服药物】{'、'.join(ctx.medications)}")
    parts.append(f"【问题】{question}")
    return "\n\n".join(parts)


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def _citations_from_hits(numbers: list[int], hits: list[dict]) -> tuple[list[int], list[Citation]]:
    """把答案里的编号映射回出处，同时保留去重前的原始编号。

    返回 (kept_numbers, citations)，两个列表一一对应。
    """
    result: list[Citation] = []
    kept: list[int] = []
    seen: set[tuple] = set()
    for number in numbers:
        if not 1 <= number <= len(hits):
            continue
        meta = hits[number - 1].get("metadata") or {}
        citation = Citation(
            source=meta.get("source", "未知来源"),
            section=meta.get("section"),
            chunk_id=hits[number - 1].get("id"),
        )
        if citation.key in seen:
            continue
        seen.add(citation.key)
        result.append(citation)
        kept.append(number)
    return kept, result


def run(
    question: str,
    ctx: ToolContext,
    client: ChatClient | None = None,
    max_turns: int = 6,
) -> AgentResult:
    """跑一轮完整的 Agent 对话。"""
    client = client or build_chat_client()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, ctx)},
    ]

    steps: list[AgentStep] = []
    result = AgentResult(question=question)

    for turn in range(1, max_turns + 1):
        reply = client.complete(messages, tools=TOOLS)
        result.turns = turn

        # 必须用它，不能自己拼 message：
        # 带 tools 的多轮里少了 reasoning_content 会被服务端判 400。
        messages.append(reply.as_assistant_message())

        if not reply.has_tool_calls:
            result.answer = reply.content
            result.refused = _is_refusal(reply.content, ctx.hits, steps)
            numbers = extract_cited_numbers(reply.content)
            kept, citations = _citations_from_hits(numbers, ctx.hits)
            result.citations = citations
            result.cited_numbers = kept
            result.cited = bool(numbers)
            result.stopped = "done"
            result.steps = steps
            return result

        for call in reply.tool_calls:
            function = call.get("function") or {}
            name = function.get("name", "")
            args = _parse_args(function.get("arguments"))

            tool_result = execute(call, ctx)
            logger.info("工具 %s(%s) → %s 字", name, args, len(tool_result.content))

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": tool_result.content,
                }
            )
            steps.append(
                AgentStep(
                    tool=name,
                    args=args,
                    output=tool_result.content,
                    error=tool_result.error,
                )
            )

    # 走到这里是轮数用尽：模型一直在调工具没停下来。
    # 不能返回空字符串——上层会把它当成「拒答」，掩盖真正的原因是「没收敛」。
    tools_used = "、".join(dict.fromkeys(s.tool for s in steps)) or "无"
    result.answer = (
        f"（已达到 {max_turns} 轮工具调用上限，未能形成最终答复。"
        f"本次调用过：{tools_used}。可缩小问题范围后重试。）"
    )
    result.stopped = "max_turns"
    result.steps = steps
    return result
