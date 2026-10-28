"""Agent 可调用的工具。

设计原则：工具要**原子**，不要包成「一键解决问题」的大函数。

原因：Agent 的价值就在于「模型自己决定查什么、查几次」。
如果把「查两个药的相互作用并核对指标」打包成一个工具，
模型就没有编排空间了，退化成一次函数调用——那和 W5 的固定流程没区别。

三个工具各管一件事：

    search_drug_label     查某药某章节的原文（最基础的数据访问）
    check_indicators      核对报告异常指标与在服药物的关联（复用 W5 的 analyzer）
    list_available_drugs  列出库内可查的药（防止模型瞎猜药名）

工具的 description 是最重要的部分——模型完全靠它判断该不该调用。
所以每条都写清楚「什么时候用」，而不是「这个函数做什么」。
"""

import json
from dataclasses import dataclass, field

from app.ingest.models import Report
from app.rag.retriever import supported_drugs


@dataclass
class ToolResult:
    """一次工具调用的结果。

    content 给模型看，hits 留给上层做引用溯源——
    模型只需要读文字，但最终答案要能指回具体章节。
    """

    content: str
    hits: list[dict] = field(default_factory=list)
    error: bool = False


class ToolContext:
    """工具执行所需的上下文。

    报告和用药是「会话级」信息，不该让模型每次通过工具去取——
    它们在一次对话里是固定的，直接放进 system prompt 更省事。
    """

    def __init__(
        self,
        report: Report | None = None,
        medications: list[str] | None = None,
        collection=None,
    ) -> None:
        self.report = report
        self.medications = [m for m in (medications or []) if m]
        self.collection = collection
        self._supported: set[str] | None = None
        # 本次会话检索到的全部片段，按编号顺序累积。
        # 工具返回给模型的文本里带 [1][2] 这样的编号，模型在答案里引用，
        # 最后就能把编号映射回具体章节——这是引用溯源在 Agent 模式下的实现方式。
        self.hits: list[dict] = []
        # chunk_id → 编号。同一段原文被多个指标命中时只发一个编号。
        self._by_chunk: dict[str, int] = {}
        # 已经贴过原文的编号。重复命中时不再重复贴，只保留编号。
        self.printed: set[int] = set()

    def register(self, hit: dict) -> int:
        """登记一条命中，返回它的全局编号（从 1 开始）。

        按 chunk_id 去重是关键。核对 11 项异常时，「阿托伐他汀·禁忌」
        这一段会被 ALT、AST、总胆固醇、HDL-C 分别命中四次——
        如果每次都发新编号，工具会返回 [1][3][6][7] 四个编号指向同一段原文，
        模型照抄进答案，用户看到正文引用 [1]~[7]、出处却只有两条，根本对不上。

        编号必须在产生时就唯一，不能等到展示层再去重——那时已经晚了，
        模型已经写出 [6] 了。
        """
        chunk_id = hit.get("id")
        if chunk_id and chunk_id in self._by_chunk:
            return self._by_chunk[chunk_id]

        self.hits.append(hit)
        number = len(self.hits)
        if chunk_id:
            self._by_chunk[chunk_id] = number
        return number

    @property
    def supported(self) -> set[str]:
        """库内可查的药。只读一次，之后的工具调用直接复用。"""
        if self._supported is None:
            from app.core.config import get_settings
            from app.rag.store import get, get_client

            collection = self.collection or get(get_client(get_settings().chroma_dir))
            self.collection = collection
            self._supported = supported_drugs(collection)
        return self._supported


# ------------------------------------------------------------------ 工具定义

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_drug_label",
            "description": (
                "查询某种药品说明书的指定章节原文。"
                "当需要了解某药的禁忌、注意事项、不良反应或药物相互作用时使用。"
                "药名支持别名（如「华法令」等同于「华法林」）。"
                "一次只查一种药；需要比较两种药时请分别调用两次。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug": {
                        "type": "string",
                        "description": "药品通用名，例如「华法林」「阿托伐他汀」",
                    },
                    "section": {
                        "type": "string",
                        "description": (
                            "章节名，可填「禁忌」「注意事项」「药物相互作用」。"
                            "不填则返回该药的全部章节。"
                        ),
                    },
                },
                "required": ["drug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_indicators",
            "description": (
                "核对体检报告中的异常指标与患者正在服用的药物之间，"
                "在说明书层面是否存在关联。"
                "当问题涉及「我的某项指标异常，和我在吃的药有没有关系」「这个药会不会影响我的某项检查」时使用。"
                "返回的是说明书原文片段，不是诊断结论。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "要核对的指标名，例如 [\"谷丙转氨酶\", \"血红蛋白\"]。"
                            "留空则核对报告中全部异常指标。"
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_drugs",
            "description": (
                "列出知识库中可查询的药品。"
                "当你不确定某种药在不在库里、或 search_drug_label 返回「未收录」时使用，"
                "用来确认正确的药名写法。"
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ------------------------------------------------------------------ 工具实现

def _register_and_format(rows: list[dict], ctx: ToolContext) -> tuple[str, list[dict]]:
    """把章节原文登记到会话的命中池，并拼成带编号的文本。

    编号全局唯一（按 chunk_id 去重），同一段原文只贴一次。
    重复命中时只返回提示——否则第二次查询会把前面贴过的原文再贴一遍，
    模型看到的上下文里同一段话出现两次，它引用时也会跟着混乱。
    """
    blocks: list[str] = []
    new_hits: list[dict] = []
    for row in rows:
        meta = row.get("metadata") or {}
        hit = {
            "id": row.get("id"),
            "text": row.get("text", ""),
            "similarity": None,
            "metadata": meta,
        }
        number = ctx.register(hit)
        if number in ctx.printed:
            continue
        ctx.printed.add(number)
        new_hits.append(hit)
        blocks.append(
            f"[{number}] {meta.get('drug', '?')} · {meta.get('section', '?')}\n"
            f"{row.get('text', '')}\n"
            f"来源：{meta.get('source', '未知')}"
        )

    if not blocks:
        return "（这些章节在前面的调用里已经列出，编号不变，可直接引用。）", []
    return "\n\n".join(blocks), new_hits


def search_drug_label(args: dict, ctx: ToolContext) -> ToolResult:
    drug = (args.get("drug") or "").strip()
    section = (args.get("section") or "").strip()

    if not drug:
        return ToolResult("没有提供药名。", error=True)

    # 先过门控：库内没有的药要明确告知，不能让模型拿相近的药糊弄
    from app.rag.guards import resolve_drugs

    match = resolve_drugs(drug, ctx.supported)
    if not match.matched:
        if match.unsupported:
            return ToolResult(
                f"知识库未收录「{'、'.join(match.unsupported)}」，无法提供该药资料。",
                error=True,
            )
        return ToolResult(
            f"无法识别「{drug}」这个药名。可以调用 list_available_drugs 查看可查询的药品。",
            error=True,
        )

    target = match.matched[0]
    where: dict = {"drug": target}
    if section:
        # 章节做包含匹配：人工语料里有「注意事项-肝功能」这类细分章节
        where = {"$and": [{"drug": target}, {"section": {"$in": [section]}}]}

    data = ctx.collection.get(where={"drug": target}, include=["documents", "metadatas"])
    rows = [
        {"id": cid, "text": text or "", "metadata": meta or {}}
        for cid, text, meta in zip(
            data.get("ids") or [], data.get("documents") or [], data.get("metadatas") or []
        )
    ]

    # 先记下全部章节再过滤。反过来写的话，过滤后 rows 已空，
    # 「现有章节」就永远显示「无」——错在顺序，不是逻辑。
    available = sorted({r["metadata"].get("section", "") for r in rows})
    if section:
        rows = [r for r in rows if section in (r["metadata"].get("section") or "")]
        if not rows:
            return ToolResult(
                f"「{target}」没有「{section}」这个章节。该药现有章节：{'、'.join(available)}",
                error=True,
            )

    text, hits = _register_and_format(rows, ctx)
    return ToolResult(text, hits=hits)


def check_indicators(args: dict, ctx: ToolContext) -> ToolResult:
    from app.report.analyzer import review

    if ctx.report is None:
        return ToolResult("本次对话没有提供体检报告，无法核对指标。", error=True)
    if not ctx.medications:
        return ToolResult("本次对话没有提供在服药物，无法核对指标相关性。", error=True)

    wanted = [w.strip() for w in (args.get("indicators") or []) if w and w.strip()]
    result = review(ctx.report, ctx.medications, collection=ctx.collection)

    findings = result.with_evidence
    if wanted:
        findings = [
            f for f in findings if any(w in f.label or w in f.item.name_raw for w in wanted)
        ]

    if not findings:
        scope = "、".join(wanted) if wanted else "报告的异常指标"
        return ToolResult(
            f"{scope}与在服药物（{'、'.join(ctx.medications)}）的说明书之间，"
            f"没有找到相关的记载。"
        )

    blocks: list[str] = []
    hits: list[dict] = []
    for finding in findings:
        item = finding.item
        unit = f" {item.unit}" if item.unit else ""
        blocks.append(f"【{finding.label} {item.value:g}{unit}（{item.flag}）】")
        found_new = False
        for evidence in finding.evidences:
            hit = {
                "id": evidence.chunk_id,
                "text": evidence.text,
                "similarity": None,
                "metadata": {
                    "drug": evidence.drug,
                    "section": evidence.section,
                    "source": evidence.source,
                },
            }
            number = ctx.register(hit)
            hits.append(hit)
            if number in ctx.printed:
                # 同一段原文前面已经贴过：只标注编号，不再重复贴原文。
                blocks.append(f"  [{number}] {evidence.drug} · {evidence.section}（同上）")
                continue
            ctx.printed.add(number)
            found_new = True
            blocks.append(
                f"[{number}] {evidence.drug} · {evidence.section}"
                f"（命中关键词：{'、'.join(evidence.matched)}）\n    {evidence.text}"
            )
        if not found_new:
            blocks.append("  （相关说明书内容在前面已列出）")

    if result.unsupported:
        blocks.append(f"（未收录、未核对的药：{'、'.join(result.unsupported)}）")

    return ToolResult("\n".join(blocks), hits=hits)


def list_available_drugs(args: dict, ctx: ToolContext) -> ToolResult:
    drugs = sorted(ctx.supported)
    if not drugs:
        return ToolResult("知识库为空。", error=True)
    return ToolResult(f"知识库共收录 {len(drugs)} 种药：\n{'、'.join(drugs)}")


_HANDLERS = {
    "search_drug_label": search_drug_label,
    "check_indicators": check_indicators,
    "list_available_drugs": list_available_drugs,
}


def execute(tool_call: dict, ctx: ToolContext) -> ToolResult:
    """执行一次工具调用。

    参数是模型给的 JSON 字符串，解析失败要返回错误信息而不是抛异常——
    抛异常会让整个循环中断，而模型看到错误后其实可以自己纠正重试。
    """
    function = tool_call.get("function") or {}
    name = function.get("name", "")
    raw_args = function.get("arguments") or "{}"

    handler = _HANDLERS.get(name)
    if handler is None:
        return ToolResult(f"不存在名为「{name}」的工具。", error=True)

    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except json.JSONDecodeError:
        return ToolResult(f"参数不是合法 JSON：{raw_args[:100]}", error=True)

    try:
        return handler(args, ctx)
    except Exception as exc:  # noqa: BLE001 —— 工具内部异常要让模型知道，而不是炸掉整个循环
        return ToolResult(f"工具执行失败：{type(exc).__name__}: {exc}", error=True)
