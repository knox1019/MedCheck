"""把核对结果讲成人话。

分工：
    analyzer.py  负责「召回」——用关键词把可能相关的说明书段落都捞出来
    narrative.py 负责「判断」——让模型看原文，决定哪些真的相关、关系是什么

为什么必须分两步：
    关键词匹配只知道「这段话里出现了『胆固醇』三个字」，分不出
    「本药可引起胆固醇升高」（药是原因）和「本药用于治疗高胆固醇血症」
    （药是治疗）。实测阿托伐他汀的「胆固醇」出现在
    「因胆固醇及其生物合成产物对胎儿发育必需」——那是在解释孕妇为何禁用，
    跟血脂指标毫无关系。这一步只能靠读原文判断。

三种关系必须区分，否则会给用户错误方向的信息：

    药物可能影响该指标       说明书提到该药可引起相关异常
    药物用于治疗该指标异常   如降脂药用于高胆固醇，这是好事不是坏事
    说明书未提及             如实说没有，不猜
"""

from dataclasses import dataclass, field

from app.llm.client import ChatClient, build_chat_client
from app.llm.prompts import REFUSAL_PHRASE, extract_cited_numbers
from app.rag.citation import Citation
from app.report.analyzer import Evidence, IndicatorFinding, MedicationReview

_FLAG_TEXT = {"high": "偏高", "low": "偏低", "normal": "正常", "unknown": "无法判断"}

SYSTEM_PROMPT = """你是体检报告解读助手。系统已从报告里找出异常指标，并用关键词从药品说明书里筛出了可能的依据。
关键词筛选是粗筛，会误召回，所以要你来判断真实关系。

【三种关系，必须区分】
- 可能相关：说明书提到该药可引起该指标异常，或要求监测该指标，或规定该指标异常者禁用、减量
- 治疗作用：该药用于治疗这个指标异常（如降脂药用于高胆固醇），这是治疗目的，不是不良反应
- 不相关：这段话虽然出现了相关词，但并非在讨论该指标（例如解释孕妇为何禁用时顺带提到胆固醇）

【必须遵守】
1. 只能依据资料原文，不得使用你自己的医学知识补充或推测。
2. 指标后面若附有补充说明（如「该指标升高可表现为黄疸」），这是术语等价提示，
   用来判断原文里的相关词是否在说同一件事——请据此判断，不要忽略。
3. 每一项写完标注依据编号，格式如 [1]、[2]。
4. **每个指标都必须归入下面三类之一**，不要用其他说法：
   - A 类「可能相关」：说明书写明该药可引起相关异常、要求监测该指标、或该指标异常者禁用减量
   - B 类「治疗作用」：该药用于治疗这个指标异常（如降脂药用于高胆固醇）
   - C 类「无关」：依据讨论的是别的问题（例如解释孕妇为何禁用时顺带提到胆固醇）

   注意：只要原文涉及该指标的监测、禁忌、不良反应，就属于 A 类。
   写成 C 类时**不要标注依据编号**。

5. **每项必须先从依据原文里原样抄一句（不超过 40 字），用「」括起来**，再说明关系分类和编号。
   不许概括、不许改写、不许添加原文里没有的内容。
   例如「谷丙转氨酶：A 类。原文「转氨酶持续升高超过正常上限3倍者应减量或停药」，说明需监测该指标 [2]」。
   **不要出现「未提及」「没有提到」这类说法**——没有依据的指标不会出现在列表里。
6. 不要下诊断结论，不要建议增减剂量或停药。
7. 总字数控制在 300 字以内，超过就精简，不要重复。"""


@dataclass
class ReportNarrative:
    """报告解读结果。

    cited_numbers 和 citations 一一对应，保留答案里使用的原始编号。
    展示时不能重新从 1 编号——否则答案写「见 [2]」、出处列「[1]」，
    用户会以为映射错了。
    """

    text: str
    cited: bool = False
    citations: list[Citation] = field(default_factory=list)
    cited_numbers: list[int] = field(default_factory=list)
    refused: bool = False


def _format_ref(item) -> str:
    """把参考范围格式化成一句可读的话。"""
    low, high = item.ref_low, item.ref_high
    if low is not None and high is not None:
        return f"参考 {low:g}–{high:g}"
    if high is not None:
        return f"参考 <{high:g}"
    if low is not None:
        return f"参考 >{low:g}"
    return "无参考范围"


def build_context(review: MedicationReview) -> tuple[str, dict[int, Evidence]]:
    """拼出给模型看的上下文。

    **只放有依据的指标。** 这一条是踩出来的：

        最初把 11 项异常全列进去，让模型逐项说明。结果模型把
        「转氨酶持续升高超过正常上限3倍者应减量」读成了「说明书未提及」，
        然后把 11 项逐一念了一遍「未提及」，还重复输出两遍。

    根因是把确定性的活交给了模型。**哪些指标没匹配到依据，代码已经知道**，
    不需要模型再生成一遍。模型只该做它擅长的部分：判断这段原文到底在说什么。

    依据仍按 chunk_id 去重：三个肝功能指标经常命中同一段注意事项，
    重复贴三遍既浪费上下文也干扰模型。
    """
    with_evidence = [f for f in review.findings if f.evidences]

    number_of: dict[str, int] = {}  # chunk_id → 编号
    index: dict[int, Evidence] = {}
    rows: list[tuple[IndicatorFinding, list[int]]] = []

    for finding in with_evidence:
        numbers: list[int] = []
        for evidence in finding.evidences:
            existing = number_of.get(evidence.chunk_id)
            if existing is None:
                existing = len(index) + 1
                number_of[evidence.chunk_id] = existing
                index[existing] = evidence
            numbers.append(existing)
        rows.append((finding, numbers))

    total = len(review.findings)
    indicator_lines = [
        f"【异常指标共 {total} 项，其中 {len(with_evidence)} 项在说明书中找到相关记载】"
    ]
    for finding, numbers in rows:
        item = finding.item
        flag = _FLAG_TEXT.get(item.flag, item.flag)
        unit = f" {item.unit}" if item.unit else ""
        refs = " ".join(f"[{n}]" for n in numbers)
        # 指标后面的等价说明来自词表（如「该指标升高可表现为黄疸」），
        # 是确定知识，不是让模型推测的内容。
        note = f"；{finding.profile.note}" if finding.profile and finding.profile.note else ""
        indicator_lines.append(
            f"- {finding.label} {item.value:g}{unit}（{_format_ref(item)}）{flag}{note} —— 依据 {refs}"
        )

    evidence_lines = ["【说明书依据】"]
    for n in sorted(index):
        evidence = index[n]
        evidence_lines.append(
            f"[{n}] {evidence.drug} · {evidence.section}"
            f"（{evidence.source}）：{evidence.text}"
        )

    context = "\n".join(indicator_lines) + "\n\n" + "\n\n".join(evidence_lines)
    return context, index


def build_messages(context: str, medications: list[str]) -> list[dict]:
    drug_line = "、".join(medications) if medications else "未提供"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"【在服药物】{drug_line}\n\n{context}\n\n"
                "请只针对上面列出的指标作答，每项一行。"
            ),
        },
    ]


def _citation_for(evidence: Evidence) -> Citation:
    return Citation(
        source=evidence.source,
        section=evidence.section,
        chunk_id=evidence.chunk_id,
    )


def narrate(
    review: MedicationReview,
    client: ChatClient | None = None,
) -> ReportNarrative:
    """生成报告解读。"""
    if not review.abnormal:
        return ReportNarrative(text="报告中未发现异常指标。", refused=False)

    # 一条依据都没召回到：不调模型。省一次调用，也避免模型对着空资料硬讲。
    if not review.with_evidence:
        names = "、".join(review.medications) or "未提供药物"
        return ReportNarrative(
            text=(
                f"报告中有 {len(review.abnormal)} 项异常指标，但{names}的说明书里"
                f"没有提到与这些指标相关的内容。{REFUSAL_PHRASE}。建议携带报告咨询医生。"
            ),
            refused=True,
        )

    context, index = build_context(review)
    messages = build_messages(context, list(review.medications))

    client = client or build_chat_client()
    text = client.chat(messages)

    cited_numbers = [n for n in extract_cited_numbers(text) if n in index]
    if cited_numbers:
        citations = [_citation_for(index[n]) for n in cited_numbers]
        cited = True
    else:
        # 模型没标引用：出处退化为全部候选，并让调用方能看到 cited=False
        citations = [_citation_for(e) for e in index.values()]
        cited_numbers = list(index)
        cited = False

    return ReportNarrative(
        text=text,
        citations=citations,
        cited_numbers=cited_numbers,
        cited=cited,
        refused=False,
    )
