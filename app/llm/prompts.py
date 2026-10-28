"""提示词与上下文拼装。

设计要点：

    1. 用一个**固定短语**作为「资料不足」的契约：
           REFUSAL_PHRASE = "资料中未提及"
       让模型在没找到依据时只输出这一句。
       好处是检测变成确定性的字符串包含判断，不需要再调一次模型去判别。

    2. 资料编号用 [1][2][3]，和答案里的引用标记一一对应。
       编号对应的是**检索结果的顺序**，不是去重后的出处顺序——
       因为同一章节可能有多个片段命中，模型看到的是片段，不是章节。

    3. 明确禁止模型用自己的医学知识补充。
       这是 RAG 在医疗场景的底线：宁可不答，也不能凭训练记忆编。
"""

import re

# 「资料不足」的固定应答。改这个常量要同步改 detect_refusal 的调用方。
REFUSAL_PHRASE = "资料中未提及"

SYSTEM_PROMPT = f"""你是医学资料检索助手。必须严格遵守以下规则：

1. 只能依据【资料】中的内容回答，不得使用你自己的医学知识补充或推测。
2. 每条结论后面必须标注来源编号，格式如 [1]、[2]。
3. 判断资料够不够用，看的是「能否据此回答问题」，**不是「原文有没有写出问题里的结论」**。
   有些问题需要从资料推一步：
     · 问「什么药不能一起吃」，资料写「某药可增强本品抗凝作用」→ 能回答
     · 问「会不会伤肝」，资料写「用药期间应复查肝功能、转氨酶升高需减量」→ 能回答
   不要因为原文没有直接出现「不能」「会」这类字眼就判为未提及。
4. 但上一条不等于可以勉强作答。如果资料讲的是**另一件事**，
   例如问服药时间而资料只讲监测指标、问忌口而资料只讲不良反应，
   仍然只输出这一句：{REFUSAL_PHRASE}
5. 不要给出用药剂量调整建议，不要下诊断结论。
6. 用简洁的中文回答，控制在 200 字以内。"""


def build_context(hits: list[dict]) -> tuple[str, dict[int, dict]]:
    """把检索结果拼成带编号的资料块。

    返回：
        context  给模型看的资料文本
        index    编号 → 该条 hit，用于把答案里的 [n] 映射回出处
    """
    blocks: list[str] = []
    index: dict[int, dict] = {}

    for number, hit in enumerate(hits, start=1):
        meta = hit.get("metadata") or {}
        source = meta.get("source", "未知来源")
        section = meta.get("section", "")
        header = f"【资料 {number}】来源：{source}"
        if section:
            header += f" · {section}"
        blocks.append(f"{header}\n{hit['text']}")
        index[number] = hit

    return "\n\n".join(blocks), index


def build_messages(question: str, context: str) -> list[dict]:
    """组装成 OpenAI 格式的消息列表。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"【资料】\n{context}\n\n【问题】\n{question}",
        },
    ]


def detect_refusal(answer_text: str) -> bool:
    """判断模型的回答是不是「资料不足」。

    用固定短语做契约，所以这里是确定性的字符串判断，
    不依赖模型输出格式的稳定性——这对小模型尤其重要。
    """
    return REFUSAL_PHRASE in answer_text


def extract_cited_numbers(answer_text: str) -> list[int]:
    """从答案里抽出引用编号，按出现顺序去重。

    >>> extract_cited_numbers("禁用情况见 [1]，监测要求见 [2]。")
    [1, 2]
    """
    numbers: list[int] = []
    for match in re.finditer(r"\[(\d+)\]", answer_text):
        number = int(match.group(1))
        if number not in numbers:
            numbers.append(number)
    return numbers
