"""问答编排：检索 → 组装上下文 → 大模型生成 → 解析答案与出处。

这是 W5 的主链路，也是整个项目对外表现的那一层。

三道拒绝关卡（前两道在检索层，第三道在这里）：

    ① 实体门控    问题里的药不在库里 → 拒答（retriever）
    ② 相关度阈值  检索结果太不相关   → 拒答（retriever）
    ③ 资料覆盖判断 资料里没有答案     → 拒答（本模块，由模型判断）

第三道是 W4 结束时留下的缺口：问「华法林是饭前吃还是饭后吃」，
检索层会把「注意事项：监测 INR」当成相关答案放行，因为字面相似度够高。
只有让模型看过资料后说「资料中未提及」，才能堵住这个漏洞。
"""

from dataclasses import dataclass, field

from app.llm.client import ChatClient, build_chat_client
from app.llm.prompts import (
    REFUSAL_PHRASE,
    build_context,
    build_messages,
    detect_refusal,
    extract_cited_numbers,
)
from app.rag.citation import Citation
from app.rag.retriever import retrieve


@dataclass
class Answer:
    """一次问答的完整结果。"""

    question: str
    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    hits: list[dict] = field(default_factory=list)
    refused: bool = False
    reason: str = ""
    # 答案里是否真的出现了 [1] 这样的引用标记。
    # 模型忘了标引用时这里为 False，是可监控的质量信号。
    cited: bool = False


def _citation_for_hit(hit: dict) -> Citation:
    """把一条检索结果转成出处。"""
    meta = hit.get("metadata") or {}
    return Citation(
        source=meta.get("source", "未知来源"),
        section=meta.get("section"),
        chunk_id=hit.get("id"),
    )


def answer(
    question: str,
    top_k: int = 4,
    use_rerank: bool = True,
    client: ChatClient | None = None,
) -> Answer:
    """回答问题。检索不到或资料不足时返回 refused=True。"""
    result = retrieve(question, top_k=top_k, use_rerank=use_rerank)

    # 检索层就已拒答：不调模型，省一次调用也避免模型瞎编
    if not result.answerable:
        return Answer(
            question=question,
            refused=True,
            reason=result.reason,
            hits=result.hits,
        )

    context, index = build_context(result.hits)
    messages = build_messages(question, context)

    client = client or build_chat_client()
    text = client.chat(messages)

    # 模型判断资料不足：如实拒答，且不给出处
    if detect_refusal(text):
        return Answer(
            question=question,
            text=REFUSAL_PHRASE,
            refused=True,
            reason="资料中未提及问题所涉及的内容",
            hits=result.hits,
        )

    # 把答案里的 [n] 映射回出处；模型没标引用时退化为「全部候选的出处」
    cited_numbers = [n for n in extract_cited_numbers(text) if n in index]
    if cited_numbers:
        citations = [_citation_for_hit(index[n]) for n in cited_numbers]
        cited = True
    else:
        citations = [_citation_for_hit(hit) for hit in result.hits]
        cited = False

    return Answer(
        question=question,
        text=text,
        citations=citations,
        hits=result.hits,
        refused=False,
        reason=result.reason,
        cited=cited,
    )
