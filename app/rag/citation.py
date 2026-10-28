"""引用溯源：让每条检索结果都能说出「这话出自哪里」。

为什么要单独建这个模块：

    RAG 系统最怕的不是答不出，而是答得斩钉截铁却查不到出处。
    医疗场景尤其如此——用户看到「华法林有出血风险」，必须能点开看到
    这句话来自哪份说明书的哪个章节。

    所以溯源不是展示层的美化，而是数据层的硬要求：
    检索结果里必须带着出处，一路传到最终答案。

两类出处需要区分：

    药品说明书   出处是「文件 + 章节」，说明书本身没有页码（我们按章节切分的）
    体检报告     出处是「文件 + 页码」，取自 PDF 解析阶段记录的 page 字段

    这两者不能混为一谈。Citation 用 section 和 page 两个字段分别承载。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    """一条出处。

    frozen=True 表示创建后不可修改，可以安全地放进集合去重，
    也避免下游代码意外改掉出处信息。
    """

    source: str  # 出处文档名，如「华法林钠片说明书」
    section: str | None = None  # 章节，如「禁忌」
    chunk_id: str | None = None  # 片段 id，便于定位到具体一条
    page: int | None = None  # 页码，仅体检报告类来源有

    @property
    def label(self) -> str:
        """给人看的一行出处，如「华法林钠片说明书 · 禁忌」。"""
        parts = [self.source]
        if self.section:
            parts.append(self.section)
        if self.page is not None:
            parts.append(f"第 {self.page} 页")
        return " · ".join(parts)

    @property
    def key(self) -> tuple:
        """去重用的键。同一文件同一章节的多个片段视为同一处出处。"""
        return (self.source, self.section, self.page)


def citations_from_hits(hits: list[dict]) -> list[Citation]:
    """从检索结果里提取出处，按文档顺序去重。

    为什么按 key 去重而不是按 chunk_id：
        用户看到「华法林钠片说明书 · 禁忌」就够了，不需要知道
        这是该章节的第 1 段还是第 3 段。chunk_id 保留在对象里，
        供开发和调试使用。
    """
    seen: set[tuple] = set()
    result: list[Citation] = []

    for hit in hits:
        meta = hit.get("metadata") or {}
        citation = Citation(
            source=meta.get("source", "未知来源"),
            section=meta.get("section"),
            chunk_id=hit.get("id"),
        )
        if citation.key in seen:
            continue
        seen.add(citation.key)
        result.append(citation)

    return result


def citation_from_lab_item(item, source: str) -> Citation:
    """把一条体检指标转成出处（报告类来源用页码定位）。

    参数：
        item     app.ingest.models.LabItem
        source   报告文件名或报告编号
    """
    return Citation(
        source=source,
        section=item.name_raw,
        page=item.page,
    )
