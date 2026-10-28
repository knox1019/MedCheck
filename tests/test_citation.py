"""引用溯源的单元测试。

前两组是纯函数测试（不需要 Ollama），第三组是集成测试。
"""

import httpx
import pytest

from app.rag.citation import Citation, citation_from_lab_item, citations_from_hits

# ------------------------------------------------------------ Citation 本身

def test_label_with_section():
    citation = Citation(source="华法林钠片说明书", section="禁忌")
    assert citation.label == "华法林钠片说明书 · 禁忌"


def test_label_with_page():
    """报告类来源用页码定位，不带章节。"""
    citation = Citation(source="report_demo.pdf", section="丙氨酸氨基转移酶 ALT", page=2)
    assert citation.label == "report_demo.pdf · 丙氨酸氨基转移酶 ALT · 第 2 页"


def test_label_source_only():
    citation = Citation(source="某说明书")
    assert citation.label == "某说明书"


def test_key_ignores_chunk_id():
    """同一文件同一章节的不同片段，应视为同一处出处。"""
    a = Citation(source="说明书A", section="禁忌", chunk_id="drug-001-禁忌-1")
    b = Citation(source="说明书A", section="禁忌", chunk_id="drug-001-禁忌-2")
    assert a.key == b.key


def test_key_distinguishes_section():
    a = Citation(source="说明书A", section="禁忌")
    b = Citation(source="说明书A", section="注意事项")
    assert a.key != b.key


def test_citation_is_immutable():
    """出处创建后不应被下游改动。"""
    citation = Citation(source="X")
    with pytest.raises(Exception):
        citation.source = "Y"


# --------------------------------------------------- 从检索结果提取出处

def make_hit(drug="华法林", section="禁忌", source="华法林钠片说明书", chunk_id="drug-001-禁忌"):
    return {
        "id": chunk_id,
        "text": "……",
        "metadata": {"drug": drug, "section": section, "source": source},
        "similarity": 0.69,
    }


def test_citations_dedup_same_section():
    """同一章节的两个片段只保留一条出处。"""
    hits = [
        make_hit(chunk_id="drug-001-禁忌-1"),
        make_hit(chunk_id="drug-001-禁忌-2"),
        make_hit(section="注意事项", chunk_id="drug-001-注意事项"),
    ]
    citations = citations_from_hits(hits)

    assert len(citations) == 2
    assert citations[0].section == "禁忌"
    assert citations[1].section == "注意事项"


def test_citations_preserve_order():
    """出处顺序应与检索结果的排名一致，前面的更相关。"""
    hits = [make_hit(section="注意事项"), make_hit(section="禁忌")]
    citations = citations_from_hits(hits)

    assert [c.section for c in citations] == ["注意事项", "禁忌"]


def test_citations_empty_input():
    assert citations_from_hits([]) == []


def test_citations_missing_metadata_fallback():
    """元数据缺失时不能崩，用「未知来源」兜底。"""
    citations = citations_from_hits([{"id": "x", "text": "……"}])
    assert citations[0].source == "未知来源"


# ------------------------------------------------- 报告类出处

def test_citation_from_lab_item():
    """体检指标转出处：用页码定位，section 存指标名。"""
    from app.ingest.models import LabItem

    item = LabItem(name_raw="丙氨酸氨基转移酶 ALT", value=68.0, page=1)
    citation = citation_from_lab_item(item, source="report_demo.pdf")

    assert citation.page == 1
    assert citation.section == "丙氨酸氨基转移酶 ALT"
    assert "第 1 页" in citation.label


# ------------------------------------------------------------ 集成测试

def _ollama_available() -> bool:
    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(), reason="需要 Ollama 服务在运行"
)


@requires_ollama
def test_retrieve_returns_citations():
    from app.rag.retriever import retrieve

    result = retrieve("华法林的禁忌是什么", top_k=2)

    assert result.answerable is True
    assert len(result.citations) >= 1
    assert result.citations[0].source == "华法林钠片说明书"
    assert result.citations[0].section == "禁忌"


@requires_ollama
def test_refusal_has_no_citations():
    """拒答时不能给出处——否则用户会误以为「查到了但不说」。"""
    from app.rag.retriever import retrieve

    # 用真正的缺口药（语料里没有它的说明书）。别嘌醇在语料扩充后已有资料，
    # 不能再当「未收录」的例子——语料一变，测试的样例也要跟着变。
    result = retrieve("格列本脲的禁忌是什么")

    assert result.answerable is False
    assert result.citations == []
