"""检索门控的单元测试。

前两组是纯函数测试，不需要 Ollama，跑得很快。
第三组是集成测试，需要 Ollama 在跑且向量库已建好，条件不满足时自动跳过。
"""

import httpx
import pytest

from app.rag.guards import UNSUPPORTED_DRUGS, resolve_drugs

# --------------------------------------------------------------- 实体识别

@pytest.mark.parametrize(
    "query, expected_drug",
    [
        ("华法林的禁忌是什么", "华法林"),
        ("我吃的是华法林，能和别的药一起用吗", "华法林"),  # 药名在句子中间
        ("华法令有什么禁忌", "华法林"),  # 别名
        ("warfarin 的相互作用", "华法林"),  # 英文名
        ("阿托伐他汀对肝功能的影响", "阿托伐他汀"),
        ("二甲双胍", "二甲双胍"),
    ],
)
def test_resolve_known_drug(query, expected_drug):
    """能认出库里的药，包括别名和英文名。"""
    match = resolve_drugs(query)
    assert expected_drug in match.matched
    assert match.primary == expected_drug
    assert not match.only_unknown


@pytest.mark.parametrize("drug", ["别嘌醇", "阿司匹林", "氨氯地平", "瑞舒伐他汀"])
def test_resolve_unsupported_drug(drug):
    """库里没有的药要能被识别出来，从而拒答而不是拿别的药凑答案。"""
    match = resolve_drugs(f"{drug}的禁忌是什么")

    assert match.matched == []
    assert drug in match.unsupported
    assert match.only_unknown is True


def test_resolve_mixed_drugs():
    """同时提到库内药和库外药时：库内药可答，库外药要提示未收录。"""
    match = resolve_drugs("布洛芬和阿司匹林能一起吃吗")

    assert match.matched == ["布洛芬"]
    assert match.unsupported == ["阿司匹林"]
    # 有库内药，所以不是 only_unknown；但 has_unknown 为真，回答时要提醒
    assert match.only_unknown is False
    assert match.has_unknown is True


def test_resolve_no_drug():
    """没提药名时不该硬认。"""
    match = resolve_drugs("今天杭州天气怎么样")

    assert match.matched == []
    assert match.unsupported == []


def test_no_duplicate_counting():
    """库内药名不应同时被算成「未收录」。"""
    match = resolve_drugs("华法林的禁忌")
    assert match.unsupported == []


def test_unsupported_list_has_no_overlap():
    """未收录列表不能和库内药品重名，否则会出现「既收录又未收录」的矛盾。"""
    from app.rag.guards import DRUG_ALIASES

    known = set(DRUG_ALIASES)
    assert known.isdisjoint(UNSUPPORTED_DRUGS)


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
def test_retrieve_rejects_unsupported_drug():
    """库外药品必须拒答，而且不能返回任何片段（否则下游可能拿去用）。"""
    from app.rag.retriever import retrieve

    result = retrieve("别嘌醇的禁忌是什么")

    assert result.answerable is False
    assert "未收录" in result.reason
    assert result.hits == []


@requires_ollama
def test_retrieve_accepts_known_drug():
    """库内药品正常放行，且命中正确的药品。"""
    from app.rag.retriever import retrieve

    result = retrieve("华法林的禁忌是什么", top_k=3)

    assert result.answerable is True
    assert result.drug == "华法林"
    assert result.hits[0]["metadata"]["drug"] == "华法林"


@requires_ollama
def test_retrieve_rejects_unrelated_topic():
    """完全跑题的问题靠阈值挡掉。"""
    from app.rag.retriever import retrieve

    result = retrieve("今天杭州天气怎么样")

    assert result.answerable is False
    assert "相关度过低" in result.reason
