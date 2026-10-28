"""检索门控的单元测试。

前两组是纯函数测试，不需要 Ollama，跑得很快。
第三组是集成测试，需要 Ollama 在跑且向量库已建好，条件不满足时自动跳过。

注意 resolve_drugs 的第二个参数 supported：它代表「库里实际收录了哪些药」，
由 retriever 从向量库元数据读出。测试里显式传入，是为了不依赖真实索引。
"""

import httpx
import pytest

from app.rag.guards import DRUG_ALIASES, resolve_drugs

# 测试用的「库内药品集合」，取别名表里的全部药名
ALL_DRUGS = set(DRUG_ALIASES)


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
        ("吗丁啉能吃吗", "多潘立酮"),  # 商品名
        ("优甲乐什么时候吃", "左甲状腺素钠"),  # 商品名
        ("扑热息痛和布洛芬", "对乙酰氨基酚"),  # 别名，取第一个命中
    ],
)
def test_resolve_known_drug(query, expected_drug):
    """能认出药名，包括别名、商品名和英文名。"""
    match = resolve_drugs(query, ALL_DRUGS)
    assert expected_drug in match.matched
    assert match.only_unknown is False


def test_resolve_classifies_by_supported():
    """同一个药名，是否算「库内」取决于 supported 参数。

    这是修复过一次的 bug：早期把「库内药品」硬编码在代码里，
    语料扩充后阿司匹林已经入库，却仍被判为「未收录」而遭拒答。
    现在改为由调用方传入真实的库内集合。
    """
    query = "阿司匹林的禁忌"

    in_library = resolve_drugs(query, {"阿司匹林"})
    assert in_library.matched == ["阿司匹林"]
    assert in_library.only_unknown is False

    not_in_library = resolve_drugs(query, {"华法林"})
    assert not_in_library.matched == []
    assert not_in_library.unsupported == ["阿司匹林"]
    assert not_in_library.only_unknown is True


@pytest.mark.parametrize("drug", ["别嘌醇", "非布司他", "辛伐他汀", "格列美脲", "缬沙坦"])
def test_resolve_unsupported_drug(drug):
    """库里没有的药要能被识别出来，从而拒答而不是拿别的药凑答案。"""
    match = resolve_drugs(f"{drug}的禁忌是什么", {"华法林"})

    assert match.matched == []
    assert drug in match.unsupported
    assert match.only_unknown is True


def test_resolve_mixed_drugs():
    """同时提到库内药和库外药时：库内药可答，库外药要提示未收录。"""
    match = resolve_drugs("布洛芬和别嘌醇能一起吃吗", {"布洛芬"})

    assert match.matched == ["布洛芬"]
    assert match.unsupported == ["别嘌醇"]
    assert match.only_unknown is False
    assert match.has_unknown is True


def test_resolve_all_supported():
    """两个药都在库内时，都应该被识别为可答。"""
    match = resolve_drugs("布洛芬和阿司匹林能一起吃吗", {"布洛芬", "阿司匹林"})

    assert match.matched == ["布洛芬", "阿司匹林"]
    assert match.unsupported == []


def test_resolve_no_drug():
    """没提药名时不该硬认。"""
    match = resolve_drugs("今天杭州天气怎么样", ALL_DRUGS)

    assert match.matched == []
    assert match.unsupported == []


def test_alias_table_has_no_cross_drug_collision():
    """别名不能跨药重复，否则同一句话会识别出两个药。

    例如「氯沙坦」和「缬沙坦」如果共用别名「沙坦」，就会互相误判。
    """
    seen: dict[str, str] = {}
    collisions = []
    for drug, aliases in DRUG_ALIASES.items():
        for alias in aliases:
            if alias in seen and seen[alias] != drug:
                collisions.append((alias, seen[alias], drug))
            seen[alias] = drug
    assert collisions == []


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

    # 用真正的缺口药：语料扩充后别嘌醇已有资料，不能再当反例
    result = retrieve("格列本脲的禁忌是什么")

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


@requires_ollama
def test_supported_drugs_come_from_index():
    """库内药品集合必须来自真实索引，且非空、与语料一致。"""
    from app.core.config import get_settings
    from app.rag.retriever import supported_drugs
    from app.rag.store import get, get_client

    collection = get(get_client(get_settings().chroma_dir))
    drugs = supported_drugs(collection)

    assert len(drugs) >= 8
    assert all(isinstance(name, str) and name for name in drugs)


# ------------------------------------------------- 官方药品名称索引（663 种）

def test_drug_index_loaded():
    """官方索引要能被加载，且数量达到国家基本药物目录的量级。"""
    from app.rag.guards import load_drug_index, scan_index

    index = load_drug_index()
    assert len(index) >= 500, f"索引只有 {len(index)} 条，可能没生成"

    merged = scan_index()
    assert len(merged) >= len(index)


def test_indexed_drug_without_corpus_is_unsupported():
    """索引里有、语料里没有的药，必须判为未收录。

    这类药如果没有被识别出来，系统会拿语义最接近的别的药去作答——
    比拒答危险得多。
    """
    from app.rag.guards import resolve_drugs

    for query, drug in [
        ("青霉素的禁忌是什么", "青霉素"),
        ("六味地黄丸能长期吃吗", "六味地黄丸"),
        ("奥司他韦有什么副作用", "奥司他韦"),
    ]:
        match = resolve_drugs(query, {"华法林"})
        assert drug in match.unsupported, f"{drug} 未被识别"
        assert match.only_unknown is True


def test_substring_pruning_for_compound_drug():
    """复方制剂不能被拆成单方成分。

    「阿莫西林」是「阿莫西林克拉维酸钾」的子串。
    若不裁剪，问复方制剂会命中单方成分，然后用单方的资料作答——
    答案看起来合理，但说的不是同一个药。
    """
    from app.rag.guards import resolve_drugs

    match = resolve_drugs("阿莫西林克拉维酸钾的禁忌", {"阿莫西林"})

    assert "阿莫西林" not in match.matched
    assert "阿莫西林克拉维酸钾" in match.unsupported
    assert match.only_unknown is True


def test_substring_pruning_keeps_correct_single_drug():
    """裁剪不能误伤正常情况：单方药本身仍应被正确识别。"""
    from app.rag.guards import resolve_drugs

    match = resolve_drugs("阿莫西林的禁忌", {"阿莫西林"})
    assert match.matched == ["阿莫西林"]
