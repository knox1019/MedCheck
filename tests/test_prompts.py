"""提示词组装与解析的单元测试。

这一组全部是纯函数，不需要 Ollama，跑得很快。
它们守的是提示词契约——一旦有人改坏了 REFUSAL_PHRASE 或资料编号格式，
生成层的拒答就整体失效，而这些测试会立刻变红。
"""

import pytest

from app.llm.prompts import (
    REFUSAL_PHRASE,
    SYSTEM_PROMPT,
    build_context,
    build_messages,
    detect_refusal,
    extract_cited_numbers,
)

# ------------------------------------------------------------ 上下文拼装

def make_hit(drug="华法林", section="禁忌", source="华法林钠片说明书", text="下列情况禁用……"):
    return {
        "id": f"{drug}-{section}",
        "text": f"{drug}【{section}】{text}",
        "metadata": {"drug": drug, "section": section, "source": source},
        "similarity": 0.69,
    }


def test_build_context_numbering():
    """资料编号必须从 1 开始且连续，模型才对得上 [1][2]。"""
    context, index = build_context([make_hit(), make_hit(section="注意事项")])

    assert "【资料 1】" in context
    assert "【资料 2】" in context
    assert set(index) == {1, 2}


def test_build_context_includes_source():
    """出处要写进资料块，模型才能知道该引哪个。"""
    context, _ = build_context([make_hit()])

    assert "华法林钠片说明书" in context
    assert "禁忌" in context


def test_build_context_index_maps_back_to_hit():
    """编号能映射回原始 hit，答案里的 [n] 才有意义。"""
    hits = [make_hit(), make_hit(section="注意事项")]
    _, index = build_context(hits)

    assert index[1] is hits[0]
    assert index[2] is hits[1]


def test_build_context_empty():
    context, index = build_context([])
    assert context == ""
    assert index == {}


def test_build_messages_structure():
    """两条消息：system 定规则，user 给资料和问题。"""
    messages = build_messages("华法林的禁忌是什么", "【资料 1】……")

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "华法林的禁忌是什么" in messages[1]["content"]


def test_system_prompt_forbids_external_knowledge():
    """系统提示必须明确禁止模型用自己的医学知识补充，这是医疗 RAG 的底线。"""
    assert "不得使用你自己的医学知识" in SYSTEM_PROMPT
    assert REFUSAL_PHRASE in SYSTEM_PROMPT


# ------------------------------------------------------------ 拒答判定

def test_detect_refusal_exact():
    assert detect_refusal(REFUSAL_PHRASE) is True


def test_detect_refusal_with_extra_text():
    """模型多说几个字也应识别为拒答。"""
    assert detect_refusal(f"{REFUSAL_PHRASE}，建议咨询医师。") is True


def test_detect_refusal_negative():
    assert detect_refusal("阿司匹林可引起消化道溃疡出血 [1]。") is False


# ------------------------------------------------------------ 引用抽取

@pytest.mark.parametrize(
    "text, expected",
    [
        ("禁用情况见 [1]，监测要求见 [2]。", [1, 2]),
        ("见 [2] 和 [1]。", [2, 1]),  # 按出现顺序，不按大小
        ("多处引用同一来源 [1][1][2]。", [1, 2]),  # 去重保序
        ("没有引用标记。", []),
        ("", []),
        ("编号超出范围 [9]。", [9]),  # 越界由调用方过滤
    ],
)
def test_extract_cited_numbers(text, expected):
    assert extract_cited_numbers(text) == expected


# ------------------------------------------------------------ 集成测试（需 Ollama）

def _ollama_available() -> bool:
    try:
        import httpx

        return httpx.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(), reason="需要 Ollama 服务在运行"
)


@requires_ollama
def test_qa_refuses_unsupported_drug():
    """库外药品在检索层就该拒答，不应调用模型。"""
    from app.llm.qa import answer

    result = answer("别嘌醇的禁忌是什么")

    assert result.refused is True
    assert "未收录" in result.reason
    assert result.text == ""


@requires_ollama
def test_qa_refuses_when_corpus_lacks_info():
    """语料里没有的信息，必须由模型判断后拒答——这是 W4 留下的缺口。"""
    from app.llm.qa import answer

    result = answer("华法林是饭前吃还是饭后吃", top_k=3)

    assert result.refused is True
    assert REFUSAL_PHRASE in result.text
