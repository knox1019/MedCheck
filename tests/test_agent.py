"""Agent 工具与编排循环的测试。

不依赖真实模型：编排放一个假 client，按预设顺序返回回复。
不依赖真实向量库：用内存集合。

重点测三件事：
    1. 工具参数异常时的行为（模型给错参数很常见，不能炸）
    2. 工具编号是全局递增的（答案里的 [3] 要能找回第 3 条）
    3. 轮数用尽时给出可解释的结果，而不是空字符串
"""

import json
import uuid

import chromadb
import pytest

from app.agent import loop as loop_module
from app.agent.loop import AgentResult, build_user_message, run
from app.agent.tools import TOOLS, ToolContext, execute
from app.ingest.models import LabItem, Report
from app.llm.client import ChatReply


@pytest.fixture
def collection():
    """两味药的小库，内容摘自真实语料。"""
    client = chromadb.EphemeralClient()
    col = client.create_collection(
        name=f"agent_{uuid.uuid4().hex[:12]}", metadata={"hnsw:space": "cosine"}
    )
    rows = [
        ("c1", "华法林【禁忌】肝肾功能损害、严重高血压、凝血功能障碍者禁用。",
         {"drug": "华法林", "section": "禁忌", "source": "华法林钠片说明书"}),
        ("c2", "华法林【注意事项】用药期间应监测 INR，注意观察有无出血征象。",
         {"drug": "华法林", "section": "注意事项", "source": "华法林钠片说明书"}),
        ("c3", "阿司匹林【禁忌】活动性消化性溃疡、消化道出血者禁用。",
         {"drug": "阿司匹林", "section": "禁忌", "source": "阿司匹林肠溶片说明书"}),
        ("c4", "阿托伐他汀【注意事项】用药期间应复查肝功能，转氨酶持续升高超过正常上限3倍应减量。",
         {"drug": "阿托伐他汀", "section": "注意事项", "source": "阿托伐他汀钙片说明书"}),
    ]
    col.add(
        ids=[r[0] for r in rows],
        embeddings=[[1.0, 0.0] for _ in rows],
        documents=[r[1] for r in rows],
        metadatas=[r[2] for r in rows],
    )
    return col


@pytest.fixture
def ctx(collection):
    report = Report(
        report_id="t-001",
        exam_date="2026-10-22",
        items=[
            LabItem(name_raw="丙氨酸氨基转移酶 ALT", value=68, unit="U/L",
                    ref_low=9, ref_high=50, flag="high"),
            LabItem(name_raw="尿酸 UA", value=486, unit="μmol/L",
                    ref_low=208, ref_high=428, flag="high"),
        ],
    )
    return ToolContext(report=report, medications=["阿托伐他汀"], collection=collection)


# ------------------------------------------------------------------ 工具定义

def test_tool_schema_is_valid():
    """工具定义要是合法的 OpenAI function schema。

    写错字段名不会报错，只会让模型看不见这个工具——
    排查起来非常费劲，所以用测试固定住结构。
    """
    names = set()
    for tool in TOOLS:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
        names.add(fn["name"])

    assert names == {"search_drug_label", "check_indicators", "list_available_drugs"}


def test_tool_descriptions_state_when_to_use():
    """描述里要说「什么时候用」，而不是「这个函数做什么」。

    模型完全靠 description 决定调不调用。
    """
    for tool in TOOLS:
        assert "时使用" in tool["function"]["description"]


# ------------------------------------------------------------------ 工具执行

def _call(name: str, args: dict | str) -> dict:
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": args if isinstance(args, str) else json.dumps(args),
        },
    }


def test_search_returns_numbered_sections(ctx):
    result = execute(_call("search_drug_label", {"drug": "华法林"}), ctx)

    assert not result.error
    assert "[1]" in result.content and "[2]" in result.content
    assert "监测 INR" in result.content
    assert len(result.hits) == 2


def test_tool_numbers_are_global_across_calls(ctx):
    """编号要跨工具调用累加，不能每次从 1 开始。

    否则答案里写「见 [2]」，上层会映射到第二次调用的第 2 条，
    而模型实际指的是第一次调用的内容。
    """
    execute(_call("search_drug_label", {"drug": "华法林"}), ctx)
    second = execute(_call("search_drug_label", {"drug": "阿司匹林"}), ctx)

    assert "[3]" in second.content  # 承接前面的 [1][2]
    assert len(ctx.hits) == 3


def test_search_unknown_drug_reports_instead_of_guessing(ctx):
    """库外药要明确说未收录，不能拿别的药顶上。"""
    result = execute(_call("search_drug_label", {"drug": "别嘌醇的禁忌"}), ctx)

    assert result.error
    assert "未收录" in result.content


def test_search_section_filter(ctx):
    result = execute(_call("search_drug_label", {"drug": "华法林", "section": "禁忌"}), ctx)

    assert "凝血功能障碍" in result.content
    assert "监测 INR" not in result.content


def test_search_missing_section_tells_available(ctx):
    result = execute(_call("search_drug_label", {"drug": "阿司匹林", "section": "药物相互作用"}), ctx)

    assert result.error
    assert "禁忌" in result.content  # 告知现有章节


def test_check_indicators_requires_report(collection):
    bare = ToolContext(collection=collection)
    result = execute(_call("check_indicators", {}), bare)

    assert result.error
    assert "没有提供体检报告" in result.content


def test_check_indicators_requires_medications(collection, ctx):
    ctx.medications = []
    result = execute(_call("check_indicators", {}), ctx)

    assert result.error
    assert "没有提供在服药物" in result.content


def test_check_indicators_finds_alt(ctx):
    result = execute(_call("check_indicators", {"indicators": ["谷丙转氨酶"]}), ctx)

    assert not result.error
    assert "转氨酶" in result.content
    assert result.hits


def test_list_available_drugs(ctx):
    result = execute(_call("list_available_drugs", {}), ctx)

    assert "华法林" in result.content
    assert "阿司匹林" in result.content


def test_unknown_tool_returns_error_not_exception(ctx):
    result = execute(_call("no_such_tool", {}), ctx)
    assert result.error
    assert "不存在" in result.content


def test_bad_json_arguments_returns_error(ctx):
    """模型给的参数不是合法 JSON 时，要报错给模型让它重试，而不是抛异常中断循环。"""
    result = execute(_call("search_drug_label", "{不是json"), ctx)

    assert result.error
    assert "合法 JSON" in result.content


# ------------------------------------------------------------------ 编排循环

class _FakeClient:
    """按预设顺序返回回复的假客户端。"""

    def __init__(self, replies: list[ChatReply]) -> None:
        self.replies = list(replies)
        self.seen_tools: list[list[dict] | None] = []

    def complete(self, messages, tools=None):
        self.seen_tools.append(tools)
        return self.replies.pop(0)


def test_loop_answers_directly_when_no_tool_needed(ctx):
    """模型不调工具也能直接回答——不是每个问题都要查资料。"""
    client = _FakeClient([ChatReply(content="你好，请问有什么可以帮你？")])
    result = run("你好", ctx, client=client)

    assert result.answer.startswith("你好")
    assert result.steps == []
    assert result.stopped == "done"
    assert client.seen_tools[0]  # 工具列表要每次都传


def test_loop_executes_tool_then_answers(ctx):
    client = _FakeClient([
        ChatReply(content="", tool_calls=(_call("search_drug_label", {"drug": "华法林"}),)),
        ChatReply(content="华法林需监测 INR [1]。"),
    ])
    result = run("华法林要注意什么", ctx, client=client)

    assert len(result.steps) == 1
    assert result.steps[0].tool == "search_drug_label"
    assert result.answer == "华法林需监测 INR [1]。"
    assert result.cited is True

    # 不能假定 Chroma get() 的返回顺序——[1] 对应的是「本次运行的第一条命中」，
    # 所以要拿 ctx.hits 对照，而不是写死某个章节名。
    assert result.citations[0].section == ctx.hits[0]["metadata"]["section"]


def test_loop_maps_citations_by_global_number(ctx):
    """答案引用的编号要映射回对应那条，不是按顺序猜。

    这里故意引用 [2] 而不是 [1]——如果实现是「按顺序取第一条」，
    断言就会失败。
    """
    client = _FakeClient([
        ChatReply(content="", tool_calls=(_call("search_drug_label", {"drug": "华法林"}),)),
        ChatReply(content="见 [2]。"),
    ])
    result = run("华法林用药注意", ctx, client=client)

    assert len(result.citations) == 1
    assert result.citations[0].section == ctx.hits[1]["metadata"]["section"]


def test_loop_detects_refusal(ctx):
    client = _FakeClient([ChatReply(content="资料中未提及")])
    result = run("华法林是饭前吃还是饭后吃", ctx, client=client)

    assert result.refused is True


def test_loop_stops_at_max_turns(ctx):
    """模型一直调工具不停时，要给出可解释的结果，不能返回空字符串。

    空字符串会被上层当成「拒答」，掩盖真正的原因是「没收敛」。
    """
    endless = [
        ChatReply(content="", tool_calls=(_call("list_available_drugs", {}),))
        for _ in range(5)
    ]
    client = _FakeClient(endless)
    result = run("一直查", ctx, client=client, max_turns=3)

    assert result.stopped == "max_turns"
    assert result.answer  # 不是空的
    assert "轮工具调用上限" in result.answer
    assert len(result.steps) == 3


def test_loop_feeds_tool_error_back_to_model(ctx):
    """工具报错要喂回给模型，让它自己纠正——而不是中断。"""
    client = _FakeClient([
        ChatReply(content="", tool_calls=(_call("search_drug_label", {"drug": "别嘌醇"}),)),
        ChatReply(content="知识库里没有别嘌醇的资料。"),
    ])
    result = run("别嘌醇的禁忌", ctx, client=client)

    assert result.steps[0].error is True
    assert "别嘌醇" in result.answer


def test_user_message_includes_report_and_medications(ctx):
    text = build_user_message("要紧吗", ctx)

    assert "丙氨酸氨基转移酶 ALT 68" in text
    assert "异常" in text
    assert "在服药物" in text
    assert "阿托伐他汀" in text


def test_user_message_skips_context_when_absent(collection):
    bare = ToolContext(collection=collection)
    text = build_user_message("华法林的禁忌", bare)

    assert "体检报告" not in text
    assert "在服药物" not in text
    assert "华法林的禁忌" in text


def test_loop_treats_unavailable_drug_as_refusal(ctx):
    """工具说「未收录」也算拒答，不能只认「资料中未提及」一个短语。

    实测漏判过一次：答案「知识库未收录『格列本脲』，无法提供该药的禁忌资料」
    被判成「作答」，评测显示失败——其实模型行为完全正确。
    Agent 会转述工具返回的原文，措辞比固定流程多样得多。
    """
    client = _FakeClient([
        ChatReply(content="", tool_calls=(_call("search_drug_label", {"drug": "别嘌醇"}),)),
        ChatReply(content="知识库未收录「别嘌醇」，无法提供该药的禁忌资料。"),
    ])
    result = run("别嘌醇的禁忌", ctx, client=client)

    assert result.refused is True
    assert result.citations == []


def test_loop_treats_no_evidence_from_tools_as_refusal(ctx):
    """调了工具但一条资料都没拿到，必然是拒答——不管它用了什么措辞。"""
    client = _FakeClient([
        ChatReply(content="", tool_calls=(_call("search_drug_label", {"drug": "别嘌醇"}),)),
        ChatReply(content="这个药我查不到。"),
    ])
    result = run("别嘌醇的禁忌", ctx, client=client)

    assert result.refused is True


def test_loop_treats_off_topic_reply_as_refusal(ctx):
    """没调工具的跑题问题，靠拒绝性措辞识别。"""
    client = _FakeClient([ChatReply(content="我只能回答药品相关问题，无法回答天气问题。")])
    result = run("今天杭州天气怎么样", ctx, client=client)

    assert result.refused is True


def test_loop_does_not_mark_cautious_answer_as_refusal(ctx):
    """给出实质内容但措辞谨慎（「资料不足以判断因果关系」）不算拒答。

    这类答案有依据、有引用，只是对因果性保持克制——
    判成拒答会让「克制」和「不会答」混为一谈。
    """
    client = _FakeClient([
        ChatReply(content="", tool_calls=(_call("search_drug_label", {"drug": "华法林"}),)),
        ChatReply(content="说明书要求监测 INR [1]，但资料不足以判断因果关系。"),
    ])
    result = run("华法林要注意什么", ctx, client=client)

    assert result.refused is False
    assert result.cited is True


def test_loop_does_not_treat_partial_caveat_as_refusal(ctx):
    """「作答 + 补一句某点未提及」不算拒答。

    实测踩过：答案「会。根据说明书……诱发消化道溃疡出血[1]。
    资料中未提及短期规范用药是否会损伤胃。」被判成拒答——
    主体明明答了，只是对边界保持克制。
    """
    client = _FakeClient([
        ChatReply(content="", tool_calls=(_call("search_drug_label", {"drug": "阿司匹林"}),)),
        ChatReply(
            content="会。长期大量用药可诱发消化道溃疡出血[1]。"
                    "资料中未提及短期规范用药是否会损伤胃。"
        ),
    ])
    result = run("阿司匹林会不会把胃吃坏", ctx, client=client)

    assert result.refused is False
    assert result.cited is True


def test_loop_treats_capability_disclaimer_as_refusal(ctx):
    """寒暄式问题的能力外声明算拒答（没调工具、没资料的情况下）。"""
    client = _FakeClient([ChatReply(content="这属于天气问题，我无法回答，只能查药品说明书。")])
    result = run("今天天气怎么样", ctx, client=client)

    assert result.refused is True


def test_loop_answers_greeting_as_not_refusal(ctx):
    """寒暄没有调用工具、也没有资料，但那是正面回答，不是拒答。"""
    client = _FakeClient([
        ChatReply(content="你好！我可以帮你查药品说明书、核对体检指标与用药的关联。")
    ])
    result = run("你好，你能做什么", ctx, client=client)

    assert result.refused is False
