"""报告核对的单元测试。

这里测的是「召回」这一层，不涉及大模型：
    指标识别  报告上的写法 → 语料里的说法
    用药核对  在服药物 → 相关说明书片段
    上下文拼装

用内存集合（EphemeralClient）而不是真实向量库，测试才自包含——
不依赖 data/chroma 有没有建过，也不依赖 Ollama 在不在跑。
"""

import uuid

import chromadb
import pytest

from app.ingest.indicators import identify, keywords_of, matches_corpus
from app.ingest.models import LabItem, Report
from app.report.analyzer import review
from app.report.narrative import build_context


# ------------------------------------------------------------------ 指标识别

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("丙氨酸氨基转移酶 ALT", "ALT"),
        ("天门冬氨酸氨基转移酶 AST", "AST"),
        ("血红蛋白  HGB", "HGB"),
        ("高密度脂蛋白胆固醇 HDL-C", "HDL-C"),  # 不能被「胆固醇」抢走
        ("总胆固醇 TC", "TC"),
        ("白细胞计数 WBC", "WBC"),
        ("钠 Na", "NA"),  # 报告用小写，表里是大写
        ("钾 K", "K"),
        ("C 反应蛋白 CRP", "CRP"),
        ("白蛋白 ALB", "ALB"),  # 不能被 ALP 误伤
    ],
)
def test_identify_common_names(raw, expected):
    profile = identify(raw)
    assert profile is not None, f"{raw} 未识别"
    assert profile.code == expected


@pytest.mark.parametrize("raw", ["", "   ", "某个没见过的指标", "肌酸激酶 CK"])
def test_identify_returns_none_for_unknown(raw):
    """认不出就返回 None，不能猜一个最像的。

    猜错的后果是把某个指标的依据挂到另一个指标上，
    输出看起来合理但说的是别的事。
    """
    assert identify(raw) is None


def test_keywords_map_to_corpus_vocabulary():
    """关键词必须是说明书里的说法，不是报告上的全称。

    报告写「丙氨酸氨基转移酶」，说明书里写「转氨酶」。
    如果关键词取全称，检索永远为空——资料不是没有，是词对不上。
    """
    keywords = keywords_of("丙氨酸氨基转移酶 ALT")
    assert "转氨酶" in keywords
    assert "丙氨酸氨基转移酶" not in keywords


def test_matches_corpus():
    assert matches_corpus("尿酸 UA", "本品可升高血尿酸水平") is True
    assert matches_corpus("尿酸 UA", "本品用于降血压") is False
    assert matches_corpus("未知指标", "任意内容") is False


def test_crp_rejects_generic_infection_context():
    """裸的「感染」会命中大量无关语境，必须换成限定搭配。

    实测抽 5 条只有 1 条真相关：
        「接受抗感染药物治疗期间应监测血药浓度」—— 说的是联用抗菌药
        「HCV 基因 3 型感染」—— 是适应症
        「除危及生命的严重感染病例外，禁用于心衰患者」—— 是禁忌例外
    这些和 CRP 升高没有关系。
    """
    keywords = keywords_of("C 反应蛋白 CRP")

    assert "感染" not in keywords
    assert "二重感染" in keywords
    assert matches_corpus("C 反应蛋白 CRP", "接受抗感染药物治疗期间应监测血药浓度") is False
    assert matches_corpus("C 反应蛋白 CRP", "基因 3 型感染的患者") is False


def test_crp_accepts_specific_infection_context():
    """限定搭配命中的才是真相关：感染状态下的用药限制、药物引起的二重感染。

    对 CRP 升高的患者，「严重感染时禁用二甲双胍」是有意义的提示。
    """
    assert matches_corpus("C 反应蛋白 CRP", "长期应用可导致二重感染，应及时处理") is True
    assert matches_corpus("C 反应蛋白 CRP", "严重感染、外伤、重大手术者禁用") is True


def test_crp_does_not_match_fever():
    """「发热」不能作为 CRP 的关键词。

    实测阿托伐他汀注意事项里「伴全身不适、发热」说的是横纹肌溶解的症状，
    与 CRP 无关。这类词会稳定制造假阳性。
    """
    assert matches_corpus("C 反应蛋白 CRP", "出现肌肉疼痛、发热时应停药") is False


# ------------------------------------------------------------------ 测试夹具

def _item(name_raw: str, value: float, flag: str, ref_low=None, ref_high=None, unit=None) -> LabItem:
    return LabItem(
        name_raw=name_raw,
        value=value,
        unit=unit,
        ref_low=ref_low,
        ref_high=ref_high,
        flag=flag,
    )


@pytest.fixture
def toy_collection():
    """两药的小集合，内容摘自真实语料。

    集合名加随机后缀：EphemeralClient 在同一个进程里是共享的，
    固定名字会在第二个用例时报「Collection already exists」。
    """
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name=f"toy_{uuid.uuid4().hex[:12]}",
        metadata={"hnsw:space": "cosine"},
    )

    rows = [
        (
            "drug-a-注意事项",
            "他汀【注意事项】用药前、用药后12周及剂量增加后应复查肝功能，转氨酶持续升高超过正常上限3倍者应减量或停药。",
            {"drug": "阿托伐他汀", "section": "注意事项", "source": "阿托伐他汀钙片说明书"},
        ),
        (
            "drug-a-禁忌",
            "他汀【禁忌】活动性肝脏疾病，包括原因不明的血清转氨酶持续升高者禁用。",
            {"drug": "阿托伐他汀", "section": "禁忌", "source": "阿托伐他汀钙片说明书"},
        ),
        (
            "drug-b-注意事项",
            "布洛芬【注意事项】本品可引起胃肠道出血、溃疡，长期用药应定期监测血常规。",
            {"drug": "布洛芬", "section": "注意事项", "source": "布洛芬缓释胶囊说明书"},
        ),
        (
            "drug-c-注意事项",
            "别嘌醇【注意事项】本品可使血尿酸水平升高，痛风急性发作期不宜使用。",
            {"drug": "别嘌醇", "section": "注意事项", "source": "别嘌醇片说明书"},
        ),
    ]

    collection.add(
        ids=[r[0] for r in rows],
        embeddings=[[1.0, 0.0] for _ in rows],
        documents=[r[1] for r in rows],
        metadatas=[r[2] for r in rows],
    )
    return collection


def _report() -> Report:
    return Report(
        report_id="test-001",
        exam_date="2026-10-22",
        items=[
            _item("丙氨酸氨基转移酶 ALT", 68, "high", 9, 50, "U/L"),
            _item("尿酸 UA", 486, "high", 208, 428, "μmol/L"),
            _item("空腹血糖 GLU", 5.43, "normal", 3.9, 6.1, "mmol/L"),
        ],
    )


# ------------------------------------------------------------------ 核对逻辑

def test_review_without_medications_lists_only_abnormal():
    """没有用药信息时不猜药，只列异常。

    从 657 种药里找「哪些药影响转氨酶」会返回十几条，对具体的人没意义。
    """
    result = review(_report(), [])

    assert len(result.abnormal) == 2  # 血糖正常，不算异常
    assert result.medications == ()
    assert all(not f.evidences for f in result.findings)
    assert result.findings[0].label == "谷丙转氨酶"


def test_review_finds_related_sections(toy_collection):
    result = review(_report(), ["阿托伐他汀", "布洛芬"], collection=toy_collection)

    alt = next(f for f in result.findings if f.label == "谷丙转氨酶")
    assert alt.has_evidence
    sections = {e.section for e in alt.evidences}
    assert sections == {"注意事项", "禁忌"}
    assert all(e.drug == "阿托伐他汀" for e in alt.evidences)

    # 尿酸两药都没提，应为空
    ua = next(f for f in result.findings if f.label == "尿酸")
    assert not ua.has_evidence


def test_review_reports_unsupported_drugs(toy_collection):
    """库里没有的药必须明确报出来。

    如果悄悄跳过，用户会以为「系统查过这些药，没问题」——
    那是比拒答更危险的误解。
    """
    result = review(_report(), ["阿托伐他汀", "不存在的药"], collection=toy_collection)

    assert "不存在的药" in result.unsupported
    assert "阿托伐他汀" not in result.unsupported


def test_review_groups_by_organ(toy_collection):
    result = review(_report(), ["阿托伐他汀"], collection=toy_collection)
    organs = result.organs

    assert "肝" in organs
    assert any(f.label == "谷丙转氨酶" for f in organs["肝"])


# ------------------------------------------------------------------ 上下文拼装

def test_context_dedupes_repeated_evidence(toy_collection):
    """同一段原文被多个指标命中时，只能出现一次。

    ALT 和 AST 都会命中同一条「注意事项」。如果每个指标下都贴一遍原文，
    同一段会重复多次，浪费上下文也干扰模型。
    """
    report = Report(
        report_id="t",
        items=[
            _item("丙氨酸氨基转移酶 ALT", 68, "high", 9, 50),
            _item("天门冬氨酸氨基转移酶 AST", 41, "high", 15, 40),
        ],
    )
    result = review(report, ["阿托伐他汀"], collection=toy_collection)
    context, index = build_context(result)

    # 两条依据（注意事项 + 禁忌），而不是四条
    assert len(index) == 2
    body = context.split("【说明书依据】")[1]
    assert body.count("用药前、用药后12周") == 1


def test_context_omits_indicators_without_evidence(toy_collection):
    """没有依据的指标不进上下文。

    这是踩出来的：最初把 11 项异常全塞给模型，让它逐项说明，
    结果它把「转氨酶持续升高者应减量」读成「未提及」，
    还把 11 项逐一念了一遍。

    哪些指标没匹配到依据，代码已经知道，不该让模型再算一遍。
    """
    result = review(_report(), ["阿托伐他汀"], collection=toy_collection)
    context, _ = build_context(result)

    assert "尿酸" not in context
    assert "共 2 项，其中 1 项" in context


def test_context_numbers_follow_evidence_order(toy_collection):
    """引用编号要从 [1] 开始连续，否则答案里的 [n] 无法映射回出处。"""
    result = review(_report(), ["阿托伐他汀"], collection=toy_collection)
    context, index = build_context(result)

    assert sorted(index) == list(range(1, len(index) + 1))
    for n in index:
        assert f"[{n}]" in context


# ------------------------------------------------------------------ 解读编排（不调模型的分支）

def test_narrate_returns_early_when_no_abnormal():
    from app.report.narrative import narrate

    report = Report(
        report_id="t",
        items=[_item("空腹血糖 GLU", 5.43, "normal", 3.9, 6.1)],
    )
    narrative = narrate(review(report, []))

    assert narrative.refused is False
    assert "未发现异常" in narrative.text


def test_narrate_skips_model_when_no_evidence(toy_collection):
    """一条依据都没召回到时，不调模型。

    两个理由：省一次调用；也避免模型对着空资料硬讲。
    布洛芬的章节只讲胃肠道，不提转氨酶和尿酸。
    """
    from app.report.narrative import REFUSAL_PHRASE, narrate

    result = review(_report(), ["布洛芬"], collection=toy_collection)
    assert not result.with_evidence

    narrative = narrate(result)  # 不传 client，若真调模型会因为没有 Ollama 而报错

    assert narrative.refused is True
    assert REFUSAL_PHRASE in narrative.text
    assert narrative.citations == []


def test_drug_name_with_dosage_form_is_normalized(toy_collection):
    """带剂型的药名不能被误报「未收录」。

    曾经这里有两套判断标准：识别走别名表，报未收录时拿原始字符串比对。
    结果输入「阿托伐他汀钙片」既核对了依据，又报「未收录」——自相矛盾，
    会让用户以为系统没查这个药。
    """
    result = review(_report(), ["阿托伐他汀钙片"], collection=toy_collection)

    assert result.unsupported == ()
    assert result.with_evidence  # 确实核对到了


def test_unrecognized_drug_is_reported(toy_collection):
    """别名表完全不认识的药名也要报出来，不能静默丢弃。

    静默跳过的话，用户会以为「系统查过这个药，没问题」。
    """
    result = review(_report(), ["阿托伐他汀钙片", "某不存在的药"], collection=toy_collection)

    assert result.unsupported == ("某不存在的药",)
    assert result.with_evidence


def test_context_includes_indicator_note(toy_collection):
    """指标后面要带上词表里的等价说明。

    实测模型召回到了正确原文（「若出现黄疸…应及时就诊」），
    却因为不知道「黄疸是胆红素升高的表现」而判成无关。
    这属于术语等价，不该让模型靠训练记忆去猜——写进词表才是确定、可审计的。
    """
    result = review(_report(), ["别嘌醇"], collection=toy_collection)
    context, _ = build_context(result)

    assert "该指标升高与痛风相关" in context


def test_note_only_rendered_for_indicators_that_have_one(toy_collection):
    """没有等价说明的指标不要凭空加一句，避免上下文被废话撑大。"""
    result = review(_report(), ["阿托伐他汀"], collection=toy_collection)
    context, _ = build_context(result)

    alt_line = next(line for line in context.splitlines() if "谷丙转氨酶" in line)
    assert "；" not in alt_line
