"""检验指标与语料用语之间的映射。

为什么需要这个文件：

报告上印的是「丙氨酸氨基转移酶 ALT」，药品说明书里写的是「转氨酶」。
两者说的是同一件事，但字面完全对不上。如果不做这层映射，
拿报告里的指标名去检索语料，命中率会非常低——不是资料没有，
是两套词汇体系不通。

所以这里维护一张「指标 → 说明书里的说法」的对照表。
它是连接 W2（报告解析）和 W3（药品知识库）的桥。

注意边界：这张表只做「词」的桥接，不解释指标的临床意义。
指标释义需要医学指南类语料，本项目语料是药品说明书，不覆盖这部分。
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IndicatorProfile:
    """一个检验指标在语料里的对应说法。"""

    code: str  # 标准缩写，如 ALT
    display: str  # 中文常用名，如 谷丙转氨酶
    keywords: tuple[str, ...]  # 说明书里可能出现的说法，用于在语料中定位
    organ: str  # 归属的器官/系统，用于归类展示
    # 报告上可能出现的其他中文写法。比如展示名是「血钾」，
    # 而报告印的是「钾 K」，需要额外认一个「钾」。
    match_names: tuple[str, ...] = ()
    # 给模型看的补充说明：与口语/说明书用语的等价关系。
    #
    # 为什么需要这个字段：实测模型召回到了正确原文，却判断错关系。
    # 例如总胆红素的依据是「若出现黄疸、明显乏力…应及时就诊」，
    # 模型知道原文提到了胆红素相关词，但不知道「黄疸是胆红素升高的表现」，
    # 于是判成「无关」。
    #
    # 这是术语等价，不是诊断推理——把它写在词表里是确定、可审计的，
    # 比让模型凭训练记忆去猜可靠得多。
    note: str = ""


# 映射表。keywords 里的词必须是药品说明书里真实会出现的写法，
# 不能写成报告上的全称——那样检索不到。
PROFILES: tuple[IndicatorProfile, ...] = (
    # ---- 肝功能 ----
    IndicatorProfile("ALT", "谷丙转氨酶", ("转氨酶", "ALT", "肝功能"), "肝"),
    IndicatorProfile("AST", "谷草转氨酶", ("转氨酶", "AST", "肝功能"), "肝"),
    IndicatorProfile(
        "TBIL", "总胆红素", ("胆红素", "黄疸", "肝功能"), "肝",
        note="该指标升高可表现为黄疸",
    ),
    IndicatorProfile("ALP", "碱性磷酸酶", ("碱性磷酸酶", "肝功能"), "肝"),
    IndicatorProfile("GGT", "γ-谷氨酰转肽酶", ("转肽酶", "肝功能"), "肝"),
    IndicatorProfile("ALB", "白蛋白", ("白蛋白", "低蛋白"), "肝"),
    IndicatorProfile("TP", "总蛋白", ("总蛋白", "低蛋白"), "肝"),
    # ---- 血常规 ----
    IndicatorProfile(
        "HGB", "血红蛋白", ("贫血", "血红蛋白"), "血液",
        note="该指标降低即贫血",
    ),
    IndicatorProfile("RBC", "红细胞计数", ("贫血", "红细胞"), "血液"),
    IndicatorProfile("WBC", "白细胞计数", ("白细胞",), "血液"),
    IndicatorProfile("NEUT", "中性粒细胞", ("中性粒", "白细胞"), "血液"),
    IndicatorProfile("LYMPH", "淋巴细胞", ("淋巴", "白细胞"), "血液"),
    IndicatorProfile("PLT", "血小板计数", ("血小板", "凝血", "出血"), "血液"),
    # ---- 血脂 ----
    IndicatorProfile("TC", "总胆固醇", ("胆固醇", "血脂"), "血脂"),
    IndicatorProfile("TG", "甘油三酯", ("甘油三酯", "血脂"), "血脂"),
    IndicatorProfile("HDL-C", "高密度脂蛋白胆固醇", ("胆固醇", "血脂"), "血脂"),
    IndicatorProfile("LDL-C", "低密度脂蛋白胆固醇", ("胆固醇", "血脂"), "血脂"),
    # ---- 血糖 ----
    IndicatorProfile("GLU", "空腹血糖", ("血糖", "低血糖", "高血糖"), "代谢"),
    IndicatorProfile("HBA1C", "糖化血红蛋白", ("血糖", "糖化"), "代谢"),
    # ---- 肾功能 ----
    IndicatorProfile(
        "CREA", "血肌酐", ("肌酐", "肾功能"), "肾",
        note="该指标升高提示肾功能减退",
    ),
    IndicatorProfile("UREA", "尿素", ("尿素", "肾功能"), "肾"),
    IndicatorProfile(
        "UA", "尿酸", ("尿酸", "痛风"), "肾",
        note="该指标升高与痛风相关",
    ),
    # ---- 炎症 ----
    # 这个词组的两个坑都是实测踩出来的：
    #
    # 1) 不放「发热」——阿托伐他汀注意事项里「伴全身不适、发热」说的是
    #    横纹肌溶解的症状，和 CRP 毫无关系。
    #
    # 2) 不用裸的「感染」——它会命中「接受抗感染治疗期间」「HCV 感染」
    #    「严重感染病例」这类语境。实测抽 5 条只有 1 条真相关。
    #    改为限定搭配后，覆盖从 125 药降到 40 药，抽 6 条有 5 条相关，
    #    命中的多是「严重感染时禁用」「长期应用可致二重感染」——
    #    这对 CRP 升高的患者是真正有用的提示。
    IndicatorProfile(
        "CRP", "C 反应蛋白",
        ("炎症", "感染风险", "继发感染", "二重感染", "易致感染", "严重感染"),
        "炎症",
        note="升高提示炎症或感染；说明书的「感染」多为感染状态下的用药限制",
    ),
    # ---- 电解质 ----
    IndicatorProfile("K", "血钾", ("血钾", "低钾", "高钾"), "电解质", ("钾",)),
    IndicatorProfile("NA", "血钠", ("血钠", "低钠", "高钠"), "电解质", ("钠",)),
)

_BY_CODE = {p.code: p for p in PROFILES}

# 英文缩写按「前后不是字母」匹配。否则 ALB 会被当成 ALP，
# 更糟的是 A 开头的短缩写互相误伤。
# 忽略大小写是必须的：报告印「钠 Na」，而标准缩写写作「NA」。
_CODE_PATTERN = re.compile(
    r"(?<![A-Za-z])(" + "|".join(sorted(_BY_CODE, key=len, reverse=True)) + r")(?![A-Za-z])",
    re.IGNORECASE,
)


def identify(name_raw: str) -> IndicatorProfile | None:
    """从报告的原始指标名识别出标准指标。

    匹配顺序：先认中文全称，再认英文缩写。
    先中文是因为中文名不会歧义；英文缩写短，容易撞车。

    >>> identify("丙氨酸氨基转移酶 ALT").code
    'ALT'
    >>> identify("血红蛋白  HGB").code
    'HGB'
    >>> identify("某个没见过的指标") is None
    True
    """
    text = (name_raw or "").strip()
    if not text:
        return None

    # 先按中文常用名匹配，取最长的那个（避免「胆固醇」抢走「高密度脂蛋白胆固醇」）
    candidates: list[tuple[int, IndicatorProfile]] = []
    for profile in PROFILES:
        for name in (profile.display, *profile.match_names):
            if name in text:
                candidates.append((len(name), profile))
                break
    if candidates:
        return max(candidates, key=lambda pair: pair[0])[1]

    # 再按英文缩写匹配（要求前后不是字母，忽略大小写）
    match = _CODE_PATTERN.search(text)
    if match:
        return _BY_CODE.get(match.group(1).upper())

    return None


def keywords_of(name_raw: str) -> tuple[str, ...]:
    """取一个指标在语料里的检索关键词。识别不出时返回空元组。"""
    profile = identify(name_raw)
    return profile.keywords if profile else ()


def matches_corpus(name_raw: str, text: str) -> bool:
    """判断一段说明书文字是否可能与该指标有关。

    这是纯字符串判断，不用向量检索。原因：
    指标名和说明书用词是两套固定词汇（转氨酶、尿酸、血小板…），
    精确匹配比语义相似度更可控，也不会把「肝功能正常的病人」误判成相关。
    """
    kws = keywords_of(name_raw)
    return any(kw in (text or "") for kw in kws)
