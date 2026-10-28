"""报告侧核对：把「异常指标」和「在服药物说明书」连起来。

这个模块解决的是项目名字里的核心问题（MedCheck = 体检报告 + 用药核对）：
报告说肝功能异常，你正在吃的药里有没有哪个会影响肝功能？

三个设计决定，每个都有理由：

1. **用精确关键词匹配，不用向量检索。**
   「转氨酶」「尿酸」「血小板」是固定词汇，两份资料说的是同一件事时
   字面一定对得上。反过来向量检索会捞进「肝功能正常者无需调整」这类
   语义接近但其实无关的句子，制造噪音。这里要的是「有没有提到」，
   不是「像不像」。

2. **必须先知道患者在吃什么药。**
   没有用药信息时，从 657 种药里找「哪些药影响转氨酶」会返回十几条，
   对具体的人毫无意义。所以没提供药物时只列异常清单，不做推测。

3. **只说「说明书提到了什么」，不说「一定是这个药引起的」。**
   因果关系需要医生判断。系统输出的是依据，不是结论。
"""

from dataclasses import dataclass, field

from app.core.config import get_settings
from app.ingest.indicators import IndicatorProfile, identify
from app.ingest.models import LabItem, Report
from app.rag.guards import resolve_drugs
from app.rag.store import get, get_client


@dataclass(frozen=True)
class Evidence:
    """一条来自说明书的依据。"""

    drug: str
    section: str
    text: str
    source: str
    chunk_id: str
    matched: tuple[str, ...]  # 命中了哪些关键词，让用户能看出为什么被选中


@dataclass(frozen=True)
class IndicatorFinding:
    """一个异常指标及其相关依据。"""

    item: LabItem
    profile: IndicatorProfile | None
    evidences: tuple[Evidence, ...] = ()

    @property
    def label(self) -> str:
        """展示用的指标名。识别不出时退回原始名。"""
        return self.profile.display if self.profile else self.item.name_raw

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidences)


@dataclass
class MedicationReview:
    """一次完整的报告核对结果。"""

    report: Report
    abnormal: tuple[LabItem, ...] = ()
    medications: tuple[str, ...] = ()
    findings: tuple[IndicatorFinding, ...] = ()
    unsupported: tuple[str, ...] = field(default_factory=tuple)

    @property
    def with_evidence(self) -> list[IndicatorFinding]:
        """只保留找到说明书依据的那几项——这是最该给用户看的部分。"""
        return [f for f in self.findings if f.has_evidence]

    @property
    def organs(self) -> dict[str, list[IndicatorFinding]]:
        """按器官/系统归类，让「肝功几项都偏高」这种聚集现象浮现出来。"""
        grouped: dict[str, list[IndicatorFinding]] = {}
        for finding in self.findings:
            key = finding.profile.organ if finding.profile else "其他"
            grouped.setdefault(key, []).append(finding)
        return grouped


def _fetch_sections(drugs: list[str], collection) -> list[dict]:
    """一次取出这些药的全部章节。

    用 where + $in 批量取，而不是逐个药检索——核对需要看完整的章节内容，
    不需要按相似度排序。1659 条的库里这个操作是毫秒级。
    """
    if not drugs:
        return []
    data = collection.get(
        where={"drug": {"$in": list(drugs)}},
        include=["documents", "metadatas"],
    )
    return [
        {
            "chunk_id": chunk_id,
            "text": text or "",
            "meta": meta or {},
        }
        for chunk_id, text, meta in zip(
            data.get("ids") or [], data.get("documents") or [], data.get("metadatas") or []
        )
    ]


def review(
    report: Report,
    medications: list[str] | None = None,
    collection=None,
) -> MedicationReview:
    """核对一份报告与在服药物。

    参数：
        medications 患者正在服用的药名列表。为空时只列异常指标，不做核对。
        collection  复用已打开的集合，批量调用时避免反复开库。
    """
    medications = [m.strip() for m in (medications or []) if m and m.strip()]
    abnormal = tuple(item for item in report.items if item.flag in ("high", "low"))

    result = MedicationReview(report=report, abnormal=abnormal, medications=tuple(medications))

    if not medications:
        # 没有用药信息：只把异常指标按指标识别出来，不猜任何药
        result.findings = tuple(
            IndicatorFinding(item=item, profile=identify(item.name_raw)) for item in abnormal
        )
        return result

    if collection is None:
        collection = get(get_client(get_settings().chroma_dir))

    supported = {
        meta["drug"]
        for meta in (collection.get(include=["metadatas"]).get("metadatas") or [])
        if meta and "drug" in meta
    }

    # 药名逐个过别名表，统一归到「可核对」或「未收录」两类。
    #
    # 早期用的是两套标准：识别时走别名表，报「未收录」时拿原始字符串比对。
    # 结果输入「阿托伐他汀钙片」会出现自相矛盾——既核对了 5 项依据，
    # 又报「未收录」，因为库里存的是「阿托伐他汀」。
    # 现在两个结论都来自同一次解析，不会再打架。
    known: list[str] = []
    unsupported: list[str] = []
    for name in medications:
        match = resolve_drugs(name, supported)
        if match.matched:
            for drug in match.matched:
                if drug not in known:
                    known.append(drug)
        elif match.unsupported:
            for drug in match.unsupported:
                if drug not in unsupported:
                    unsupported.append(drug)
        else:
            # 别名表完全不认识这个名字，库里也不可能有
            unsupported.append(name)

    result.unsupported = tuple(unsupported)

    sections = _fetch_sections(known, collection)

    findings: list[IndicatorFinding] = []
    for item in abnormal:
        profile = identify(item.name_raw)
        if profile is None:
            findings.append(IndicatorFinding(item=item, profile=None))
            continue

        evidences: list[Evidence] = []
        for section in sections:
            hit = tuple(kw for kw in profile.keywords if kw in section["text"])
            if not hit:
                continue
            evidences.append(
                Evidence(
                    drug=section["meta"].get("drug", "未知药品"),
                    section=section["meta"].get("section", "未标注章节"),
                    text=section["text"],
                    source=section["meta"].get("source", "未标注来源"),
                    chunk_id=section["chunk_id"],
                    matched=hit,
                )
            )

        findings.append(IndicatorFinding(item=item, profile=profile, evidences=tuple(evidences)))

    result.findings = tuple(findings)
    return result
