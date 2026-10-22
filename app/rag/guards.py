"""检索前的门控：先认药名，再决定要不要检索。

为什么需要这一层——看实测数据：

    相关查询（库内有该药）    相似度 0.684 – 0.791
    不相关查询（药不在库内）  相似度 0.472 – 0.667
    完全无关查询（天气/做饭） 相似度 0.284 – 0.353

"药不在库内"和"相关"的最高/最低值只差 0.017，任何阈值都无法可靠区分。
而这类错误最危险：用户问「别嘌醇的禁忌」，系统返回布洛芬的禁忌，
相似度 0.612 看起来完全正常，用户根本发现不了。

所以拒答不能只靠相似度阈值，必须做**实体级判断**：
问题里提到的药，在不在库里？不在就直接拒答，不走向量检索。

注意：别名表写在代码里是为了简单。真实项目应该放到
data/corpus/drug_aliases.json 里，方便非程序员维护。
"""

from dataclasses import dataclass, field

# 库内收录的药品 → 别名（含通用名、商品名、英文名）
DRUG_ALIASES: dict[str, list[str]] = {
    "二甲双胍": ["二甲双胍", "metformin", "格华止"],
    "华法林": ["华法林", "华法令", "warfarin"],
    "头孢克肟": ["头孢克肟", "cefixime"],
    "奥美拉唑": ["奥美拉唑", "omeprazole", "洛赛克"],
    "布洛芬": ["布洛芬", "ibuprofen", "芬必得"],
    "氯吡格雷": ["氯吡格雷", "clopidogrel", "波立维"],
    "阿托伐他汀": ["阿托伐他汀", "atorvastatin", "立普妥"],
    "阿莫西林": ["阿莫西林", "amoxicillin"],
}

# 常见但本项目未收录的药品。
# 作用：识别出「用户问了一个真实存在的药，但我们没有它的资料」，
# 从而给出明确拒答，而不是拿别的药的内容凑答案。
UNSUPPORTED_DRUGS: list[str] = [
    "别嘌醇",
    "阿司匹林",
    "氨氯地平",
    "瑞舒伐他汀",
    "辛伐他汀",
    "非布司他",
    "左氧氟沙星",
    "头孢呋辛",
    "美托洛尔",
    "缬沙坦",
    "呋塞米",
    "螺内酯",
    "格列美脲",
    "对乙酰氨基酚",
    "左甲状腺素",
    "地高辛",
]


@dataclass
class DrugMatch:
    """从问题里识别出的药品情况。"""

    matched: list[str] = field(default_factory=list)  # 命中且库里有资料
    unsupported: list[str] = field(default_factory=list)  # 提到了但库里没有

    @property
    def primary(self) -> str | None:
        """主药品。多药同时出现时取第一个命中的。"""
        return self.matched[0] if self.matched else None

    @property
    def has_unknown(self) -> bool:
        """是否提到了未收录的药品。"""
        return bool(self.unsupported)

    @property
    def only_unknown(self) -> bool:
        """只提到了未收录的药，没法回答。"""
        return not self.matched and bool(self.unsupported)


def _scan(text: str, candidates: dict[str, list[str]]) -> list[str]:
    """在 query 里扫出命中的药名（按别名子串匹配）。

    用子串匹配而不是分词，是因为中文药名没有空格边界，
    「我吃的是华法林」这种句子里直接找「华法林」最可靠。
    """
    lowered = text.lower()
    hits: list[str] = []
    for drug, aliases in candidates.items():
        if any(alias.lower() in lowered for alias in aliases):
            hits.append(drug)
    return hits


def resolve_drugs(query: str) -> DrugMatch:
    """从问题里识别药品。

    >>> resolve_drugs("华法林的禁忌")  → matched=['华法林']
    >>> resolve_drugs("别嘌醇的禁忌")  → unsupported=['别嘌醇']
    >>> resolve_drugs("布洛芬和阿司匹林能一起吃吗")
        → matched=['布洛芬'], unsupported=['阿司匹林']
    """
    matched = _scan(query, DRUG_ALIASES)

    # 未收录药品用简单列表扫描。
    # 注意要排除已经是库内药品的别名（避免重复计数）。
    unsupported = [
        drug
        for drug in UNSUPPORTED_DRUGS
        if drug in query and drug not in matched
    ]

    return DrugMatch(matched=matched, unsupported=unsupported)
