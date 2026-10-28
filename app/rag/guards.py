"""检索前的门控：先认药名，再决定要不要检索。

为什么需要这一层——看实测数据：

    相关查询（库内有该药）    相似度 0.684 – 0.791
    不相关查询（药不在库内）  相似度 0.472 – 0.667
    完全无关查询（天气/做饭） 相似度 0.284 – 0.353

"药不在库内"和"相关"的最高/最低值只差 0.017，任何阈值都无法可靠区分。
而这类错误最危险：用户问「别嘌醇的禁忌」，系统返回布洛芬的禁忌，
相似度 0.612 看起来完全正常，用户根本发现不了。

所以拒答必须做**实体级判断**：问题里提到的药，在不在库里？

⚠️ 设计要点：**「库内有哪些药」不写在代码里，而是从向量库实时读取。**
早期版本把支持/不支持的药名各写了一份清单，结果语料扩充后两份清单
互相矛盾——阿司匹林明明已经入库，却仍被列为"未收录"而遭拒答。
现在改成：本文件只维护「能认出哪些药名」（别名表），
「哪些已收录」由 retriever 从 Chroma 的元数据里取。
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# 官方药品名称索引（国家基本药物目录 663 种）。
# 它只说明系统「认识」这个药名，不代表语料里有资料。
DRUG_INDEX_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus" / "drug_name_index.json"

# 能识别的药名 → 别名（通用名、商品名、英文名）。
# 这里收录两类：已经在语料里的，以及常见但暂时没收录的。
# 后者的作用不是回答问题，而是"认出来并明确告知没有资料"，
# 避免系统拿别的药的内容凑答案。
DRUG_ALIASES: dict[str, list[str]] = {
    # ---- 抗凝 / 抗血小板 ----
    "华法林": ["华法林", "华法令", "warfarin"],
    "利伐沙班": ["利伐沙班", "rivaroxaban", "拜瑞妥"],
    "氯吡格雷": ["氯吡格雷", "clopidogrel", "波立维"],
    "阿司匹林": ["阿司匹林", "aspirin", "巴米尔", "拜阿司匹灵"],
    # ---- 降糖 ----
    "二甲双胍": ["二甲双胍", "metformin", "格华止"],
    "阿卡波糖": ["阿卡波糖", "acarbose", "拜唐苹"],
    "格列美脲": ["格列美脲", "glimepiride", "亚莫利"],
    # ---- 降压 / 心衰 ----
    "氨氯地平": ["氨氯地平", "amlodipine", "络活喜"],
    "卡托普利": ["卡托普利", "captopril", "开博通"],
    "氯沙坦": ["氯沙坦", "losartan", "科素亚"],
    "缬沙坦": ["缬沙坦", "valsartan", "代文"],
    "美托洛尔": ["美托洛尔", "metoprolol", "倍他乐克"],
    "普萘洛尔": ["普萘洛尔", "propranolol", "心得安"],
    "呋塞米": ["呋塞米", "速尿", "furosemide"],
    "螺内酯": ["螺内酯", "spironolactone", "安体舒通"],
    "地高辛": ["地高辛", "digoxin"],
    "硝酸甘油": ["硝酸甘油", "nitroglycerin", "耐绞宁"],
    # ---- 调脂 ----
    "阿托伐他汀": ["阿托伐他汀", "atorvastatin", "立普妥"],
    "瑞舒伐他汀": ["瑞舒伐他汀", "rosuvastatin", "可定"],
    "辛伐他汀": ["辛伐他汀", "simvastatin", "舒降之"],
    # ---- 消化 ----
    "奥美拉唑": ["奥美拉唑", "omeprazole", "洛赛克"],
    "多潘立酮": ["多潘立酮", "domperidone", "吗丁啉"],
    "蒙脱石散": ["蒙脱石散", "蒙脱石", "思密达", "smectite"],
    # ---- 抗感染 ----
    "阿莫西林": ["阿莫西林", "amoxicillin"],
    "头孢克肟": ["头孢克肟", "cefixime"],
    "头孢呋辛": ["头孢呋辛", "cefuroxime", "西力欣"],
    "阿奇霉素": ["阿奇霉素", "azithromycin", "希舒美"],
    "左氧氟沙星": ["左氧氟沙星", "levofloxacin", "可乐必妥"],
    "甲硝唑": ["甲硝唑", "metronidazole", "灭滴灵"],
    # ---- 其他 ----
    "布洛芬": ["布洛芬", "ibuprofen", "芬必得"],
    "对乙酰氨基酚": ["对乙酰氨基酚", "扑热息痛", "paracetamol", "acetaminophen", "泰诺"],
    "氯雷他定": ["氯雷他定", "loratadine", "开瑞坦"],
    "左甲状腺素钠": ["左甲状腺素钠", "左甲状腺素", "levothyroxine", "优甲乐"],
    "别嘌醇": ["别嘌醇", "allopurinol", "别嘌呤醇"],
    "非布司他": ["非布司他", "febuxostat", "优立通"],
}


@dataclass
class DrugMatch:
    """从问题里识别出的药品情况。"""

    matched: list[str] = field(default_factory=list)  # 识别到，且库里确实有资料
    unsupported: list[str] = field(default_factory=list)  # 识别到，但库里没有资料

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


@lru_cache
def load_drug_index() -> dict[str, list[str]]:
    """加载官方药品名称索引，返回 药名 → 别名列表。

    只读一次，用 lru_cache 缓存。文件不存在时返回空字典，
    这样即使忘了先生成索引，系统仍能用内置的 DRUG_ALIASES 工作。
    """
    if not DRUG_INDEX_PATH.exists():
        return {}
    import json

    payload = json.loads(DRUG_INDEX_PATH.read_text(encoding="utf-8"))
    return {entry["name"]: entry["aliases"] for entry in payload.get("drugs", [])}


def scan_index() -> dict[str, list[str]]:
    """合并内置别名表与官方索引。

    内置表里有商品名和英文名（芬必得、warfarin），官方索引有 663 个通用名。
    两份合并后系统的「可识别药名」才是完整的。
    """
    merged = dict(load_drug_index())
    merged.update(DRUG_ALIASES)  # 内置表优先，覆盖同名条目
    return merged


def _drop_substring_names(drugs: list[str]) -> list[str]:
    """去掉「被更长药名包含」的短名。

    「青霉素」是「苄星青霉素」和「阿莫西林克拉维酸钾」里没有，
    但「阿莫西林」确实是「阿莫西林克拉维酸钾」的子串。
    不清理的话，问复方制剂会命中单方成分，然后用单方的资料作答——
    这是比拒答更严重的错误：答案看起来很合理，但说的不是同一个药。
    """
    return [
        drug
        for drug in drugs
        if not any(drug != other and drug in other for other in drugs)
    ]


def _scan(text: str, index: dict[str, list[str]]) -> list[str]:
    """在 query 里扫出命中的药名（按别名子串匹配）。

    用子串匹配而不是分词，是因为中文药名没有空格边界，
    「我吃的是华法林」这种句子里直接找「华法林」最可靠。

    返回结果**按药名在问题中出现的先后排序**，而不是按字典顺序。
    因为 primary（主药品）取的是第一个，语义上应该是「用户先提到的那个」：
    「布洛芬和阿司匹林能一起吃吗」的主语是布洛芬，不是阿司匹林。
    """
    lowered = text.lower()
    hits: list[tuple[int, str]] = []
    for drug, aliases in index.items():
        positions = [lowered.find(alias.lower()) for alias in aliases]
        found = [pos for pos in positions if pos >= 0]
        if found:
            hits.append((min(found), drug))
    hits.sort()
    return [drug for _, drug in hits]


def resolve_drugs(query: str, supported: set[str]) -> DrugMatch:
    """从问题里识别药品，并按 supported（库内实际收录的药）分类。

    supported 必须来自真实索引，不能硬编码——否则语料更新后
    会出现「已入库的药仍被判为未收录」的矛盾。

    可识别的药名 = 内置别名表 + 官方药品名称索引（663 种），
    因此「知道这个药但没有资料」能被准确识别并拒答，
    而不是拿语义上最接近的别的药去凑答案。

    >>> resolve_drugs("华法林的禁忌", {"华法林"})        → matched=['华法林']
    >>> resolve_drugs("别嘌醇的禁忌", {"华法林"})        → unsupported=['别嘌醇']
    >>> resolve_drugs("布洛芬和阿司匹林", {"布洛芬"})     → matched=['布洛芬'], unsupported=['阿司匹林']
    """
    hits = _drop_substring_names(_scan(query, scan_index()))
    matched = [drug for drug in hits if drug in supported]
    unsupported = [drug for drug in hits if drug not in supported]
    return DrugMatch(matched=matched, unsupported=unsupported)
