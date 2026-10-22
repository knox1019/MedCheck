"""检索编排：门控 → 向量检索 → 相关度判断。

这是 W3 的 search 脚本和 W5 的问答接口共用的检索层。
对外只暴露 retrieve()，返回结果自带「能不能回答」的判断。

三层防线，顺序不能反：

    第一层  实体门控    问题里的药在不在库里？不在直接拒答
    第二层  元数据过滤  命中药名就只在那种药的条目里检索
    第三层  相关度阈值  兜住「没提药名、但话题完全无关」的情况

为什么实体门控必须排在阈值前面：
    实测「药不在库」的相似度最高到 0.667，而「相关」最低是 0.684，
    两者只差 0.017。靠阈值区分不了，但靠药名匹配可以 100% 区分。
"""

from dataclasses import dataclass, field

from app.core.config import get_settings
from app.rag.embedder import Embedder
from app.rag.guards import DrugMatch, resolve_drugs
from app.rag.store import get, get_client, search

# 相关度下限。
# 这个数字不是拍脑袋定的，来自实测分布：
#     完全无关查询   0.284 – 0.353
#     主题相关查询   0.472 – 0.791
# 取 0.45 是为了挡住「天气/编程/做饭」这类完全跑题的查询。
#
# ⚠️ 它挡不住「主题相关但实体不存在」——那由实体门控负责。
# ⚠️ 换 embedding 模型后必须重新测量，这个值不能跨模型复用。
DEFAULT_MIN_SIMILARITY = 0.45


@dataclass
class RetrieveResult:
    """一次检索的完整结果，包含「能不能回答」的判断。"""

    query: str
    hits: list[dict] = field(default_factory=list)
    answerable: bool = False
    reason: str = ""
    drug_match: DrugMatch = field(default_factory=DrugMatch)

    @property
    def drug(self) -> str | None:
        return self.drug_match.primary

    @property
    def top_similarity(self) -> float | None:
        return self.hits[0]["similarity"] if self.hits else None


def retrieve(
    query: str,
    top_k: int = 5,
    min_similarity: float | None = None,
    drug: str | None = None,
    collection=None,
) -> RetrieveResult:
    """检索并判断是否足以回答。

    参数：
        drug           手动指定药品，优先于自动识别
        min_similarity 覆盖默认阈值，评测时用来做对比实验
        collection     复用已打开的集合，批量调用时避免反复开库
    """
    threshold = DEFAULT_MIN_SIMILARITY if min_similarity is None else min_similarity
    settings = get_settings()

    result = RetrieveResult(query=query)

    # ---- 第一层：实体门控 ----
    result.drug_match = resolve_drugs(query)

    if result.drug_match.only_unknown:
        names = "、".join(result.drug_match.unsupported)
        result.answerable = False
        result.reason = f"知识库未收录「{names}」，无法基于资料回答"
        return result

    # ---- 第二层：元数据过滤 ----
    target_drug = drug or result.drug_match.primary
    where = {"drug": target_drug} if target_drug else None

    if collection is None:
        collection = get(get_client(settings.chroma_dir))

    query_vector = Embedder().embed_one(query)
    hits = search(collection, query_vector, top_k=top_k, where=where)

    # 指定了药品却一条都没检到，说明语料里这个药其实没有对应章节
    if target_drug and not hits:
        result.answerable = False
        result.reason = f"「{target_drug}」在知识库中没有匹配的章节"
        return result

    result.hits = hits

    # ---- 第三层：相关度阈值 ----
    if not hits:
        result.answerable = False
        result.reason = "知识库为空或未检索到任何内容"
        return result

    if hits[0]["similarity"] < threshold:
        result.answerable = False
        result.reason = (
            f"检索结果与问题相关度过低（最高 {hits[0]['similarity']:.3f}，"
            f"阈值 {threshold:.2f}），不足以回答"
        )
        return result

    result.answerable = True
    if result.drug_match.has_unknown:
        names = "、".join(result.drug_match.unsupported)
        result.reason = f"已基于「{target_drug}」的资料检索；注意「{names}」未收录"
    elif target_drug:
        result.reason = f"命中「{target_drug}」的说明书章节"
    else:
        result.reason = "未识别出具体药品，已按语义相似度检索"

    return result
