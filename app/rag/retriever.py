"""检索编排：门控 → 向量检索 → 相关度判断。

这是 search 脚本和后续问答接口共用的检索层。
对外只暴露 retrieve()，返回结果自带「能不能回答」的判断。

三层防线，顺序不能反：

    第一层  实体门控    问题里的药在不在库里？不在直接拒答
    第二层  元数据过滤  命中药名就只在那种药的条目里检索
    第三层  相关度阈值  兜住「没提药名、但话题完全无关」的情况

为什么实体门控必须排在阈值前面：
    实测「药不在库」的相似度最高到 0.667，而「相关」最低是 0.684，
    两者只差 0.017。靠阈值区分不了，但靠药名匹配可以 100% 区分。

⚠️ 「库内有哪些药」从 Chroma 元数据实时读取，不硬编码。
"""

from dataclasses import dataclass, field

from app.core.config import get_settings
from app.rag.citation import Citation, citations_from_hits
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
    reranked: bool = False
    # 出处列表，随 hits 一起产出。拒答时为空——没有依据就不该有出处。
    citations: list[Citation] = field(default_factory=list)

    @property
    def drug(self) -> str | None:
        return self.drug_match.primary

    @property
    def top_similarity(self) -> float | None:
        return self.hits[0]["similarity"] if self.hits else None


def supported_drugs(collection) -> set[str]:
    """从向量库的元数据里读出「实际收录了哪些药」。

    这是刻意设计成动态读取的：语料一改，门控自动跟着变，
    不需要再回来改代码。早期把药名清单硬编码，结果语料扩充后
    出现了「已入库的药仍被判为未收录」的矛盾。
    """
    data = collection.get(include=["metadatas"])
    return {
        meta["drug"]
        for meta in (data.get("metadatas") or [])
        if meta and "drug" in meta
    }


# 重排器做成模块级单例：加载 ONNX 模型要 3-4 秒，
# 如果每次查询都重新加载，评测 27 条就要多花一分多钟。
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from app.rag.reranker import Reranker

        _reranker = Reranker()
    return _reranker


def retrieve(
    query: str,
    top_k: int = 5,
    min_similarity: float | None = None,
    drug: str | None = None,
    collection=None,
    use_rerank: bool = False,
    recall_k: int | None = None,
) -> RetrieveResult:
    """检索并判断是否足以回答。

    参数：
        drug           手动指定药品，优先于自动识别
        min_similarity 覆盖默认阈值，评测时用来做对比实验
        collection     复用已打开的集合，批量调用时避免反复开库
        use_rerank     是否启用 cross-encoder 重排
        recall_k       召回阶段的候选数。开启重排时默认取 top_k 的 4 倍，
                       因为重排需要比最终结果更大的候选池才有发挥空间
    """
    threshold = DEFAULT_MIN_SIMILARITY if min_similarity is None else min_similarity
    settings = get_settings()

    result = RetrieveResult(query=query)

    if collection is None:
        collection = get(get_client(settings.chroma_dir))

    # ---- 第一层：实体门控 ----
    result.drug_match = resolve_drugs(query, supported_drugs(collection))

    if result.drug_match.only_unknown:
        names = "、".join(result.drug_match.unsupported)
        result.answerable = False
        result.reason = f"知识库未收录「{names}」，无法基于资料回答"
        return result

    # ---- 第二层：元数据过滤 ----
    # 命中多个药时用 $in 一起检索，而不是只取第一个——
    # 「布洛芬和阿司匹林能一起吃吗」这类问题，两种药的资料都需要。
    target_drugs = [drug] if drug else result.drug_match.matched
    if len(target_drugs) == 1:
        where = {"drug": target_drugs[0]}
    elif len(target_drugs) > 1:
        where = {"drug": {"$in": target_drugs}}
    else:
        where = None

    # 召回阶段：开启重排时多取候选
    candidate_k = recall_k or (top_k * 4 if use_rerank else top_k)

    query_vector = Embedder().embed_one(query)
    hits = search(collection, query_vector, top_k=candidate_k, where=where)

    # 指定了药品却一条都没检到，说明这个药在库里没有对应章节
    if target_drugs and not hits:
        result.answerable = False
        result.reason = f"「{'、'.join(target_drugs)}」在知识库中没有匹配的章节"
        return result

    if not hits:
        result.answerable = False
        result.reason = "知识库为空或未检索到任何内容"
        return result

    # ---- 第三层：相关度阈值 ----
    # 注意：阈值判断必须在重排之前，用「向量相似度」来判。
    # 重排分数是 cross-encoder 的 logit，量纲不同，不能拿来跟 0.45 比。
    if hits[0]["similarity"] < threshold:
        result.answerable = False
        result.reason = (
            f"检索结果与问题相关度过低（最高 {hits[0]['similarity']:.3f}，"
            f"阈值 {threshold:.2f}），不足以回答"
        )
        return result

    # ---- 第四层（可选）：重排 ----
    # 召回已保证「相关内容在候选里」，重排负责把最该在前面的拎上来。
    if use_rerank:
        hits = get_reranker().rerank(query, hits, top_k=top_k)
        result.reranked = True
    else:
        hits = hits[:top_k]

    result.hits = hits

    # ---- 出处：只有放行的情况才有出处 ----
    # 拒绝回答时绝不能给出处——那会让用户以为系统「查过资料但不想说」，
    # 而事实是根本没有可用资料。
    result.citations = citations_from_hits(hits)

    result.answerable = True
    if result.drug_match.has_unknown:
        names = "、".join(result.drug_match.unsupported)
        result.reason = f"已基于「{'、'.join(target_drugs)}」的资料检索；注意「{names}」未收录"
    elif target_drugs:
        result.reason = f"命中「{'、'.join(target_drugs)}」的说明书章节"
    else:
        result.reason = "未识别出具体药品，已按语义相似度检索"

    return result
