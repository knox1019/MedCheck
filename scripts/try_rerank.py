"""学习用脚本：对比「纯向量检索」和「加重排」的排序差异。

这个脚本不是线上流程的一部分，专门用来观察 cross-encoder 到底做了什么。

用法：
    uv run python -m scripts.try_rerank

关注两件事：
    1. 重排前后 top-1 有没有变（看 rank_before 那一列）
    2. 重排分数和向量相似度的量级完全不同（logit vs 0~1），不能混比
"""

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.embedder import Embedder
from app.rag.retriever import get_reranker
from app.rag.store import get, get_client, search

# 三个提问，故意用不同意图：
#   第一个问「禁忌」——想找"谁不能用"
#   第二个问「怎么办」——想找"风险和处理"
#   第三个问「一起吃」——想找"相互作用"
QUERIES = [
    ("华法林的禁忌是什么", "华法林"),
    ("吃华法林牙龈出血怎么办", "华法林"),
    ("华法林和阿司匹林能一起吃吗", "华法林"),
    ("华法林是饭后吃还是饭前吃", "华法林"),
    ("华法林和什么药不能一起吃", "华法林"),
]


def show(title: str, hits: list[dict], score_key: str) -> None:
    print(f"  {title}")
    for rank, hit in enumerate(hits, start=1):
        section = hit["metadata"]["section"]
        if score_key == "rerank_score":
            extra = f"重排分 {hit['rerank_score']:+.4f}（原第 {hit['rank_before']} 名）"
        else:
            extra = f"相似度 {hit['similarity']:.4f}"
        print(f"    {rank}. {section:<8} {extra}")
    print()


def main() -> None:
    setup_logging()
    embedder = Embedder()
    collection = get(get_client(get_settings().chroma_dir))
    reranker = get_reranker()

    for query, drug in QUERIES:
        print("=" * 66)
        print(f"提问：{query}")

        # 第一步：向量检索，只取这个药的 3 个章节
        vector_hits = search(
            collection,
            embedder.embed_one(query),
            top_k=3,
            where={"drug": drug},
        )
        show("【纯向量检索】", vector_hits, "similarity")

        # 第二步：cross-encoder 对这个 3 条重新打分
        reranked = reranker.rerank(query, vector_hits)
        show("【加重排后】", reranked, "rerank_score")

        changed = reranked[0]["rank_before"] != 1
        print(f"  → top-1 {'变了' if changed else '没变'}\n")


if __name__ == "__main__":
    main()
