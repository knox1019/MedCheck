"""检索命令行：给一句话，返回最相关的几条语料，并说明能否回答。

用法：
    uv run python -m scripts.search "华法林的禁忌是什么"
    uv run python -m scripts.search "别嘌醇的禁忌"              # 应明确拒答
    uv run python -m scripts.search "药物相互作用" --drug 华法林
    uv run python -m scripts.search "肝功能" --min-sim 0.6      # 调阈值做对比实验
    uv run python -m scripts.search "华法林出血" --rerank        # 启用重排

输出结构：结论（能否回答）→ 出处（引用列表）→ 证据（命中的片段）。
先给出处再看原文是刻意的顺序——真实的 RAG 答案也是这么组织的。

这是 W3 的验收工具，和 W5 的问答接口共用同一套检索逻辑。
"""

import argparse

from app.core.logging import setup_logging
from app.rag.retriever import DEFAULT_MIN_SIMILARITY, retrieve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="药品说明书检索")
    parser.add_argument("query", help="检索语句")
    parser.add_argument("--top", type=int, default=5, help="返回条数，默认 5")
    parser.add_argument("--drug", help="手动指定药品，如 --drug 华法林")
    parser.add_argument(
        "--min-sim",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help=f"相关度阈值，默认 {DEFAULT_MIN_SIMILARITY}",
    )
    parser.add_argument("--rerank", action="store_true", help="启用 cross-encoder 重排")
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()

    result = retrieve(
        args.query,
        top_k=args.top,
        min_similarity=args.min_sim,
        drug=args.drug,
        use_rerank=args.rerank,
    )

    print()
    print(f'查询："{args.query}"')
    print("-" * 64)

    # 先给结论，再给证据。真实系统里这个判断会直接决定要不要调大模型
    if result.answerable:
        print(f"✅ 可以回答 —— {result.reason}")
    else:
        print(f"⛔ 拒答 —— {result.reason}")

    # 出处只在意 answerable 时展示。
    # 拒答却列出处，会让用户误以为「查到了但不想说」，事实是根本没有可用资料。
    if result.citations:
        print()
        print("出处：")
        for index, citation in enumerate(result.citations, start=1):
            print(f"  [{index}] {citation.label}")

    print()
    for rank, hit in enumerate(result.hits, start=1):
        meta = hit["metadata"]
        # 重排后展示重排分，否则展示向量相似度。两种分数量纲不同，不能混着比
        if "rerank_score" in hit:
            score = f"重排 {hit['rerank_score']:+.3f}"
        else:
            score = f"相似度 {hit['similarity']:.3f}"
        print(f"{rank}. [{score}] {meta['drug']} · {meta['section']}")
        print(f"   {hit['text'][:80]}……")
        print(f"   来源：{meta['source']}（{hit['id']}）")
        print()


if __name__ == "__main__":
    main()
