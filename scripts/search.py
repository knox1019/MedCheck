"""检索命令行：给一句话，返回最相关的几条语料，并说明能否回答。

用法：
    uv run python -m scripts.search "华法林的禁忌是什么"
    uv run python -m scripts.search "别嘌醇的禁忌"              # 应明确拒答
    uv run python -m scripts.search "药物相互作用" --drug 华法林
    uv run python -m scripts.search "肝功能" --min-sim 0.6      # 调阈值做对比实验

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
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()

    result = retrieve(
        args.query,
        top_k=args.top,
        min_similarity=args.min_sim,
        drug=args.drug,
    )

    print()
    print(f'查询："{args.query}"')
    print("-" * 64)

    # 先给结论，再给证据。真实系统里这个判断会直接决定要不要调大模型
    if result.answerable:
        print(f"✅ 可以回答 —— {result.reason}")
    else:
        print(f"⛔ 拒答 —— {result.reason}")
    print()

    for rank, hit in enumerate(result.hits, start=1):
        meta = hit["metadata"]
        print(f"{rank}. [{hit['similarity']:.3f}] {meta['drug']} · {meta['section']}")
        print(f"   {hit['text'][:80]}……")
        print(f"   来源：{meta['source']}")
        print()


if __name__ == "__main__":
    main()
