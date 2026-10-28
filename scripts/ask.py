"""问答命令行：问一句，拿带引用的答案。

用法：
    uv run python -m scripts.ask "华法林的禁忌是什么"
    uv run python -m scripts.ask "华法林是饭前吃还是饭后吃"     # 应该如实说资料没提到
    uv run python -m scripts.ask "别嘌醇的禁忌"                # 应该拒答
    uv run python -m scripts.ask "华法林出血" --no-rerank
"""

import argparse

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.llm.qa import answer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="体检报告与用药禁忌问答")
    parser.add_argument("question", help="要问的问题")
    parser.add_argument("--top", type=int, default=4, help="送入模型的资料条数，默认 4")
    parser.add_argument("--no-rerank", action="store_true", help="关闭重排（更快）")
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    settings = get_settings()

    result = answer(args.question, top_k=args.top, use_rerank=not args.no_rerank)

    print()
    print(f"问题：{args.question}")
    print(f"模型：{settings.llm_model}")
    print("-" * 64)

    if result.refused:
        print(f"⛔ 拒答 —— {result.reason}")
        if result.text:
            print(f"   回答：{result.text}")
        return

    print(result.text)

    if result.citations:
        print()
        print("出处：")
        for index, citation in enumerate(result.citations, start=1):
            print(f"  [{index}] {citation.label}")

    # 模型没标引用是可监控的质量信号，直接暴露出来而不是掩盖
    if not result.cited:
        print()
        print("⚠️ 答案中未出现引用标记，出处为候选资料全量列出")


if __name__ == "__main__":
    main()
