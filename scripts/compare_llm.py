"""模型对比实验：同一批问题，不同大模型各跑一遍。

用途：回答「本地小模型够不够用」这个决策问题。
指标：引用标注率、拒答正确率、延迟、答案长度。

用法：
    uv run python -m scripts.compare_llm
    uv run python -m scripts.compare_llm --models qwen2.5:1.5b qwen2.5:7b deepseek-chat

注意：本地模型走 CPU 推理，7b 单次可能要 30–60 秒。
"""

import argparse
import statistics
import time

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.llm.client import ChatClient
from app.llm.qa import answer

# (问题, 期望行为)
#   answer  → 应给出带依据的回答
#   refuse  → 应拒答（语料未覆盖或药不在库）
CASES: list[tuple[str, str]] = [
    ("华法林的禁忌是什么", "answer"),
    ("阿司匹林会不会把胃吃坏", "answer"),
    ("多潘立酮会导致心脏问题吗", "answer"),
    ("华法林和什么药不能一起吃", "answer"),
    ("华法林是饭前吃还是饭后吃", "refuse"),
    ("别嘌醇的禁忌是什么", "refuse"),
]


def run_one(model: str, question: str) -> dict:
    """用指定模型回答一个问题，记录各项指标。"""
    client = ChatClient(model=model)
    started = time.perf_counter()
    result = answer(question, top_k=3, use_rerank=True, client=client)
    elapsed = time.perf_counter() - started

    return {
        "refused": result.refused,
        "cited": result.cited,
        "text": result.text,
        "length": len(result.text),
        "elapsed": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="大模型对比实验")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5:1.5b", "qwen2.5:7b"],
        help="要对比的模型名",
    )
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    print(f"检索配置：embedding={settings.embedding_model}  重排={settings.rerank_model}")
    print(f"对比模型：{', '.join(args.models)}")
    print()

    stats: dict[str, dict] = {}

    for model in args.models:
        print("=" * 70)
        print(f"模型：{model}")
        print("-" * 70)

        records = []
        for question, expected in CASES:
            record = run_one(model, question)
            record["expected"] = expected
            record["question"] = question

            # 判定：期望 answer 时未拒答即为通过；期望 refuse 时拒答即为通过
            if expected == "answer":
                record["pass"] = not record["refused"]
            else:
                record["pass"] = record["refused"]

            records.append(record)

            mark = "✅" if record["pass"] else "❌"
            mode = "拒答" if record["refused"] else "作答"
            cited = "有引用" if record["cited"] else "无引用"
            print(f"  {mark} {question[:22]:<24} {mode}  {cited}  {record['elapsed']:>5.1f}s")

        answered = [r for r in records if not r["refused"]]
        stats[model] = {
            "pass": sum(r["pass"] for r in records),
            "total": len(records),
            "cited": sum(r["cited"] for r in answered),
            "answered": len(answered),
            "median": statistics.median(r["elapsed"] for r in records),
            "avg_len": statistics.mean([r["length"] for r in answered]) if answered else 0,
        }
        print()

    # 汇总表
    print("=" * 70)
    print("汇总")
    print("-" * 70)
    header = f"{'模型':<18}{'行为正确':<12}{'引用标注':<14}{'延迟中位':<12}{'答案长度'}"
    print(header)
    for model, s in stats.items():
        cited_rate = f"{s['cited']}/{s['answered']}" if s["answered"] else "-"
        print(
            f"{model:<18}{s['pass']}/{s['total']:<10}{cited_rate:<14}"
            f"{s['median']:.1f}s{'':<6}{s['avg_len']:.0f} 字"
        )


if __name__ == "__main__":
    main()
