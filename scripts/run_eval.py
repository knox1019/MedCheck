"""检索评测：跑评测集，输出指标与 markdown 报告。

用法：
    uv run python -m scripts.run_eval                    # 用默认评测集
    uv run python -m scripts.run_eval --min-sim 0.6      # 调阈值做对比实验

设计要点：
    1. 指标分「药品命中」和「药品+章节命中」两种。
       章节判定偏严格——问「XX 注意事项」时检索到「XX 禁忌」其实也有价值，
       所以两个指标都看，不只看严格的那个。
    2. 拒答单独统计，不和检索混在一起算。
       否则「全部拒答」的策略会得到一个很好看的总分。
    3. 每次跑都把结果写成 markdown，便于横向对比不同方案的差异。
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.retriever import DEFAULT_MIN_SIMILARITY, retrieve
from app.rag.store import get, get_client

DEFAULT_GOLDSET = Path("eval/goldset.jsonl")
DEFAULT_REPORT = Path("eval/report.md")


def load_goldset(path: Path) -> list[dict]:
    cases: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
    return cases


def evaluate(cases: list[dict], top_k: int = 3, min_similarity: float | None = None, use_rerank: bool = False) -> dict:
    """跑完整评测集，返回各项指标。"""
    settings = get_settings()
    collection = get(get_client(settings.chroma_dir))

    answerable = [c for c in cases if c["type"] == "answerable"]
    unsupported = [c for c in cases if c["type"] == "refuse_unsupported"]
    offtopic = [c for c in cases if c["type"] == "refuse_offtopic"]

    drug_hit_1 = 0
    pair_hit_1 = 0
    pair_hit_3 = 0
    same_drug_1 = 0
    latencies: list[float] = []
    failures: list[dict] = []

    for case in answerable:
        started = time.perf_counter()
        result = retrieve(
            case["query"], top_k=top_k, min_similarity=min_similarity,
            collection=collection, use_rerank=use_rerank,
        )
        latencies.append((time.perf_counter() - started) * 1000)

        # 一个问题可能有多个章节都算答对，例如「甲硝唑喝酒」：
        # 相互作用章节讲机制，注意事项章节给了停药窗口和具体症状，两章都对。
        # 只允许一个标准答案会低估系统表现。
        acceptable = set(case.get("acceptable_sections") or [case["expected_section"]])
        expected_drug = case["expected_drug"]

        got_pairs = [
            (hit["metadata"]["drug"], hit["metadata"]["section"]) for hit in result.hits
        ]

        top_drug_ok = bool(result.hits) and got_pairs[0][0] == expected_drug
        top_pair_ok = bool(got_pairs) and got_pairs[0][0] == expected_drug and got_pairs[0][1] in acceptable
        in_top3 = any(drug == expected_drug and section in acceptable for drug, section in got_pairs)

        drug_hit_1 += top_drug_ok
        pair_hit_1 += top_pair_ok
        pair_hit_3 += in_top3
        # 药对了但章节不对：属于「有资料但定位偏」，比完全错好，单独统计
        same_drug_1 += top_drug_ok and not top_pair_ok

        if not top_pair_ok:
            failures.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "expected": f"{expected_drug}·{'/'.join(sorted(acceptable))}",
                    "got": f"{got_pairs[0][0]}·{got_pairs[0][1]}" if got_pairs else "无结果",
                    "answerable": result.answerable,
                }
            )

    unsupported_ok = 0
    for case in unsupported:
        result = retrieve(
            case["query"], top_k=top_k, min_similarity=min_similarity,
            collection=collection, use_rerank=use_rerank,
        )
        if not result.answerable and "未收录" in result.reason:
            unsupported_ok += 1
        else:
            failures.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "expected": f"拒答（未收录 {case['unsupported_drug']}）",
                    "got": "放行" if result.answerable else result.reason,
                    "answerable": result.answerable,
                }
            )

    offtopic_ok = 0
    for case in offtopic:
        result = retrieve(
            case["query"], top_k=top_k, min_similarity=min_similarity,
            collection=collection, use_rerank=use_rerank,
        )
        if not result.answerable:
            offtopic_ok += 1
        else:
            failures.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "expected": "拒答（跑题）",
                    "got": "放行",
                    "answerable": result.answerable,
                }
            )

    n_answerable = len(answerable)
    n_refuse = len(unsupported) + len(offtopic)
    correct = pair_hit_1 + unsupported_ok + offtopic_ok

    return {
        "total": len(cases),
        "n_answerable": n_answerable,
        "n_refuse": n_refuse,
        "drug_hit_1": drug_hit_1,
        "pair_hit_1": pair_hit_1,
        "pair_hit_3": pair_hit_3,
        "same_drug_1": same_drug_1,
        "unsupported_ok": unsupported_ok,
        "n_unsupported": len(unsupported),
        "offtopic_ok": offtopic_ok,
        "n_offtopic": len(offtopic),
        "overall_correct": correct,
        "latency_median_ms": statistics.median(latencies) if latencies else 0,
        "latency_max_ms": max(latencies) if latencies else 0,
        "failures": failures,
        "min_similarity": DEFAULT_MIN_SIMILARITY if min_similarity is None else min_similarity,
        "use_rerank": use_rerank,
    }


def rate(hit: int, total: int) -> str:
    return f"{hit}/{total} = {hit / total * 100:.1f}%" if total else "0/0"


def render_report(metrics: dict) -> str:
    n_a = metrics["n_answerable"]
    lines = [
        "# 检索评测报告",
        "",
        "> 由 `uv run python -m scripts.run_eval` 自动生成，请勿手工修改。",
        "",
        f"- 评测集：{metrics['total']} 条（可答 {n_a} 条 / 应拒答 {metrics['n_refuse']} 条）",
        f"- 相似度阈值：{metrics['min_similarity']}",
        f"- 重排：{'已开启（bge-reranker-v2-m3-int8）' if metrics['use_rerank'] else '未开启'}",
        "",
        "## 检索指标（仅可答用例）",
        "",
        "| 指标 | 含义 | 结果 |",
        "| --- | --- | --- |",
        f"| 药品命中@1 | top-1 命中了正确的药品 | {rate(metrics['drug_hit_1'], n_a)} |",
        f"| 药品+章节命中@1 | top-1 的药品和章节都对 | {rate(metrics['pair_hit_1'], n_a)} |",
        f"| 药品+章节命中@3 | top-3 里包含正确的药品和章节 | {rate(metrics['pair_hit_3'], n_a)} |",
        f"| 药对但章节偏 | top-1 药品正确、章节不在可接受范围内 | {rate(metrics['same_drug_1'], n_a)} |",
        "",
        "## 拒答指标",
        "",
        "| 指标 | 含义 | 结果 |",
        "| --- | --- | --- |",
        f"| 库外药品拒答 | 问题里的药不在库时正确拒答 | {rate(metrics['unsupported_ok'], metrics['n_unsupported'])} |",
        f"| 跑题拒答 | 与医疗无关的问题正确拒答 | {rate(metrics['offtopic_ok'], metrics['n_offtopic'])} |",
        "",
        "## 总体",
        "",
        f"- **完全正确：{rate(metrics['overall_correct'], metrics['total'])}**",
        f"  （可答用例要求药品+章节命中@1，拒答用例要求正确拒答）",
        f"- 检索延迟：中位 {metrics['latency_median_ms']:.0f} ms / 最大 {metrics['latency_max_ms']:.0f} ms",
        "",
    ]

    if metrics["failures"]:
        lines += [
            "## 失败用例",
            "",
            "| id | 查询 | 期望 | 实际 |",
            "| --- | --- | --- | --- |",
        ]
        for failure in metrics["failures"]:
            lines.append(
                f"| {failure['id']} | {failure['query']} | {failure['expected']} | {failure['got']} |"
            )
        lines.append("")
    else:
        lines += ["## 失败用例", "", "无。", ""]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="检索评测")
    parser.add_argument("--goldset", default=str(DEFAULT_GOLDSET))
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--min-sim", type=float, default=None)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--rerank", action="store_true", help="启用 cross-encoder 重排")
    args = parser.parse_args()

    setup_logging()

    cases = load_goldset(Path(args.goldset))
    print(f"评测集：{len(cases)} 条")

    metrics = evaluate(cases, top_k=args.top, min_similarity=args.min_sim, use_rerank=args.rerank)
    report = render_report(metrics)
    Path(args.report).write_text(report, encoding="utf-8")

    print()
    print("=" * 52)
    print(f"药品命中@1        {rate(metrics['drug_hit_1'], metrics['n_answerable'])}")
    print(f"药品+章节命中@1    {rate(metrics['pair_hit_1'], metrics['n_answerable'])}")
    print(f"药品+章节命中@3    {rate(metrics['pair_hit_3'], metrics['n_answerable'])}")
    print(f"药对但章节偏       {rate(metrics['same_drug_1'], metrics['n_answerable'])}")
    print("-" * 52)
    print(f"库外药品拒答       {rate(metrics['unsupported_ok'], metrics['n_unsupported'])}")
    print(f"跑题拒答          {rate(metrics['offtopic_ok'], metrics['n_offtopic'])}")
    print("-" * 52)
    print(f"总体完全正确       {rate(metrics['overall_correct'], metrics['total'])}")
    print(f"检索延迟中位数     {metrics['latency_median_ms']:.0f} ms")
    print("=" * 52)
    print(f"报告已写入：{args.report}")

    if metrics["failures"]:
        print(f"\n失败 {len(metrics['failures'])} 条：")
        for failure in metrics["failures"]:
            print(f"  {failure['id']}  {failure['query']}")
            print(f"      期望 {failure['expected']} → 实际 {failure['got']}")


if __name__ == "__main__":
    main()
