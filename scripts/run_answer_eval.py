"""答案级评测：不只测「检索对不对」，而是测「答案对不对」。

与 run_eval.py 的区别：
    run_eval.py      只看检索命中了哪个章节，不涉及大模型
    run_answer_eval  跑完整问答链路，评估答案本身

五项指标，全部是确定性判定，不引入「用模型评模型」的不可靠环节：

    行为正确      该答的答了、该拒的拒了
    引用标注率    答案里是否真的出现 [n] 标记
    引用有效率    引用的编号是否落在实际资料范围内
    关键点命中率  答案是否包含该题的标准关键词（在 goldset 里预先标注）
    药名幻觉率    答案提到的药名，是否根本不在本次提供的资料里

用法：
    uv run python -m scripts.run_answer_eval
    uv run python -m scripts.run_answer_eval --model qwen2.5:1.5b   # 换模型对比
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.llm.client import ChatClient
from app.llm.qa import answer
from app.llm.prompts import REFUSAL_PHRASE, extract_cited_numbers
from app.rag.guards import DRUG_ALIASES

DEFAULT_SET = Path("eval/answerset.jsonl")
DEFAULT_REPORT = Path("eval/report_answer.md")
RAW_RESULT = Path("eval/answer_results.json")


def load_set(path: Path) -> list[dict]:
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


def find_hallucinated_drugs(answer_text: str, context_text: str, question: str) -> list[str]:
    """找出答案里提到、但资料里没有的药名。

    医疗场景里最危险的幻觉是「凭空说出一个资料里没提的药」——
    用户会据此调整用药。所以这一项单独检测。

    判定方式：在答案中扫描已知药名词表，
    若该药没出现在提供给模型的资料里、也没出现在问题里，则判为幻觉。
    问题里提到的药要排除——用户自己说的不算模型编的。
    """
    hallucinated: list[str] = []
    for drug, aliases in DRUG_ALIASES.items():
        if drug in question:
            continue
        mentioned = any(alias in answer_text for alias in aliases)
        if mentioned and drug not in context_text:
            hallucinated.append(drug)
    return hallucinated


def evaluate(cases: list[dict], model: str) -> dict:
    settings = get_settings()
    client = ChatClient(model=model)

    records: list[dict] = []
    for case in cases:
        started = time.perf_counter()
        result = answer(case["question"], top_k=3, use_rerank=True, client=client)
        elapsed = time.perf_counter() - started

        # 送给模型的资料全文，用于幻觉检测
        context_text = "".join(hit["text"] for hit in result.hits)

        expect = case["expect"]
        if expect == "answer":
            behavior_ok = not result.refused
        else:
            behavior_ok = result.refused

        cited_numbers = extract_cited_numbers(result.text)
        # 引用编号必须在 1..len(hits) 范围内才算有效
        valid_cites = [n for n in cited_numbers if 1 <= n <= len(result.hits)]

        must_have = case.get("must_have") or []
        keypoint_hits = [kw for kw in must_have if kw in result.text]

        records.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expect": expect,
                "refused": result.refused,
                "answer": result.text,
                "behavior_ok": behavior_ok,
                "cited": bool(cited_numbers),
                "cites_valid": bool(cited_numbers) and len(valid_cites) == len(cited_numbers),
                "must_have": must_have,
                "keypoint_hits": keypoint_hits,
                "hallucinated": find_hallucinated_drugs(result.text, context_text, case["question"]),
                "elapsed": elapsed,
            }
        )

    answered = [r for r in records if not r["refused"]]
    should_answer = [r for r in records if r["expect"] == "answer"]
    answered_expected = [r for r in should_answer if not r["refused"]]

    keypoint_total = sum(len(r["must_have"]) for r in answered_expected)
    keypoint_ok = sum(len(r["keypoint_hits"]) for r in answered_expected)

    return {
        "model": model,
        "total": len(records),
        "behavior_ok": sum(r["behavior_ok"] for r in records),
        "n_should_answer": len(should_answer),
        "n_should_refuse": len(records) - len(should_answer),
        "cited": sum(r["cited"] for r in answered),
        "n_answered": len(answered),
        "cites_valid": sum(r["cites_valid"] for r in answered),
        "keypoint_ok": keypoint_ok,
        "keypoint_total": keypoint_total,
        "hallucinated_cases": sum(1 for r in answered if r["hallucinated"]),
        "latency_median": statistics.median(r["elapsed"] for r in records),
        "records": records,
    }


def rate(hit: int, total: int) -> str:
    return f"{hit}/{total} = {hit / total * 100:.1f}%" if total else "0/0"


def render_report(m: dict) -> str:
    lines = [
        "# 答案级评测报告",
        "",
        "> 由 `uv run python -m scripts.run_answer_eval` 自动生成，请勿手工修改。",
        "",
        f"- 模型：`{m['model']}`",
        f"- 评测集：{m['total']} 条（应作答 {m['n_should_answer']} 条 / 应拒答 {m['n_should_refuse']} 条）",
        f"- 检索延迟中包含大模型生成时间，中位 {m['latency_median']:.1f}s",
        "",
        "## 指标",
        "",
        "| 指标 | 含义 | 结果 |",
        "| --- | --- | --- |",
        f"| 行为正确 | 该答的答了、该拒的拒了 | {rate(m['behavior_ok'], m['total'])} |",
        f"| 引用标注率 | 作答时答案里出现 [n] 标记 | {rate(m['cited'], m['n_answered'])} |",
        f"| 引用有效率 | 引用的编号都在资料范围内 | {rate(m['cites_valid'], m['n_answered'])} |",
        f"| 关键点命中率 | 答案包含标准关键词 | {rate(m['keypoint_ok'], m['keypoint_total'])} |",
        f"| 药名幻觉 | 答案提到资料里没有的药 | {m['hallucinated_cases']} 条（{rate(m['hallucinated_cases'], m['n_answered'])}） |",
        "",
        "## 逐条结果",
        "",
        "| id | 问题 | 期望 | 实际 | 引用 | 关键点 | 幻觉 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in m["records"]:
        actual = "拒答" if r["refused"] else "作答"
        expect_cn = "作答" if r["expect"] == "answer" else "拒答"
        cited = "有" if r["cited"] else "无"
        keypoint = (
            f"{len(r['keypoint_hits'])}/{len(r['must_have'])}" if r["must_have"] else "-"
        )
        halluc = "、".join(r["hallucinated"]) if r["hallucinated"] else "-"
        mark = "" if r["behavior_ok"] else " ❌"
        lines.append(
            f"| {r['id']} | {r['question']} | {expect_cn} | {actual}{mark} | {cited} | {keypoint} | {halluc} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="答案级评测")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument("--model", default=None, help="默认取配置里的 llm_model")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    setup_logging()
    model = args.model or get_settings().llm_model
    cases = load_set(Path(args.set))

    print(f"评测集：{len(cases)} 条   模型：{model}")
    print("本地模型跑完整链路较慢，请耐心等待……")

    metrics = evaluate(cases, model)

    Path(args.report).write_text(render_report(metrics), encoding="utf-8")
    # 汇总与逐条明细写进同一个 JSON 对象。
    # 早期版本把两个 JSON 文档直接拼接，产出的文件不是合法 JSON，读取时会报 Extra data。
    Path(RAW_RESULT).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 56)
    print(f"行为正确        {rate(metrics['behavior_ok'], metrics['total'])}")
    print(f"引用标注率      {rate(metrics['cited'], metrics['n_answered'])}")
    print(f"引用有效率      {rate(metrics['cites_valid'], metrics['n_answered'])}")
    print(f"关键点命中率    {rate(metrics['keypoint_ok'], metrics['keypoint_total'])}")
    print(f"药名幻觉        {metrics['hallucinated_cases']} 条")
    print(f"延迟中位        {metrics['latency_median']:.1f}s")
    print("=" * 56)
    print(f"报告：{args.report}")

    failures = [r for r in metrics["records"] if not r["behavior_ok"]]
    if failures:
        print(f"\n行为错误 {len(failures)} 条：")
        for r in failures:
            print(f"  {r['id']}  {r['question']}")
            print(f"      期望{'作答' if r['expect'] == 'answer' else '拒答'} → 实际{'作答' if not r['refused'] else '拒答'}")


if __name__ == "__main__":
    main()
