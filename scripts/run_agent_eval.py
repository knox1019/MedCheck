"""Agent 评测：测「该调用哪些工具」，而不是「答案对不对」。

和 run_answer_eval 的分工：
    run_answer_eval   固定流程（检索→生成），测答案质量
    run_agent_eval    Agent 模式，测工具选择与编排行为

为什么必须单独一套：
    Agent 的失效方式和固定流程不同。固定流程答错是「资料没找对」，
    Agent 还会「该查没查」「不该查乱查」「反复查同一个东西」。
    这些用答案指标看不出来——答案可能碰巧是对的，但过程不可复现。

五项指标，全部确定性判定：

    工具选择正确   期望的工具都调了，且没调不该调的
    行为正确       该答的答了、该拒的拒了
    引用标注率     答案里有 [n]，且编号能映射回真实片段
    多余调用       调用了既不在期望集、也不在白名单里的工具
    收敛率         没有撞上轮数上限

关于「额外调用」的判定：
    有些额外调用是合理的补救行为——比如 search_drug_label 返回「未收录」后，
    模型换 list_available_drugs 去确认药名。所以每条用例带 allow_extra 白名单，
    白名单里的工具调了不扣分。

用法：
    uv run python -m scripts.run_agent_eval
    uv run python -m scripts.run_agent_eval --only ag12 ag07    # 只跑指定用例
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from app.agent.loop import run
from app.agent.tools import ToolContext
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.ingest.pdf_parser import parse_report
from app.llm.client import ChatClient
from app.llm.prompts import extract_cited_numbers

DEFAULT_SET = Path("eval/agentset.jsonl")
DEFAULT_REPORT = Path("eval/report_agent.md")
RAW_RESULT = Path("eval/agent_results.json")
SAMPLE_PDF = Path("data/samples/report_demo.pdf")


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


def build_context(case: dict, report) -> ToolContext:
    """按用例声明的 context 装配上下文。

    用「声明式」而不是「根据问题文本猜」：评测集的每条用例都写清楚
    它处于什么场景，脚本不该再猜一遍。
    """
    kind = case.get("context", "none")
    medications = case.get("medications") or []
    if kind == "none":
        return ToolContext()
    if kind == "meds":
        return ToolContext(medications=medications)
    if kind == "report+meds":
        return ToolContext(report=report, medications=medications)
    raise ValueError(f"未知的 context 类型：{kind}")


def judge_tools(case: dict, steps: list) -> dict:
    """判断工具选择是否正确。

    missing  期望调了却没调 —— 该查没查，答案多半是编的
    extra    调了不该调的 —— 白费调用，且可能引入污染
    """
    used = {step.tool for step in steps}
    expected = set(case.get("expect_tools") or [])
    allowed_extra = set(case.get("allow_extra") or [])

    missing = sorted(expected - used)
    extra = sorted(used - expected - allowed_extra)
    return {
        "used": sorted(used),
        "expected": sorted(expected),
        "missing": missing,
        "extra": extra,
        "ok": not missing and not extra,
    }


def evaluate(cases: list[dict], model: str, report) -> dict:
    client = ChatClient(model=model)
    records: list[dict] = []

    for index, case in enumerate(cases, start=1):
        ctx = build_context(case, report)
        started = time.perf_counter()
        try:
            result = run(case["question"], ctx, client=client)
            failed = ""
        except Exception as exc:  # noqa: BLE001 —— 单条失败不该中断整轮评测
            result = None
            failed = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started

        if result is None:
            records.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "error": failed,
                    "tool_ok": False,
                    "behavior_ok": False,
                    "cited": False,
                    "refused": True,
                    "turns": 0,
                    "n_steps": 0,
                    "used": [],
                    "expected": case.get("expect_tools") or [],
                    "missing": case.get("expect_tools") or [],
                    "extra": [],
                    "converged": False,
                    "n_hits": 0,
                    "expect_refuse": case["expect_refuse"],
                    "elapsed": elapsed,
                    "answer": "",
                }
            )
            print(f"  [{index}/{len(cases)}] {case['id']} 异常：{failed}")
            continue

        tools = judge_tools(case, result.steps)
        behavior_ok = result.refused == case["expect_refuse"]
        numbers = extract_cited_numbers(result.answer)
        # 编号必须落在本次会话真实检索到的片段范围内
        cited = bool(numbers) and all(1 <= n <= len(ctx.hits) for n in numbers)

        records.append(
            {
                "id": case["id"],
                "question": case["question"],
                "error": "",
                "tool_ok": tools["ok"],
                "behavior_ok": behavior_ok,
                "cited": cited,
                "refused": result.refused,
                "turns": result.turns,
                "n_steps": len(result.steps),
                "used": tools["used"],
                "expected": tools["expected"],
                "missing": tools["missing"],
                "extra": tools["extra"],
                "converged": result.stopped != "max_turns",
                "n_hits": len(ctx.hits),
                "expect_refuse": case["expect_refuse"],
                "elapsed": elapsed,
                "answer": result.answer[:200],
            }
        )

        flag = "✓" if (tools["ok"] and behavior_ok) else "✗"
        print(
            f"  [{index}/{len(cases)}] {flag} {case['id']}  "
            f"工具={','.join(tools['used']) or '无'}  "
            f"轮数={result.turns}  {elapsed:.0f}s"
        )

    answered = [r for r in records if not r["refused"]]
    # 分母只算「有资料可引的作答」。寒暄这类没查资料的回答本来就没有出处可标，
    # 把它算进分母会让指标虚低——10/11 和 10/10 是两回事。
    citable = [r for r in answered if r["n_hits"] > 0]
    return {
        "model": model,
        "total": len(records),
        "tool_ok": sum(r["tool_ok"] for r in records),
        "behavior_ok": sum(r["behavior_ok"] for r in records),
        "cited": sum(r["cited"] for r in citable),
        "n_answered": len(answered),
        "n_citable": len(citable),
        "n_extra": sum(1 for r in records if r["extra"]),
        "converged": sum(r["converged"] for r in records),
        "avg_steps": statistics.mean(r["n_steps"] for r in records),
        "avg_turns": statistics.mean(r["turns"] for r in records),
        "latency_median": statistics.median(r["elapsed"] for r in records),
        "records": records,
    }


def rate(hit: int, total: int) -> str:
    return f"{hit}/{total} = {hit / total * 100:.1f}%" if total else "0/0"


def render_report(m: dict) -> str:
    lines = [
        "# Agent 评测报告",
        "",
        "> 由 `uv run python -m scripts.run_agent_eval` 自动生成，请勿手工修改。",
        "",
        f"- 模型：`{m['model']}`",
        f"- 评测集：{m['total']} 条",
        f"- 平均工具调用 {m['avg_steps']:.1f} 次 / 平均轮数 {m['avg_turns']:.1f}",
        "",
        "## 指标",
        "",
        "| 指标 | 含义 | 结果 |",
        "| --- | --- | --- |",
        f"| 工具选择正确 | 该调的调了，且没调不该调的 | {rate(m['tool_ok'], m['total'])} |",
        f"| 行为正确 | 该答的答了、该拒的拒了 | {rate(m['behavior_ok'], m['total'])} |",
        f"| 引用标注率 | 有资料可引的作答中，答案里有可映射的 [n] | {rate(m['cited'], m['n_citable'])} |",
        f"| 多余调用 | 调了既不在期望集也不在白名单的工具 | {m['n_extra']} 条 |",
        f"| 收敛率 | 没有撞上轮数上限 | {rate(m['converged'], m['total'])} |",
        f"| 延迟中位 | 单条耗时 | {m['latency_median']:.1f}s |",
        "",
        "## 逐条结果",
        "",
        "| id | 问题 | 期望工具 | 实际工具 | 轮数 | 行为 | 引用 | 结果 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in m["records"]:
        expect = "、".join(r["expected"]) or "无"
        used = "、".join(r["used"]) or "无"
        expect_cn = "拒答" if r["expect_refuse"] else "作答"
        behavior = "拒答" if r["refused"] else "作答"
        cited = "有" if r["cited"] else "无"
        ok = r["tool_ok"] and r["behavior_ok"]
        mark = "✅" if ok else "❌"
        if r["missing"]:
            mark += f" 缺 {'、'.join(r['missing'])}"
        if r["extra"]:
            mark += f" 多 {'、'.join(r['extra'])}"
        if not r["behavior_ok"]:
            mark += f" 应为{expect_cn}"
        lines.append(
            f"| {r['id']} | {r['question'][:24]} | {expect} | {used} | {r['turns']} | {behavior} | {cited} | {mark} |"
        )

    failures = [r for r in m["records"] if not (r["tool_ok"] and r["behavior_ok"])]
    if failures:
        lines += ["", "## 失败详情", ""]
        for r in failures:
            lines.append(f"### {r['id']} {r['question']}")
            lines.append("")
            lines.append(f"- 期望工具：{'、'.join(r['expected']) or '无'}")
            lines.append(f"- 实际工具：{'、'.join(r['used']) or '无'}")
            if r["missing"]:
                lines.append(f"- **缺：{'、'.join(r['missing'])}**")
            if r["extra"]:
                lines.append(f"- **多：{'、'.join(r['extra'])}**")
            lines.append(f"- 实际行为：{'拒答' if r['refused'] else '作答'}")
            if r["answer"]:
                lines.append(f"- 答案片段：{r['answer']}")
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 评测")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument("--model", default=None, help="默认取配置里的 llm_model")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--only", nargs="*", help="只跑指定 id")
    args = parser.parse_args()

    setup_logging()
    model = args.model or get_settings().llm_model
    cases = load_set(Path(args.set))
    if args.only:
        wanted = set(args.only)
        cases = [c for c in cases if c["id"] in wanted]

    # 报告只解析一次。18 条用例里 4 条需要它，重复解析纯属浪费。
    report = parse_report(SAMPLE_PDF) if SAMPLE_PDF.exists() else None
    if report is None:
        print("⚠️ 样本报告不存在，report+meds 用例会退化为无报告场景")

    print(f"评测集：{len(cases)} 条   模型：{model}")
    print("每条要跑 1-3 轮模型调用，请耐心等待……\n")

    metrics = evaluate(cases, model, report)

    Path(args.report).write_text(render_report(metrics), encoding="utf-8")
    Path(RAW_RESULT).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print("=" * 56)
    print(f"工具选择正确    {rate(metrics['tool_ok'], metrics['total'])}")
    print(f"行为正确        {rate(metrics['behavior_ok'], metrics['total'])}")
    print(f"引用标注率      {rate(metrics['cited'], metrics['n_citable'])}")
    print(f"多余调用        {metrics['n_extra']} 条")
    print(f"收敛率          {rate(metrics['converged'], metrics['total'])}")
    print(f"平均工具调用    {metrics['avg_steps']:.1f} 次 / 轮数 {metrics['avg_turns']:.1f}")
    print(f"延迟中位        {metrics['latency_median']:.1f}s")
    print("=" * 56)
    print(f"报告：{args.report}")

    failures = [r for r in metrics["records"] if not (r["tool_ok"] and r["behavior_ok"])]
    if failures:
        print(f"\n失败 {len(failures)} 条：")
        for r in failures:
            detail = []
            if r["missing"]:
                detail.append(f"缺 {'、'.join(r['missing'])}")
            if r["extra"]:
                detail.append(f"多 {'、'.join(r['extra'])}")
            if not r["behavior_ok"]:
                detail.append("行为不符")
            print(f"  {r['id']}  {r['question'][:30]}  → {'；'.join(detail)}")


if __name__ == "__main__":
    main()
