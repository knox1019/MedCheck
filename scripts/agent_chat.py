"""Agent 对话命令行。

和 scripts/ask.py 的区别：ask 是固定流程（检索→生成），
agent_chat 由模型自己决定查什么、查几次。

    # 只问药物问题
    uv run python -m scripts.agent_chat "华法林和阿司匹林能一起吃吗"

    # 带上报告和在服药物，模型会在需要时自行核对指标
    uv run python -m scripts.agent_chat "我报告里转氨酶偏高，和吃的药有关系吗" ^
        --pdf data/samples/report_demo.pdf --meds 阿托伐他汀

需要 Ollama 在运行（向量化用），以及可用的对话模型。
"""

import argparse
import sys

from app.agent.loop import build_user_message, run
from app.agent.tools import ToolContext
from app.ingest.pdf_parser import parse_report
from app.llm.client import ChatClient


def split_meds(raw: str | None) -> list[str]:
    if not raw:
        return []
    for sep in ("，", "、", ";", "；"):
        raw = raw.replace(sep, ",")
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent 模式问答（模型自主调用工具）")
    parser.add_argument("question", help="要问的问题")
    parser.add_argument("--pdf", default=None, help="体检报告 PDF（可选）")
    parser.add_argument("--meds", default="", help="在服药物，逗号分隔（可选）")
    parser.add_argument("--model", default=None, help="覆盖默认大模型")
    parser.add_argument("--max-turns", type=int, default=6, help="工具调用轮数上限")
    parser.add_argument("--verbose", action="store_true", help="打印每步工具的返回内容")
    args = parser.parse_args()

    report = parse_report(args.pdf) if args.pdf else None
    medications = split_meds(args.meds)
    ctx = ToolContext(report=report, medications=medications)

    print("=" * 72)
    print(f"问题：{args.question}")
    if report:
        print(f"报告：{report.report_id}（{len(report.items)} 项）")
    if medications:
        print(f"用药：{'、'.join(medications)}")
    print("=" * 72)
    print("\n【上下文】")
    print(build_user_message(args.question, ctx))
    print("\n" + "-" * 72)

    result = run(
        args.question,
        ctx,
        client=ChatClient(model=args.model) if args.model else None,
        max_turns=args.max_turns,
    )

    print(f"\n【工具调用】共 {len(result.steps)} 次，{result.turns} 轮")
    for i, step in enumerate(result.steps, start=1):
        mark = "✗" if step.error else "✓"
        print(f"  {mark} {i}. {step.tool}({step.args})")
        head = step.output.strip().splitlines()
        preview = head[0][:70] if head else "（空）"
        print(f"      → {len(step.output)} 字：{preview}")
        if args.verbose and len(step.output) > 200:
            print("      " + step.output[200:600].replace("\n", "\n      "))

    print("\n【回答】")
    print(result.answer)

    if result.citations:
        print("\n【出处】")
        for i, citation in enumerate(result.citations, start=1):
            print(f"  [{i}] {citation.label}")
    elif result.steps:
        print("\n⚠️ 答案里没有引用编号，无法定位到具体章节")

    if result.stopped == "max_turns":
        print("\n⚠️ 未收敛：模型在轮数上限内没有给出最终答复")

    return 0


if __name__ == "__main__":
    sys.exit(main())
