"""报告核对命令行。

用法：

    uv run python -m scripts.report_check data/samples/report_demo.pdf --meds 阿托伐他汀,布洛芬
    uv run python -m scripts.report_check data/samples/report_demo.pdf          # 只看异常清单

需要 Ollama 在运行。只列异常清单时不调模型，不需要 Ollama。
"""

import argparse
import sys

from app.ingest.pdf_parser import parse_report
from app.llm.client import ChatClient
from app.report.analyzer import review
from app.report.narrative import narrate

_FLAG_TEXT = {"high": "偏高", "low": "偏低", "normal": "正常", "unknown": "未知"}


def split_meds(raw: str | None) -> list[str]:
    """切分药名列表，中英文逗号、顿号、分号都认。

    用户从聊天记录里复制药名时，分隔符经常是全角的——
    只认半角逗号会让「阿托伐他汀，布洛芬」被当成一个药名。
    """
    if not raw:
        return []
    for sep in ("，", "、", ";", "；"):
        raw = raw.replace(sep, ",")
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="体检报告 + 在服药物核对")
    parser.add_argument("pdf", help="体检报告 PDF 路径")
    parser.add_argument("--meds", default="", help="在服药物，逗号分隔")
    parser.add_argument("--model", default=None, help="覆盖默认大模型")
    parser.add_argument("--no-llm", action="store_true", help="只输出核对结果，不调模型")
    args = parser.parse_args()

    report = parse_report(args.pdf)
    medications = split_meds(args.meds)
    result = review(report, medications)

    print("=" * 72)
    print(f"报告编号：{report.report_id}    采样日期：{report.exam_date or '未识别'}")
    print(f"指标总数：{len(report.items)}    异常指标：{len(result.abnormal)}")
    if medications:
        print(f"在服药物：{'、'.join(medications)}")
    if result.unsupported:
        print(f"⚠️ 未收录（无法核对）：{'、'.join(result.unsupported)}")
    print("=" * 72)

    print("\n【异常指标】")
    for order, finding in enumerate(result.findings, start=1):
        item = finding.item
        unit = f" {item.unit}" if item.unit else ""
        flag = _FLAG_TEXT.get(item.flag, item.flag)
        mark = "  ← 说明书有相关记载" if finding.has_evidence else ""
        print(f"  {order:>2}. {finding.label:<12}{item.value:>8g}{unit:<9}{flag}{mark}")

    if result.with_evidence:
        print("\n【说明书依据】")
        for finding in result.with_evidence:
            print(f"\n  ▸ {finding.label}")
            for evidence in finding.evidences:
                print(f"    · {evidence.drug} · {evidence.section}（命中：{'、'.join(evidence.matched)}）")
                print(f"      {evidence.text}")

    if args.no_llm or not medications:
        if not medications:
            print("\n（未提供在服药物，只列异常清单。加 --meds 可做用药核对）")
        return 0

    print("\n【解读】")
    narrative = narrate(result, client=ChatClient(model=args.model) if args.model else None)
    print(narrative.text)

    if narrative.citations:
        print("\n【出处】")
        # 用答案里的原始编号，不重新从 1 编——否则答案写「见 [2]」、出处列「[1]」
        numbers = narrative.cited_numbers or list(range(1, len(narrative.citations) + 1))
        for number, citation in zip(numbers, narrative.citations):
            print(f"  [{number}] {citation.label}")
    if not narrative.cited and not narrative.refused:
        print("\n⚠️ 答案中未出现引用标记，出处为候选资料全量列出")

    return 0


if __name__ == "__main__":
    sys.exit(main())
