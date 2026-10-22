"""命令行入口：解析一份报告 PDF 并打印结构化结果。

用法：
    uv run python scripts/parse_report.py data/samples/report_demo.pdf
"""

import json
import sys
from pathlib import Path

from app.ingest.pdf_parser import parse_report


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/report_demo.pdf"

    if not Path(path).exists():
        print(f"找不到文件：{path}")
        raise SystemExit(1)

    report = parse_report(path)

    # model_dump_json 已经产出 JSON 字符串，这里转回 dict 只为控制缩进、让中文不转义
    print(json.dumps(json.loads(report.model_dump_json()), ensure_ascii=False, indent=2))

    abnormal = [item for item in report.items if item.flag in ("high", "low")]
    unknown = [item for item in report.items if item.flag == "unknown"]

    print()
    print(f"报告编号：{report.report_id}")
    print(f"采样日期：{report.exam_date}")
    print(f"指标总数：{len(report.items)}")
    print(f"异常指标：{len(abnormal)}")
    print(f"无法判断：{len(unknown)}")


if __name__ == "__main__":
    main()
