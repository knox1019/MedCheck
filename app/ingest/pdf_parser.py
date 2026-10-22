"""把检验报告 PDF 解析成结构化的 Report 对象。

整体流程分两段：

    1. PyMuPDF 抽正文  → 用正则取「条码号」「采样时间」这类元信息
    2. pdfplumber 抽表格 → 逐行转成 LabItem

为什么用两个库：PyMuPDF 抽正文干净但表格会被拆成散行，
pdfplumber 能识别表格结构但正文容易乱。各取所长。

⚠️ 模板假设：本模块按「杭州市第一人民医院（钱塘院区）检验报告单」的列布局编写，
   逻辑列索引为 0=项目名称、3=结果、6=单位、9=参考范围。
   换一家医院的报告，列索引需要重新探测。自适应解析属于后续版本。
"""

import re
from pathlib import Path

import pdfplumber
import pymupdf

from app.ingest.models import LabItem, Report

# 表格列索引。数字看起来很奇怪（0/3/6/9），是因为原 PDF 每列横跨 3 个物理格，
# 中间两格是合并单元格产生的 None。
COL_NAME = 0
COL_VALUE = 3
COL_UNIT = 6
COL_REF = 9


def extract_exam_date(text: str) -> str | None:
    """从正文里取采样日期。

    报告里写的是「采样时间：2026-10-22 08:12」，只取日期部分。
    取不到返回 None，不抛异常——元信息缺失不该让整个解析失败。
    """
    match = re.search(r"采样时间：(\d{4}-\d{2}-\d{2})", text)
    return match.group(1) if match else None


def extract_report_id(text: str) -> str | None:
    """从正文里取条码号，作为报告的唯一标识。"""
    match = re.search(r"条码号：(\S+)", text)
    return match.group(1) if match else None


def parse_ref(text: str) -> tuple[float | None, float | None]:
    """把参考范围字符串拆成 (下限, 上限)。

    支持三种写法：
        "57-97"  → (57.0, 97.0)     双边范围
        "<8"     → (None, 8.0)      只有上限
        ">5"     → (5.0, None)      只有下限

    解析不出来时返回 (None, None)，由 calc_flag 判为 unknown。
    宁可标记为未知，也不要瞎猜一个范围。
    """
    text = (text or "").strip()
    if not text:
        return None, None

    # 单边范围必须放在双边正则之前判断，
    # 否则 "<8" 用双边正则匹配不上，会直接落到最后的 return None, None
    if text.startswith("<"):
        number = re.search(r"[\d.]+", text)
        return None, float(number.group()) if number else None

    if text.startswith(">"):
        number = re.search(r"[\d.]+", text)
        return float(number.group()) if number else None, None

    match = re.match(r"([\d.]+)\s*[-~]\s*([\d.]+)", text)
    if match:
        return float(match.group(1)), float(match.group(2))

    return None, None


def calc_flag(value: float, ref_low: float | None, ref_high: float | None) -> str:
    """根据数值和参考范围判断异常状态。

    注意：这里刻意不使用 PDF 上的提示列（↑ / H / L）。
    原因有三：
        1. 有些报告只标红不写字，PDF 里没有文字标记
        2. 同一家医院可能混用 ↑ 和 H
        3. 标记本身可能印错
    自己算才是唯一可靠的做法。
    """
    is_low_enabled = ref_low is not None
    is_high_enabled = ref_high is not None

    if is_high_enabled and value > ref_high:
        return "high"

    if is_low_enabled and value < ref_low:
        return "low"

    # 两个边界都没有，说明参考范围没解析出来，无法判断
    if not is_low_enabled and not is_high_enabled:
        return "unknown"

    return "normal"


def parse_row(row: list, page: int = 1) -> LabItem | None:
    """把表格的一行转成一个 LabItem；不是有效数据行时返回 None。

    返回 None 的三种情况：表头行、空行、取不到数值的行。
    由上层过滤掉，这样单个坏行不会毁掉整份报告。
    """
    # 关键一步：把 None 统一换成空字符串再 strip。
    # pdfplumber 抽出来的空单元格是 None 不是 ""，直接 strip 会 AttributeError。
    cells = [(cell or "").strip() for cell in row]

    # 表头行：表头第一格是空的，「项目」在第 1 格，所以必须扫整行
    if any("项目" in cell for cell in cells):
        return None

    # 防御：列数不足时直接跳过，避免 IndexError
    if len(cells) <= COL_REF:
        return None

    name_raw = cells[COL_NAME]
    if not name_raw:
        return None

    # 从结果列里抠出数字。报告里可能是 "68"、"21.3"、"11.20"，
    # 也可能是 "58 ↑" 这种带标记的，所以用 re.search 而不是 float() 直接转
    number = re.search(r"-?\d+\.?\d*", cells[COL_VALUE])
    if not number:
        return None

    value = float(number.group())
    ref_low, ref_high = parse_ref(cells[COL_REF])

    return LabItem(
        name_raw=name_raw,
        value=value,
        unit=cells[COL_UNIT] or None,  # 空字符串统一转成 None，语义更准确
        ref_low=ref_low,
        ref_high=ref_high,
        flag=calc_flag(value, ref_low, ref_high),
        page=page,
    )


def parse_report(path: str | Path) -> Report:
    """解析一份 PDF 报告，返回 Report。"""
    path = Path(path)

    # 第一段：抽正文，拿元信息
    doc = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()  # 必须手动关，否则 Windows 上文件会被占用

    # 条码号取不到就用文件名兜底，保证 report_id 这个必填字段永远有值
    report_id = extract_report_id(text) or path.stem
    exam_date = extract_exam_date(text)

    # 第二段：抽表格，逐行转 LabItem
    items: list[LabItem] = []
    with pdfplumber.open(path) as pdf:
        # start=1 让页码和 PDF 阅读器一致，以后引用溯源要说「第 1 页」
        for page_no, page in enumerate(pdf.pages, start=1):
            # 该页没有表格时 extract_tables() 可能返回 None，用 or [] 兜住
            for table in page.extract_tables() or []:
                for row in table:
                    item = parse_row(row, page_no)
                    if item is not None:
                        items.append(item)

    return Report(report_id=report_id, exam_date=exam_date, items=items)
