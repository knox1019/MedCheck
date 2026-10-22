"""PDF 解析模块的单元测试。

分三层，从下往上：
    1. parse_ref   纯字符串处理，最容易测，覆盖所有格式分支
    2. calc_flag   纯数值判断，重点测边界值和单边范围
    3. parse_row   需要构造假的行数据，重点测「跳过坏行」和「不依赖提示列」
    4. 集成       解析真实样本 PDF，校验总数与异常数

测试的重点不是「证明代码现在是对的」，而是「以后改动时能立刻发现改坏了没有」。
"""

from pathlib import Path

import pytest

from app.ingest.pdf_parser import calc_flag, parse_ref, parse_report, parse_row

SAMPLE_PDF = Path("data/samples/report_demo.pdf")


# ---------------------------------------------------------------- parse_ref

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("57-97", (57.0, 97.0)),        # 常见双边范围
        ("3.9-11.1", (3.9, 11.1)),      # 带小数
        ("9 - 50", (9.0, 50.0)),        # 中间有空格
        ("0.5~1.2", (0.5, 1.2)),        # 用波浪线而不是横线
        ("<8", (None, 8.0)),            # 只有上限
        (">5", (5.0, None)),            # 只有下限
        ("", (None, None)),             # 空字符串
        (None, (None, None)),           # None（pdfplumber 的空单元格）
        ("   ", (None, None)),          # 只有空格
        ("阴性", (None, None)),          # 解析不出来的文本
    ],
)
def test_parse_ref(raw, expected):
    """覆盖参考范围的三种合法写法和四种脏数据。"""
    assert parse_ref(raw) == expected


# --------------------------------------------------------------- calc_flag

@pytest.mark.parametrize(
    "value, low, high, expected",
    [
        (68, 9, 50, "high"),            # 高于上限
        (15.2, 20, 50, "low"),          # 低于下限
        (5.43, 3.9, 6.1, "normal"),     # 区间内
        (50, 9, 50, "normal"),          # 边界值：等于上限不算异常
        (9, 9, 50, "normal"),           # 边界值：等于下限不算异常
        (11.2, None, 8, "high"),        # 只有上限
        (2.0, 5, None, "low"),          # 只有下限
        (5.0, None, None, "unknown"),   # 参考范围完全缺失
    ],
)
def test_calc_flag(value, low, high, expected):
    """重点覆盖两个容易写错的地方：边界值、单边范围。"""
    assert calc_flag(value, low, high) == expected


# ---------------------------------------------------------------- parse_row

def make_row(name, value, unit="", ref="", hint="", fill=15):
    """按真实报告的列布局造一行假数据。

    真实布局是每列横跨 3 个物理格，值只填第一格：
        0=项目名称  3=结果  6=单位  9=参考范围  12=提示
    中间那些格子保持 None，和 pdfplumber 的真实输出一致。
    """
    row = [None] * fill
    row[0] = name
    row[3] = value
    row[6] = unit
    row[9] = ref
    row[12] = hint
    return row


def test_parse_row_normal():
    """正常数据行应该解析成完整的 LabItem。"""
    item = parse_row(make_row("丙氨酸氨基转移酶 ALT", "68", "U/L", "9-50", "H"), page=2)

    assert item is not None
    assert item.name_raw == "丙氨酸氨基转移酶 ALT"
    assert item.value == 68.0
    assert item.unit == "U/L"
    assert item.ref_low == 9.0
    assert item.ref_high == 50.0
    assert item.flag == "high"
    assert item.page == 2


def test_parse_row_ignores_hint_column():
    """注入矛盾数据：提示列写 ↑，但数值在正常范围内。

    解析结果必须判为 normal。如果这条测试失败，说明实现偷偷读了提示列——
    而那正是我们刻意要避免的（真实报告里标记经常缺失或印错）。
    """
    item = parse_row(make_row("丙氨酸氨基转移酶 ALT", "30", "U/L", "9-50", "↑"))

    assert item is not None
    assert item.flag == "normal"


def test_parse_row_skips_header():
    """表头行应被跳过。注意「项目」在索引 1，不在索引 0。"""
    header = [None] * 15
    header[1] = "项目名称"
    header[4] = "结果"

    assert parse_row(header) is None


def test_parse_row_skips_empty_row():
    """整行为空时应被跳过。"""
    assert parse_row([None] * 15) is None


def test_parse_row_skips_non_numeric_value():
    """结果不是数字（如「阴性」）时跳过该行，而不是让整个解析崩掉。"""
    assert parse_row(make_row("尿蛋白", "阴性", "", "")) is None


def test_parse_row_skips_short_row():
    """列数不足时跳过，避免 IndexError。"""
    assert parse_row(["只有一格"]) is None


def test_parse_row_handles_empty_unit():
    """单位为空的单元格是 None，应转成 None 而不是空字符串。"""
    item = parse_row(make_row("白细胞计数 WBC", "6.82", "", "3.5-9.5"))

    assert item is not None
    assert item.unit is None


# ------------------------------------------------------------------- 集成测试

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="缺少样本 PDF，跳过集成测试")
def test_parse_report_on_sample():
    """解析真实样本，校验条数与异常数。

    这两个数字一旦变化，通常意味着解析逻辑被改坏了：
        21 条指标、11 条异常、0 条 unknown
    """
    report = parse_report(SAMPLE_PDF)

    assert report.report_id == "20261022A0031"
    assert report.exam_date == "2026-10-22"
    assert len(report.items) == 21

    abnormal = [item for item in report.items if item.flag in ("high", "low")]
    unknown = [item for item in report.items if item.flag == "unknown"]

    assert len(abnormal) == 11
    assert len(unknown) == 0


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="缺少样本 PDF，跳过集成测试")
def test_parse_report_single_sided_range():
    """C 反应蛋白的参考范围是 <8，属于单边范围，下限必须为 None。"""
    report = parse_report(SAMPLE_PDF)
    crp = next(item for item in report.items if "CRP" in item.name_raw)

    assert crp.value == 11.2
    assert crp.ref_low is None
    assert crp.ref_high == 8.0
    assert crp.flag == "high"
