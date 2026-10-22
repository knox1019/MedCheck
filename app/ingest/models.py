"""体检报告的数据模型。

这个文件只负责"数据长什么样"，不负责"数据从哪来"。
从 PDF 里取数据的逻辑在 pdf_parser.py。
"""

from pydantic import BaseModel


class LabItem(BaseModel):
    """一条检验指标。"""

    name_raw: str  # PDF 上的原始名称，保留作为核对证据
    name: str | None = None  # 归一化后的名称（如 ALT），W3 才填
    value: float  # 检验结果，必须是数字，下游要比大小
    unit: str | None = None  # 单位，如 U/L
    ref_low: float | None = None  # 参考范围下限，<8 这类单边范围时为 None
    ref_high: float | None = None  # 参考范围上限
    flag: str = "unknown"  # high / low / normal / unknown，由 value 和参考范围算出
    page: int = 1  # 该指标所在的 PDF 页码，用于引用溯源


class Report(BaseModel):
    """一份完整的检验报告。"""

    report_id: str  # 条码号，取不到时退化为文件名
    exam_date: str | None = None  # 采样日期，格式 YYYY-MM-DD
    items: list[LabItem]
