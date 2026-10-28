"""HTTP 接口的数据契约。

请求和响应都用 Pydantic 模型定义，不直接返回 dict。三个理由：

    1. /docs 自动生成的接口文档会带出字段说明，调用方不用猜
    2. 响应字段被改掉时，测试会立刻发现——裸 dict 改错了没人知道
    3. 请求体自动校验，question 传空字符串会在入口就被挡住
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.ingest.models import Report
from app.rag.citation import Citation


class ParseResponse(BaseModel):
    """解析结果。"""

    report: Report


class ChatRequest(BaseModel):
    """一次问答请求。

    报告和在服药物都是可选的：不传就是纯药物问答，传了才做指标核对。
    接口设计成无状态——报告由调用方持有并回传，服务端不存会话。
    这样多实例部署时不需要共享存储，也不会因为重启丢数据。
    """

    question: str = Field(min_length=1, max_length=500, description="用户问题")
    report: Report | None = Field(
        default=None, description="体检报告解析结果，可选。传入后会核对指标与用药的关联"
    )
    medications: list[str] = Field(
        default_factory=list, description="在服药物列表，可选"
    )
    mode: Literal["auto", "qa", "agent"] = Field(
        default="auto",
        description=(
            "auto：带报告或用药时走 Agent，否则走固定问答流程；"
            "qa：强制走固定流程（快）；"
            "agent：强制走 Agent（模型自主调用工具）"
        ),
    )


class CitationOut(BaseModel):
    """一条出处。"""

    source: str
    section: str | None = None
    page: int | None = None
    label: str = Field(description="可直接展示的一行文本，如「华法林钠片说明书 · 禁忌」")
    number: int | None = Field(
        default=None,
        description=(
            "答案里使用的原始编号。多个编号可能指向同一出处（同一章节被多个指标引用），"
            "此时 citations 去重后只剩一条，但正文里仍写着 [2][4][6]。"
            "调用方应优先用它显示，而不是按数组下标重新编号。"
        ),
    )

    @classmethod
    def from_citation(cls, citation: Citation, number: int | None = None) -> "CitationOut":
        return cls(
            source=citation.source,
            section=citation.section,
            page=citation.page,
            label=citation.label,
            number=number,
        )


class StepOut(BaseModel):
    """Agent 模式下的一次工具调用，用于让调用方看到「它查了什么」。"""

    tool: str
    args: dict = Field(default_factory=dict)
    chars: int = Field(description="工具返回内容的字数")
    error: bool = False


class ChatResponse(BaseModel):
    """一次问答结果。"""

    answer: str
    refused: bool = Field(description="是否因资料不足而拒答")
    reason: str = Field(default="", description="拒答或命中的原因，便于排查")
    citations: list[CitationOut] = Field(default_factory=list)
    mode: Literal["qa", "agent"] = Field(description="实际走的路径，未必等于请求里的 mode")
    turns: int = Field(default=0, description="Agent 模式下的模型往返轮数")
    steps: list[StepOut] = Field(default_factory=list, description="Agent 模式的工具调用记录")
