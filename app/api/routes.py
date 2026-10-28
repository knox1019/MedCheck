"""HTTP 路由。

两个接口：

    POST /api/report/parse   上传 PDF → 结构化指标
    POST /api/chat           提问 → 答案 + 出处

三个刻意的设计：

    1. 无状态。报告由调用方持有并回传，服务端不存会话。
       多实例部署时不用共享存储，重启也不丢数据。

    2. 同步路由（def 而不是 async def）用于问答。
       大模型调用是阻塞的，写成 async 会卡住事件循环，
       导致整个服务在等模型时无法响应其他请求。
       用 def 时 FastAPI 会自动把它放进线程池。

    3. 模型不可用返回 503 而不是 500。
       503 表示「依赖服务挂了，稍后重试可能有用」，
       500 表示「服务端代码有 bug」——这两种情况运维处理方式完全不同。
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.agent.loop import run as run_agent
from app.agent.tools import ToolContext
from app.api.schemas import ChatRequest, ChatResponse, CitationOut, ParseResponse, StepOut
from app.ingest.pdf_parser import parse_report
from app.llm.qa import answer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["api"])

# 上传大小上限。体检报告通常几百 KB，20MB 足够宽松，
# 同时挡住误传的大文件把内存吃满。
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@router.post(
    "/report/parse",
    response_model=ParseResponse,
    summary="解析体检报告 PDF",
    response_description="结构化指标，含异常判定与页码",
)
async def parse_report_endpoint(
    file: UploadFile = File(..., description="体检报告 PDF"),
) -> ParseResponse:
    """把 PDF 转成结构化指标。

    取不到条码号时，用上传时的文件名作为 report_id——
    临时文件名会变成 random string，对调用方没有意义。
    """
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 上限",
        )

    # 用原始文件名的 stem 当临时文件名：pdf_parser 在取不到条码号时
    # 会用 path.stem 兜底，临时随机名会污染 report_id。
    #
    # ignore_cleanup_errors 是必须的：解析失败时（比如坏 PDF）pymupdf 可能
    # 还持有文件句柄，Windows 上删除会抛 PermissionError，
    # 把本该返回的 422 覆盖成 500。清理失败不该影响业务结果。
    stem = Path(filename).stem or "upload"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as workdir:
        path = Path(workdir) / f"{stem}.pdf"
        path.write_bytes(data)
        try:
            report = parse_report(path)
        except Exception as exc:  # noqa: BLE001 —— 解析失败的原因很多，统一转成 422
            logger.warning("解析失败 %s: %s", filename, exc)
            raise HTTPException(
                status_code=422, detail=f"无法解析该 PDF：{type(exc).__name__}: {exc}"
            ) from exc

    logger.info("解析完成 %s → %s 项指标", filename, len(report.items))
    return ParseResponse(report=report)


def _run_qa(question: str) -> ChatResponse:
    result = answer(question)
    return ChatResponse(
        answer=result.text,
        refused=result.refused,
        reason=result.reason,
        citations=[CitationOut.from_citation(c) for c in result.citations],
        mode="qa",
    )


def _run_agent(req: ChatRequest) -> ChatResponse:
    ctx = ToolContext(report=req.report, medications=req.medications)
    result = run_agent(req.question, ctx)

    # 出处要带上答案里的原始编号：同一章节可能被多个指标引用，
    # citations 去重后只剩一条，但正文里仍写着 [2][4][6]。
    numbers = result.cited_numbers or list(range(1, len(result.citations) + 1))
    return ChatResponse(
        answer=result.answer,
        refused=result.refused,
        reason="" if not result.refused else "资料不足以回答",
        citations=[
            CitationOut.from_citation(citation, number)
            for number, citation in zip(numbers, result.citations)
        ],
        mode="agent",
        turns=result.turns,
        steps=[
            StepOut(tool=s.tool, args=s.args, chars=len(s.output), error=s.error)
            for s in result.steps
        ],
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="问答（支持携带报告与用药）",
    response_description="答案、出处，以及 Agent 模式下的工具调用记录",
)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """提问并拿到带出处的答案。

    走哪条路：

        带了报告或用药  → Agent（模型自己决定查几个药、查哪一章）
        什么都没带      → 固定流程（检索一次，快）

    纯药物问题不需要 Agent——多一次模型决策只会更慢，不会更准。
    """
    use_agent = req.mode == "agent" or (
        req.mode == "auto" and (req.report is not None or req.medications)
    )

    try:
        response = _run_agent(req) if use_agent else _run_qa(req.question)
    except RuntimeError as exc:
        # 检索失败、模型超时、API key 失效都会走到这里。
        # 归为 503：调用方重试有意义，运维该去看依赖服务而不是查代码。
        logger.error("问答失败: %s", exc)
        raise HTTPException(status_code=503, detail=f"模型服务不可用：{exc}") from exc

    logger.info(
        "问答完成 mode=%s refused=%s citations=%s",
        response.mode,
        response.refused,
        len(response.citations),
    )
    return response
