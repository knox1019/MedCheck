"""HTTP 接口测试。

不调用真实模型：问答部分用 monkeypatch 替掉 answer() 和 run_agent()，
测的是「路由选择正确、状态码正确、响应结构正确」。

这样做的理由：接口层和模型层是两件事。模型答得好不好由答案评测管，
接口该管的是「带了报告会不会走 Agent」「模型挂了返不返 503」。
混在一起测，会让接口测试因为模型波动而红，反而掩盖真实问题。
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agent.loop import AgentResult, AgentStep
from app.api import routes
from app.llm.qa import Answer
from app.main import app
from app.rag.citation import Citation

client = TestClient(app)

SAMPLE_PDF = Path("data/samples/report_demo.pdf")


# ------------------------------------------------------------------ 解析接口

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="样本 PDF 不存在")
def test_parse_returns_structured_report():
    response = client.post(
        "/api/report/parse",
        files={"file": ("report_demo.pdf", SAMPLE_PDF.read_bytes(), "application/pdf")},
    )

    assert response.status_code == 200
    report = response.json()["report"]
    assert len(report["items"]) == 21
    abnormal = [i for i in report["items"] if i["flag"] in ("high", "low")]
    assert len(abnormal) == 11


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="样本 PDF 不存在")
def test_parse_uses_filename_when_no_barcode():
    """取不到条码号时用上传文件名兜底，而不是临时随机名。

    临时文件名形如 tmpXXXXXX，对调用方没有意义；
    而且同一个文件每次上传会得到不同的 report_id，无法去重。
    """
    response = client.post(
        "/api/report/parse",
        files={"file": ("我的体检报告.pdf", SAMPLE_PDF.read_bytes(), "application/pdf")},
    )

    assert response.status_code == 200
    # 样本 PDF 有条码号，所以这里确认的是「没有用临时文件名」
    assert "tmp" not in response.json()["report"]["report_id"]


def test_parse_rejects_non_pdf():
    response = client.post(
        "/api/report/parse",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_parse_rejects_empty_file():
    response = client.post(
        "/api/report/parse",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert "空" in response.json()["detail"]


def test_parse_returns_422_for_broken_pdf():
    """坏文件是 422（内容无法处理），不是 500（代码错误）。

    这两种状态码运维处理方式不同：422 是调用方的问题，500 是我们的 bug。
    """
    response = client.post(
        "/api/report/parse",
        files={"file": ("broken.pdf", b"%PDF-1.4 garbage", "application/pdf")},
    )

    assert response.status_code == 422


# ------------------------------------------------------------------ 问答接口

def _fake_answer(question: str) -> Answer:
    return Answer(
        question=question,
        text="答案正文 [1]",
        citations=[Citation(source="华法林钠片说明书", section="禁忌", chunk_id="c1")],
        refused=False,
        reason="命中「华法林」的说明书章节",
        cited=True,
    )


def _fake_agent(question, ctx, **kwargs) -> AgentResult:
    return AgentResult(
        question=question,
        answer="Agent 答案 [1]",
        citations=[Citation(source="阿托伐他汀钙片说明书", section="注意事项", chunk_id="c2")],
        cited=True,
        turns=2,
        steps=[
            AgentStep(tool="search_drug_label", args={"drug": "华法林"}, output="x" * 50),
        ],
    )


def test_chat_without_context_uses_qa(monkeypatch):
    """纯药物问题不需要 Agent——多一次模型决策只会更慢，不会更准。"""
    monkeypatch.setattr(routes, "answer", _fake_answer)

    response = client.post("/api/chat", json={"question": "华法林的禁忌是什么"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "qa"
    assert body["answer"] == "答案正文 [1]"
    assert body["citations"][0]["label"] == "华法林钠片说明书 · 禁忌"


def test_chat_with_medications_uses_agent(monkeypatch):
    monkeypatch.setattr(routes, "run_agent", _fake_agent)

    response = client.post(
        "/api/chat",
        json={"question": "要紧吗", "medications": ["华法林"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "agent"
    assert body["turns"] == 2
    assert body["steps"][0]["tool"] == "search_drug_label"
    assert body["steps"][0]["chars"] == 50


def test_chat_mode_qa_overrides_agent(monkeypatch):
    """显式指定 qa 时，即使带了用药也不走 Agent。

    场景：调用方只想快速拿一个答案，不想等模型多轮决策。
    """
    monkeypatch.setattr(routes, "answer", _fake_answer)

    response = client.post(
        "/api/chat",
        json={"question": "华法林的禁忌", "medications": ["华法林"], "mode": "qa"},
    )

    assert response.json()["mode"] == "qa"


def test_chat_mode_agent_forces_agent(monkeypatch):
    """显式指定 agent 时，没带上下文也走 Agent。

    Agent 可以只用工具回答（比如先 list_available_drugs 看看有什么）。
    """
    monkeypatch.setattr(routes, "run_agent", _fake_agent)

    response = client.post("/api/chat", json={"question": "有什么药", "mode": "agent"})

    assert response.json()["mode"] == "agent"


def test_chat_returns_503_when_model_unavailable(monkeypatch):
    """模型服务挂了返回 503，不是 500。

    503 告诉运维「去查依赖服务」，500 会让人以为是本服务代码有问题。
    """

    def boom(question):
        raise RuntimeError("大模型连续失败 3 次")

    monkeypatch.setattr(routes, "answer", boom)

    response = client.post("/api/chat", json={"question": "华法林的禁忌"})

    assert response.status_code == 503
    assert "模型服务不可用" in response.json()["detail"]


def test_chat_rejects_empty_question():
    """空问题在入口就被挡住，不会走到模型。"""
    response = client.post("/api/chat", json={"question": ""})

    assert response.status_code == 422  # Pydantic 校验失败


def test_chat_accepts_report_payload(monkeypatch):
    """报告以 JSON 形式随请求传入，服务端不存会话。"""
    monkeypatch.setattr(routes, "run_agent", _fake_agent)

    report = {
        "report_id": "t-001",
        "exam_date": "2026-10-22",
        "items": [
            {
                "name_raw": "丙氨酸氨基转移酶 ALT",
                "value": 68,
                "unit": "U/L",
                "ref_low": 9,
                "ref_high": 50,
                "flag": "high",
                "page": 1,
            }
        ],
    }
    response = client.post(
        "/api/chat",
        json={"question": "要紧吗", "report": report},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "agent"


def test_health_still_works():
    """加了路由之后健康检查不能坏——容器编排靠它判断存活。"""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
