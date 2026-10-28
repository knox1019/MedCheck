"""大模型客户端的单元测试。

不真的发网络请求：用假的 httpx.Client 顶替，测的是
    1. 请求体构造（思考模式下哪些参数不该发）
    2. 响应解析（reasoning_content 与 content 分开）
    3. 异常响应（空答案不能静默变成空字符串）

为什么这些值得测：
    参数发错了，服务端通常不报错、只是静默不生效（temperature 就是这样），
    只看「调用成功」永远发现不了。
"""

import pytest

from app.llm import client as client_module
from app.llm.client import ChatClient, ChatReply


class _FakeResponse:
    def __init__(self, data: dict) -> None:
        self._data = data
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._data


class _FakeClient:
    """替换 httpx.Client，记录发出去的请求体。"""

    last_payload: dict = {}
    response_data: dict = {}

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, json=None, headers=None):
        _FakeClient.last_payload = json or {}
        return _FakeResponse(_FakeClient.response_data)


@pytest.fixture
def fake_http(monkeypatch):
    """把 httpx.Client 换成假实现，并给一个默认的模型响应。"""

    def _install(data: dict) -> None:
        _FakeClient.response_data = data
        _FakeClient.last_payload = {}
        monkeypatch.setattr(client_module.httpx, "Client", _FakeClient)

    _install(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "答案正文"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
    )
    return _install


# ------------------------------------------------------------------ 请求体构造

def test_payload_omits_thinking_params_by_default(fake_http):
    """没配置思考模式时，这两个字段根本不该出现在请求里。

    本地 Ollama 不认识它们。一旦无条件发送，就会出现
    「配过一次云端、切回本地就报错」。
    """
    client = ChatClient(reasoning_effort="", thinking="")
    client.complete([{"role": "user", "content": "hi"}])

    payload = _FakeClient.last_payload
    assert "reasoning_effort" not in payload
    assert "thinking" not in payload
    assert payload["temperature"] == client.temperature


def test_payload_includes_thinking_when_configured(fake_http):
    client = ChatClient(reasoning_effort="high", thinking="enabled")
    client.complete([{"role": "user", "content": "hi"}])

    payload = _FakeClient.last_payload
    assert payload["reasoning_effort"] == "high"
    assert payload["thinking"] == {"type": "enabled"}


def test_payload_drops_temperature_in_thinking_mode(fake_http):
    """思考模式下 temperature 不生效。

    服务端不报错、也不采纳。与其发一个静默无效的参数，
    不如不发——否则日志会让人误以为温度真的起作用了。
    """
    client = ChatClient(reasoning_effort="high")
    client.complete([{"role": "user", "content": "hi"}])

    assert "temperature" not in _FakeClient.last_payload


def test_thinking_active_property():
    assert ChatClient(reasoning_effort="high", thinking="").thinking_active is True
    assert ChatClient(reasoning_effort="", thinking="enabled").thinking_active is True
    assert ChatClient(reasoning_effort="", thinking="disabled").thinking_active is False
    assert ChatClient(reasoning_effort="", thinking="").thinking_active is False


# ------------------------------------------------------------------ 响应解析

def test_parses_reasoning_and_content_separately(fake_http):
    """思维链和答案必须分开，不能拼在一起。

    拼在一起之后，既没法单独统计思考消耗，
    也会让上层的引用标记解析把思考里的方括号误当成引用编号。
    """
    fake_http(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "先看资料 1，再看资料 2，结论是……",
                        "content": "答案正文 [1]",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 200,
                "total_tokens": 210,
                "completion_tokens_details": {"reasoning_tokens": 150},
            },
        }
    )

    reply = ChatClient().complete([{"role": "user", "content": "hi"}])

    assert reply.content == "答案正文 [1]"
    assert reply.reasoning.startswith("先看资料")
    assert reply.has_reasoning is True
    assert reply.reasoning_tokens == 150


def test_missing_reasoning_is_not_an_error(fake_http):
    """非推理模型没有 reasoning_content，这不算异常。"""
    reply = ChatClient().complete([{"role": "user", "content": "hi"}])

    assert reply.content == "答案正文"
    assert reply.reasoning == ""
    assert reply.has_reasoning is False
    assert reply.reasoning_tokens == 0


def test_empty_content_raises_without_retrying(fake_http, monkeypatch):
    """空答案要报错，而且不能重试三次。

    两个理由：
      · 静默返回空串，上层会当成「拒答」，于是配置错误被伪装成资料不足
      · 解析类问题重试没有意义，同一份请求再发一次结果一样
    """
    attempts = {"n": 0}

    class _Counting(_FakeClient):
        def post(self, url, json=None, headers=None):
            attempts["n"] += 1
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "", "reasoning_content": "想了很久"},
                            "finish_reason": "length",
                        }
                    ]
                }
            )

    monkeypatch.setattr(client_module.httpx, "Client", _Counting)

    with pytest.raises(RuntimeError, match="空答案"):
        ChatClient().complete([{"role": "user", "content": "hi"}])

    assert attempts["n"] == 1, "解析类错误不该重试"


# ------------------------------------------------------------------ 多轮回传

def test_as_assistant_message_includes_reasoning():
    """带 tools 的多轮循环必须回传 reasoning_content，否则 API 返回 400。

    本项目现在都是单轮调用，用不到这个方法；
    但 W6 做工具调用时如果忘了它，报错信息会非常难懂，所以先测下来。
    """
    reply = ChatReply(content="答案", reasoning="思考过程")
    message = reply.as_assistant_message()

    assert message["role"] == "assistant"
    assert message["content"] == "答案"
    assert message["reasoning_content"] == "思考过程"


def test_as_assistant_message_omits_empty_reasoning():
    """没有思考内容时不要塞一个空的 reasoning_content 键。"""
    message = ChatReply(content="答案").as_assistant_message()

    assert "reasoning_content" not in message


def test_chat_returns_content_only(fake_http):
    """chat() 只返回答案，保持旧调用方不受影响。"""
    fake_http(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "reasoning_content": "思考", "content": "答案"},
                    "finish_reason": "stop",
                }
            ]
        }
    )

    assert ChatClient().chat([{"role": "user", "content": "hi"}]) == "答案"
