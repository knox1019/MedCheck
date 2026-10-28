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


# ------------------------------------------------------------------ 主备降级

class _StubPrimary:
    """可编程的主客户端替身：按脚本依次抛错或返回。"""

    def __init__(self, model: str, script: list) -> None:
        self.model = model
        self.script = list(script)
        self.calls = 0

    def complete(self, messages, tools=None) -> ChatReply:
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def test_fallback_switches_when_primary_exhausts_retries():
    """主模型「重试耗尽」（网络不通 / 5xx 的统一出口）→ 切备用模型。

    备用模型用 stub 而不是真实 ChatClient：这些是单元测试，
    一旦真发 HTTP，CI（没有 Ollama）和干净机器上就会随机失败。
    """
    wrapper = client_module.FallbackChatClient(
        _StubPrimary("cn:hy3", [RuntimeError("大模型连续失败 3 次")]),
        _StubPrimary("qwen2.5:7b", [ChatReply(content="备用模型的回答", model="qwen2.5:7b")]),
    )

    reply = wrapper.complete([{"role": "user", "content": "hi"}])

    assert reply.content                      # 备用模型确实返回了内容
    assert reply.model == "qwen2.5:7b"
    assert wrapper.last_model_used == "qwen2.5:7b"


def test_fallback_switches_on_malformed_primary_response():
    """主模型返回的不是 OpenAI 协议格式（网关坏了）→ 也切。"""
    wrapper = client_module.FallbackChatClient(
        _StubPrimary("cn:hy3", [RuntimeError("大模型响应格式异常：<html>…")]),
        _StubPrimary("qwen2.5:7b", [ChatReply(content="备用模型的回答", model="qwen2.5:7b")]),
    )

    reply = wrapper.complete([{"role": "user", "content": "hi"}])

    assert reply.model == "qwen2.5:7b"


def test_fallback_does_not_trigger_on_empty_answer():
    """空答案是 max_tokens 配置问题，换模型解决不了 → 原样抛出。

    如果这里也降级，「配置错了」就会被伪装成「备用模型的回答」，
    排查方向全歪——这正是当初把空答案设计成报错的原因。
    """
    primary = _StubPrimary("cn:hy3", [RuntimeError("模型返回空答案（finish_reason=length）")])
    fallback_calls = []

    class _RecordingFallback(_StubPrimary):
        def complete(self, messages, tools=None):
            fallback_calls.append(1)
            return super().complete(messages, tools=tools)

    wrapper = client_module.FallbackChatClient(
        primary,
        _RecordingFallback("qwen2.5:7b", [ChatReply(content="不该被用到")]),
    )

    with pytest.raises(RuntimeError, match="空答案"):
        wrapper.complete([{"role": "user", "content": "hi"}])
    assert fallback_calls == []


def test_fallback_not_used_when_primary_healthy():
    """主模型正常时绝不碰备用——降级是异常路径，不是负载均衡。"""
    good = ChatReply(content="主模型的回答", model="cn:hy3")
    wrapper = client_module.FallbackChatClient(
        _StubPrimary("cn:hy3", [good]),
        _StubPrimary("qwen2.5:7b", [RuntimeError("备用模型不该被调用")]),
    )

    reply = wrapper.complete([{"role": "user", "content": "hi"}])

    assert reply.model == "cn:hy3"
    assert wrapper.last_model_used == "cn:hy3"


def test_build_chat_client_returns_wrapper_when_fallback_configured(monkeypatch):
    """主备配置不同 → 返回降级外壳。"""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:7863/v1")
    monkeypatch.setenv("LLM_MODEL", "cn:hy3")
    monkeypatch.setenv("LLM_FALLBACK_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "qwen2.5:7b")
    client_module.get_settings.cache_clear()

    try:
        wrapper = client_module.build_chat_client()
        assert isinstance(wrapper, client_module.FallbackChatClient)
        assert wrapper.primary.model == "cn:hy3"
        assert wrapper.fallback.model == "qwen2.5:7b"
    finally:
        client_module.get_settings.cache_clear()


def test_build_chat_client_skips_wrapper_when_primary_is_fallback(monkeypatch):
    """主模型本身就是本地 Ollama 时不再包一层——挂了再请求同一个地址没有意义。"""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    monkeypatch.delenv("LLM_FALLBACK_BASE_URL", raising=False)
    client_module.get_settings.cache_clear()

    try:
        client = client_module.build_chat_client()
        assert isinstance(client, ChatClient)
    finally:
        client_module.get_settings.cache_clear()


def test_build_chat_client_disabled_by_empty_fallback_url(monkeypatch):
    """显式把备用 base_url 置空 = 关闭降级。"""
    monkeypatch.setenv("LLM_FALLBACK_BASE_URL", "")
    client_module.get_settings.cache_clear()

    try:
        client = client_module.build_chat_client()
        assert isinstance(client, ChatClient)
    finally:
        client_module.get_settings.cache_clear()
