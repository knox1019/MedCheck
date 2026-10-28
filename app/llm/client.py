"""大模型客户端。

和 embedder 一样走 OpenAI 兼容协议，所以：
    本地 Ollama   → http://localhost:11434/v1
    DeepSeek      → https://api.deepseek.com  （注意：不带 /v1）
    硅基流动      → https://api.siliconflow.cn/v1

换后端只改 .env 里的几个值，业务代码一行不动。

三个工程细节：
    1. 指数退避重试 —— 免费额度有速率限制，超了返回 429
    2. 温度固定 0    —— 医疗问答必须可复现，同样的资料要得到同样的答案
    3. 关掉流式     —— 本阶段不需要打字机效果，直接拿完整响应更好解析

—— 关于「思考过程」（推理模型）——

推理模型会在给出答案前先输出一段思维链，落在响应的
message.reasoning_content 字段里，与 message.content 同级。
两者要分开对待：

    展示给用户看的      content
    调试 / 观察用的     reasoning_content
    用于评测和判定的    content（引用标记、拒答短语都在这里）

所以 chat() 仍然只返回 content，需要看思考过程时用 complete()。

⚠️ 多轮对话的回传规则（实测文档确认，两个方向都踩过坑）：

    请求不带 tools 时   reasoning_content 不需要回传；传了也会被服务端忽略
    请求带 tools 时     必须把历史轮次的 reasoning_content 完整回传，
                        否则 API 直接返回 400

本项目当前所有调用都是单轮（system + user 一次性发出），不涉及历史拼接。
等 W6 做 Agent 工具调用时，必须改用 ChatReply.as_assistant_message()
把 reasoning_content 一起带上——这个方法已经为此准备好。
"""

import logging
import time
from dataclasses import dataclass, field

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatReply:
    """一次完整的模型回复。

    content 和 reasoning 分开放，不合并成一个字符串——
    合并之后就没法单独统计思考消耗、也没法在展示时折叠。
    """

    content: str  # 最终答案
    reasoning: str = ""  # 思维链，非推理模型为空
    usage: dict = field(default_factory=dict)  # token 用量，部分后端不返回
    tool_calls: tuple[dict, ...] = ()  # 模型要求调用的工具，没有则为空
    model: str = ""  # 实际回答的模型。带降级时能看出「这次是谁答的」

    @property
    def has_reasoning(self) -> bool:
        return bool(self.reasoning.strip())

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def reasoning_tokens(self) -> int:
        """思考消耗的 token 数。部分后端不返回明细，此时为 0。"""
        details = self.usage.get("completion_tokens_details") or {}
        return int(details.get("reasoning_tokens") or 0)

    def as_assistant_message(self) -> dict:
        """转成可回传给下一轮请求的 assistant 消息。

        带 tools 的多轮循环里必须用它，而不是自己拼 {"role": "assistant", "content": ...}：
            · 少了 reasoning_content，思考模式 + tools 会直接返回 400
            · 少了 tool_calls，下一轮的 tool 消息就对不上调用方
        """
        message: dict = {"role": "assistant", "content": self.content}
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        if self.tool_calls:
            message["tool_calls"] = list(self.tool_calls)
        return message


class ChatClient:
    """OpenAI 兼容的对话客户端。"""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        reasoning_effort: str | None = None,
        thinking: str | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.llm_api_key
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.timeout = settings.llm_timeout if timeout is None else timeout
        self.max_retries = max_retries
        self.reasoning_effort = (
            settings.llm_reasoning_effort if reasoning_effort is None else reasoning_effort
        )
        self.thinking = settings.llm_thinking if thinking is None else thinking

    @property
    def thinking_active(self) -> bool:
        """这次请求是否会走思考模式。

        用途见 _build_payload：思考模式下 temperature 不生效，
        与其传一个静默无效的参数，不如不传。
        """
        return bool(self.reasoning_effort) or self.thinking == "enabled"

    def _build_payload(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        # 思考模式下 temperature / presence_penalty / frequency_penalty 都不生效
        # （服务端不报错，但也不采纳）。传了只会让日志看起来像是生效了，不如省略。
        if not self.thinking_active and self.temperature is not None:
            payload["temperature"] = self.temperature

        # 这两个字段只在显式配置时才发。本地 Ollama 不认识它们，
        # 留空可以保证切换后端时不出现「配过一次云端、切回本地就报错」。
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.thinking:
            payload["thinking"] = {"type": self.thinking}
        if tools:
            payload["tools"] = tools

        return payload

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> ChatReply:
        """发一轮对话，返回答案、思维链和工具调用请求。"""
        url = f"{self.base_url}/chat/completions"
        payload = self._build_payload(messages, tools=tools)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 只对网络和限流类错误重试。解析类问题（空答案、格式异常）重试没用，
        # 同一份请求再发一次只会得到同样的结果，白等三轮。
        last_error: Exception | None = None
        data: dict | None = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                break
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_error = exc
                wait = 2**attempt
                logger.warning(
                    "大模型请求失败（第 %s/%s 次）：%s，%s 秒后重试",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        else:
            raise RuntimeError(f"大模型连续失败 {self.max_retries} 次") from last_error

        try:
            choice = data["choices"][0]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"大模型响应格式异常：{str(data)[:300]}") from exc

        message = choice.get("message") or {}
        # 推理模型的思维链在 reasoning_content，普通模型的 content 则直接是答案
        reasoning = (message.get("reasoning_content") or "").strip()
        content = (message.get("content") or "").strip()
        tool_calls = tuple(message.get("tool_calls") or ())

        # 模型决定先调工具时，content 本来就是空的——这是正常流程，不是故障。
        # 只有「既没有答案、也没要求调工具」才是真异常。
        #
        # 这种情况通常是 max_tokens 给小了：思考把小模型「想完了」，输出被截断。
        # 必须报错而不是静默返回空串——上层会把空串当成拒答，
        # 于是「配置错了」被伪装成「资料不足」，排查方向就全歪了。
        if not content and not tool_calls:
            raise RuntimeError(
                f"模型返回空答案（finish_reason={choice.get('finish_reason')}），"
                f"思考内容 {len(reasoning)} 字，通常是 max_tokens 给得太小"
            )

        if reasoning:
            logger.info(
                "思考过程 %s 字，答案 %s 字，工具调用 %s 个",
                len(reasoning),
                len(content),
                len(tool_calls),
            )

        return ChatReply(
            content=content,
            reasoning=reasoning,
            usage=data.get("usage") or {},
            tool_calls=tool_calls,
            model=self.model,
        )

    def chat(self, messages: list[dict]) -> str:
        """只要答案，丢掉思维链。

        保留这个方法是因为问答和报告解读都只需要 content，
        而且评测脚本（引用标注率、拒答判断）判定的也是 content。
        """
        return self.complete(messages).content


class FallbackChatClient:
    """主模型不可用时自动降级到备用模型的客户端外壳。

    对调用方完全透明：complete() / chat() 的签名和返回值与 ChatClient 一致，
    上层（问答 / 报告核对 / Agent 循环）不需要知道自己拿到的是哪个。

    触发降级的错误（按「换一个模型能不能解决」来划界）：

        会切换   网络不通 / 超时 / 服务端 5xx —— complete() 内部重试耗尽后
                 统一包成「大模型连续失败 N 次」抛出；以及响应格式根本不像
                 OpenAI 协议（主模型服务本身坏了）
        不切换   空答案（max_tokens 配置问题）、鉴权 401 —— 换模型解决不了，
                 报出来让人修配置才是对的

    代价要如实说：主模型挂掉时，complete() 内部会先重试 3 次（约 7 秒）
    才轮到备用模型。这个延迟换来的是 429 限流时先原地重试、
    不轻易降到更弱的本地模型——对医疗问答这是更保守、更对的选择。
    """

    def __init__(self, primary: ChatClient, fallback: ChatClient) -> None:
        self.primary = primary
        self.fallback = fallback
        # 最近一次实际完成回答的模型。诊断「答案质量为什么变了」时先看它。
        self.last_model_used: str = primary.model

    @property
    def model(self) -> str:
        """主模型名。调用方一般只用来打日志，够用。"""
        return self.primary.model

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> ChatReply:
        try:
            reply = self.primary.complete(messages, tools=tools)
            self.last_model_used = self.primary.model
            return reply
        except RuntimeError as exc:
            message = str(exc)
            if "连续失败" not in message and "响应格式异常" not in message:
                raise
            logger.warning(
                "主模型 %s 不可用（%s），自动切换备用模型 %s",
                self.primary.model,
                message,
                self.fallback.model,
            )
        reply = self.fallback.complete(messages, tools=tools)
        self.last_model_used = self.fallback.model
        return reply

    def chat(self, messages: list[dict]) -> str:
        return self.complete(messages).content


def build_chat_client(model: str | None = None) -> ChatClient | FallbackChatClient:
    """按配置构造对话客户端；配了备用模型时返回带降级能力的外壳。

    这是业务代码拿客户端的统一入口（问答 / 报告核对 / Agent 的默认路径）。
    需要精确指向某个模型的场景（评测脚本、探活）仍直接构造 ChatClient——
    评测必须知道「这次测的是谁」，静默降级会让指标失真。

    主备配置相同时不包外壳：主模型挂了再去请求同一个地址没有意义。
    """
    settings = get_settings()
    primary = ChatClient(model=model)

    fallback_url = settings.llm_fallback_base_url
    if not fallback_url:
        return primary
    if (
        fallback_url.rstrip("/") == primary.base_url
        and settings.llm_fallback_model == primary.model
    ):
        return primary

    fallback = ChatClient(
        base_url=fallback_url,
        model=settings.llm_fallback_model,
        api_key=settings.llm_fallback_api_key,
    )
    return FallbackChatClient(primary, fallback)
