"""大模型客户端。

和 embedder 一样走 OpenAI 兼容协议，所以：
    本地 Ollama   → http://localhost:11434/v1
    DeepSeek      → https://api.deepseek.com/v1
    硅基流动      → https://api.siliconflow.cn/v1

换后端只改 .env 里的三个值，业务代码一行不动。

三个工程细节：
    1. 指数退避重试 —— 免费额度有速率限制，超了返回 429
    2. 温度固定 0    —— 医疗问答必须可复现，同样的资料要得到同样的答案
    3. 关掉流式     —— 本阶段不需要打字机效果，直接拿完整响应更好解析
"""

import logging
import time

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


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
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.llm_api_key
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.timeout = settings.llm_timeout if timeout is None else timeout
        self.max_retries = max_retries

    def chat(self, messages: list[dict]) -> str:
        """发一轮对话，返回助手回复的纯文本。"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()

            except (httpx.HTTPStatusError, httpx.TransportError, KeyError) as exc:
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

        raise RuntimeError(f"大模型连续失败 {self.max_retries} 次") from last_error
