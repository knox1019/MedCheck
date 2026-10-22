"""文本向量化。

对外只暴露 embed() / embed_one() 两个方法，上层不需要知道背后是
本地 Ollama 还是云端 API——换后端只改 .env 里的 base_url，代码不动。

Ollama 提供的是 OpenAI 兼容接口 /v1/embeddings，所以这里的调用写法
和调云端 API 完全一致。

三个必须做的事：
    1. 分批     —— 一次传太多会超时或被服务端拒绝
    2. 重试     —— 免费额度有速率限制，超了返回 429
    3. L2 归一化 —— 统一向量长度，让余弦相似度计算稳定
"""

import logging
import math
import time

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def normalize(vector: list[float]) -> list[float]:
    """把向量缩放到长度为 1（L2 归一化）。

    归一化之后，余弦相似度等价于点积，计算更快，也不受向量长度影响。

    Ollama 的 bge-m3 输出本身已经归一化过（实测模长 1.0000），
    这里再做一次是幂等的。保留它的理由是防御性：
    以后换成别的 provider，输出不一定归一化。
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class Embedder:
    """Embedding 客户端。"""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = 16,
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.embedding_base_url).rstrip("/")
        self.model = model or settings.embedding_model
        self.api_key = api_key or settings.embedding_api_key
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

    def _post_batch(self, texts: list[str]) -> list[list[float]]:
        """发一批文本，返回向量列表。失败时指数退避重试。"""
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": texts}
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
                    return [item["embedding"] for item in response.json()["data"]]

            # HTTPStatusError 覆盖 4xx/5xx（比如 429 限流）
            # TransportError 覆盖连不上、超时
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_error = exc
                # 指数退避：等 1s、2s、4s……避免立刻重试又被拒
                wait = 2**attempt
                logger.warning(
                    "embedding 请求失败（第 %s/%s 次）：%s，%s 秒后重试",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"embedding 连续失败 {self.max_retries} 次") from last_error

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。返回顺序与输入一致。"""
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            logger.info("向量化 %s-%s / %s", start + 1, start + len(batch), len(texts))
            vectors.extend(self._post_batch(batch))

        return [normalize(v) for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        """单条向量化，查询时用。"""
        return self.embed([text])[0]
