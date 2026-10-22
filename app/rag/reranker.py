"""重排：用 cross-encoder 对候选集重新打分。

为什么要重排：

    向量检索（bi-encoder）把 query 和文档各自压成一个向量再算距离，
    快，但丢失了细粒度信息——它分不清「会不会导致」和「什么人不能用」的差别。

    cross-encoder 把 (query, 文档) 拼成一对一起送进模型，直接输出相关度分数。
    慢（每对都要过一次模型），但判断精细得多。所以标准做法是：
    **先用向量检索粗召回 top-k（快），再用 cross-encoder 精排（准）。**

本项目实测的排序问题正属于这一类：
    阿司匹林「会不会把胃吃坏」，top-1 是「禁忌」（已经溃疡的人不能用），
    正确答案「注意事项」（可诱发消化道溃疡出血）只差 0.0056 分屈居第二。

模型选择：
    BAAI/bge-reranker-v2-m3-int8 约 0.57GB，fp32 版 2.27GB。
    int8 量化对重排任务的影响很小，而体积只有四分之一，性价比更高。

依赖说明：
    用 fastembed（ONNX 后端），不需要装 PyTorch（约 2.5GB）。
"""

import logging
import os

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class Reranker:
    """cross-encoder 重排器。

    模型是懒加载的：第一次调用 rerank 时才初始化。
    因为加载 ONNX 模型要几秒，而很多流程（比如只做向量检索）用不到它。
    """

    def __init__(self, model_name: str | None = None, threads: int | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.rerank_model
        self.threads = threads
        self._encoder = None

        # 必须在导入 huggingface_hub 之前设置，否则镜像地址不生效。
        # 国内直连 huggingface.co 是不通的，不设这个环境变量会卡在下载上。
        os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)

    def _load(self):
        if self._encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            logger.info("加载重排模型：%s", self.model_name)
            self._encoder = TextCrossEncoder(
                model_name=self.model_name,
                threads=self.threads,
            )
        return self._encoder

    def rerank(self, query: str, hits: list[dict], top_k: int | None = None) -> list[dict]:
        """对候选片段重新打分并按新分数排序。

        参数：
            hits   检索结果，每条至少要有 text 字段
            top_k  只保留前几条；None 表示全部保留

        返回：
            新的列表，每条形如
                {"text":..., "metadata":..., "similarity": 向量相似度,
                 "rerank_score": 重排分数, "rank_before": 原排名}
            保留 similarity 是为了能对比「重排前后谁被拎上来了」。
        """
        if not hits:
            return []

        encoder = self._load()
        documents = [hit["text"] for hit in hits]

        # rerank 内部就是逐对 (query, document) 打分，返回与输入同序的分数
        scores = [float(score) for score in encoder.rerank(query, documents)]

        reranked = []
        for rank_before, (hit, score) in enumerate(zip(hits, scores), start=1):
            reranked.append(
                {
                    **hit,
                    "rerank_score": score,
                    "rank_before": rank_before,
                }
            )

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked
