"""向量库读写。

封装 Chroma，业务层不需要知道底层用的是什么库。

Chroma 内部做的事其实很朴素：把 (id, 向量, 原文, 元数据) 存在一起，
查询时对每条算一次距离再排序。没有魔法，所以也不用怕它。

一个概念要对齐：
    distance   = 距离，越小越像（余弦距离 = 1 - 余弦相似度）
    similarity = 相似度，越大越像
我们对外的接口统一用 similarity，因为它更符合直觉。
"""

from pathlib import Path

import chromadb

# 集合名。Chroma 要求 3-512 字符，只能含字母数字和 ._-
COLLECTION_NAME = "drug_corpus"


def get_client(persist_dir: str | Path) -> chromadb.ClientAPI:
    """打开（或创建）持久化的向量库。

    PersistentClient 会把数据写到磁盘目录，进程退出后仍然在。
    想清空重建时，直接删掉这个目录即可。
    """
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def rebuild(
    client: chromadb.ClientAPI,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """重建集合：先删后建，再写入。

    为什么先删：保证脚本可以重复执行，结果一致（幂等）。
    如果直接往旧集合里 add，重复跑会出现重复条目或 id 冲突。
    数据量小，全量重建比增量更新简单可靠得多。
    """
    try:
        client.delete_collection(name)
    except Exception:
        # 首次运行时集合还不存在，删不掉是正常的
        pass

    # hnsw:space=cosine 表示用余弦距离。
    # 因为我们的向量已经 L2 归一化，用内积和余弦的结果是一样的。
    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return collection


def get(client: chromadb.ClientAPI, name: str = COLLECTION_NAME) -> chromadb.Collection:
    """取已有集合。集合不存在会抛异常，这是有意的——没建库就该报错。"""
    return client.get_collection(name)


def search(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """向量检索，返回 top_k 条，按相似度从高到低。

    where 用于元数据过滤，例如 {"drug": "华法林"} 只在华法林的条目里检。
    在真实场景里这很关键：已知患者在吃某种药，就不该去检索别的药。
    """
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    # Chroma 返回的是「批量查询」的结构，每个字段都是二维列表：
    # result["ids"][0] 才是第一个 query 的结果
    hits: list[dict] = []
    for i in range(len(result["ids"][0])):
        distance = result["distances"][0][i]
        hits.append(
            {
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "distance": distance,
                # 余弦距离 = 1 - 余弦相似度，所以换算回来更直观
                "similarity": 1 - distance,
            }
        )
    return hits
