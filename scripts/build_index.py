"""建库脚本：读语料 → 向量化 → 写入向量库。

用法：
    uv run python -m scripts.build_index
    uv run python -m scripts.build_index data/corpus/drugs.jsonl

可以重复执行，每次都会全量重建，结果一致。
"""

import json
import sys
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.embedder import Embedder
from app.rag.store import get_client, rebuild

DEFAULT_CORPUS = Path("data/corpus/drugs.jsonl")


def load_corpus(path: Path) -> list[dict]:
    """读 jsonl 语料。每行一个独立 JSON 对象。"""
    records: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行不是合法 JSON：{exc}") from exc
    return records


def to_embedding_text(record: dict) -> str:
    """把一条语料拼成用于向量化的文本。

    关键点：必须把药名和章节名一起拼进去。

    因为说明书正文里往往不会出现"华法林"三个字（都是"本品"），
    如果只拿 text 去向量化，查"华法林的禁忌"会匹配不上。
    这是 RAG 里很常见的一个坑：**检索用的文本要和用户的问法对得上**。
    """
    return f"{record['drug']}【{record['section']}】{record['text']}"


def main() -> None:
    setup_logging()
    settings = get_settings()

    corpus_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CORPUS
    if not corpus_path.exists():
        print(f"找不到语料文件：{corpus_path}")
        raise SystemExit(1)

    print(f"读取语料：{corpus_path}")
    records = load_corpus(corpus_path)
    print(f"共 {len(records)} 条")

    texts = [to_embedding_text(r) for r in records]

    print(f"向量化中（模型 {settings.embedding_model}）……")
    started = time.perf_counter()
    embedder = Embedder()
    vectors = embedder.embed(texts)
    elapsed = time.perf_counter() - started
    print(f"向量化完成，耗时 {elapsed:.1f}s，维度 {len(vectors[0])}")

    client = get_client(settings.chroma_dir)
    collection = rebuild(
        client,
        ids=[r["id"] for r in records],
        documents=texts,
        embeddings=vectors,
        metadatas=[
            {"drug": r["drug"], "section": r["section"], "source": r["source"]}
            for r in records
        ],
    )

    drugs = sorted({r["drug"] for r in records})
    sections = sorted({r["section"] for r in records})

    print()
    print(f"建库完成：{settings.chroma_dir}")
    print(f"  集合条目数：{collection.count()}")
    print(f"  药品数量：{len(drugs)}（{'、'.join(drugs)}）")
    print(f"  章节类型：{'、'.join(sections)}")


if __name__ == "__main__":
    main()
