"""合并语料：人工核对版 + 网页抓取版。

合并策略（重要）：
    原有 28 种药的 85 条语料是逐条核对过的，质量高于网页抓取，
    因此**同药同章节时优先保留原有条目**，抓取内容不覆盖。
    其余药全部来自抓取，并保留 source_url 便于回溯。

为什么不直接覆盖：
    两份内容都来自公开说明书，但原文经过人工整理后更精炼、
    术语一致。评测基线是在原有语料上建立的，替换会引入无关变量。

用法：
    uv run python -m scripts.merge_corpus
"""

import json
from pathlib import Path

MANUAL = Path("data/corpus/drugs.jsonl")
WEB = Path("data/corpus/drugs_web.jsonl")
OUTPUT = Path("data/corpus/drugs_all.jsonl")


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main() -> None:
    manual = load(MANUAL)
    web = load(WEB)

    merged: list[dict] = list(manual)
    # 已有人工条目的 (药, 章节) 组合，抓取版遇到就跳过
    covered = {(r["drug"], r["section"]) for r in manual}

    added = skipped = 0
    seen_ids = {r["id"] for r in merged}

    for record in web:
        key = (record["drug"], record["section"])
        if key in covered:
            skipped += 1
            continue
        if record["id"] in seen_ids:
            skipped += 1
            continue
        if len(record.get("text", "")) < 15:
            skipped += 1
            continue
        merged.append(record)
        seen_ids.add(record["id"])
        covered.add(key)
        added += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as file:
        for record in merged:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    drugs = {r["drug"] for r in merged}
    sections = {r["section"] for r in merged}
    print(f"人工语料：{len(manual)} 条 / {len({r['drug'] for r in manual})} 种药")
    print(f"抓取语料：{len(web)} 条 / {len({r['drug'] for r in web})} 种药")
    print(f"合并新增：{added} 条   跳过（已有或过短）：{skipped}")
    print()
    print(f"输出：{OUTPUT}")
    print(f"  总计 {len(merged)} 条 / {len(drugs)} 种药")
    print(f"  章节类型：{'、'.join(sorted(sections))}")


if __name__ == "__main__":
    main()
