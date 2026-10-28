# split_corpus.py
import json
import re
from pathlib import Path


def split_merge(text: str, min_len: int = 60) -> list[str]:
    """把一段文字按中英文分号切开，太短的碎片合并回去。
    目标：输出2–3条小段落，每条约60字
    """
    # 用正则同时匹配 英文; 和 中文；
    raw_parts = [p.strip() for p in re.split(r"[；;]", text) if p.strip()]
    if not raw_parts:
        return []

    result = []
    current = raw_parts[0]

    for part in raw_parts[1:]:
        candidate = current + "；" + part
        if len(candidate) <= min_len * 1.3:
            current = candidate
        else:
            result.append(current)
            current = part
    result.append(current)

    # 兜底：只剩一条且过长，在标点附近切为两段
    if len(result) == 1 and len(result[0]) > min_len * 1.5:
        content = result[0]
        mid = len(content) // 2
        cut_pos = content.rfind("；", 0, mid)
        if cut_pos <= 0:
            cut_pos = content.rfind("。", 0, mid)
        if cut_pos <= 0:
            cut_pos = mid
        result = [content[:cut_pos+1].strip(), content[cut_pos+1:].strip()]

    return result


def main():
    in_path = Path("data/corpus/drugs.jsonl")
    out_path = Path("data/corpus/drugs_split.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src_count = 0
    dst_count = 0

    with in_path.open("r", encoding="utf-8") as f_in, \
         out_path.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            src_rec = json.loads(line)
            src_count += 1

            base_id = src_rec["id"]
            drug = src_rec["drug"]
            section = src_rec["section"]
            source = src_rec["source"]
            raw_text = src_rec["text"]

            chunks = split_merge(raw_text, min_len=60)
            # 给同一条原始记录的分片加后缀 -1 -2 -3
            for idx, chunk_text in enumerate(chunks, start=1):
                new_id = f"{base_id}-{idx}"
                out_rec = {
                    "id": new_id,
                    "drug": drug,
                    "section": section,
                    "text": chunk_text,
                    "source": source
                }
                f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                dst_count += 1

    print(f"{src_count} 条 → {dst_count} 条")


if __name__ == "__main__":
    main()
