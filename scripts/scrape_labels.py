"""从中国医药信息查询平台抓取药品说明书章节，扩充语料。

数据来源：中国医药信息查询平台（m.dayi.org.cn），国家药品监督管理局主管的公开平台。
每条记录都保存来源 URL，便于回溯核对。

抓取流程（每种药两次请求）：
    1. GET /search?keyword=<药名>   → 解析出第一个 /drug/<id>.html
    2. GET /drug/<id>.html          → 按锚点提取 禁忌 / 注意事项 / 药物相互作用

设计要点：
    - 串行 + 延迟：对公开站点保持礼貌，不并发压测
    - 增量落盘：每抓完一种就追加写入，中断后可续跑
    - 可恢复：已抓过的药名跳过，不重复请求

用法：
    uv run python -m scripts.scrape_labels              # 全量
    uv run python -m scripts.scrape_labels --limit 20   # 只抓前 20 种（验证质量）
"""

import argparse
import json
import re
import time
import urllib.parse
from pathlib import Path

import httpx

INDEX = Path("data/corpus/drug_name_index.json")
OUTPUT = Path("data/corpus/drugs_web.jsonl")

BASE = "https://www.dayi.org.cn"
SOURCE = "中国医药信息查询平台"

SECTIONS = [
    ("禁忌", "taboo"),
    ("注意事项", "notice"),
    ("药物相互作用", "interactions"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def clean_html(segment: str) -> str:
    """去掉标签与多余空白，得到纯文本。"""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", segment, flags=re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sections(html: str) -> dict[str, str]:
    """按锚点切出三个章节。

    每个章节的范围是「本锚点 → 下一个 id= 锚点」。
    这样不会把下一节的内容混进来，也不需要处理嵌套 div。
    """
    found: list[tuple[int, str]] = []
    for label, anchor in SECTIONS:
        match = re.search(rf'id="{anchor}"', html)
        if not match:
            continue
        # 从标签结束的 > 之后开始取，否则会把 class="..." 这些属性碎片当成正文
        tag_end = html.find(">", match.start())
        start = tag_end + 1 if tag_end != -1 else match.start()
        found.append((start, label))

    if not found:
        return {}

    found.sort()
    result: dict[str, str] = {}
    for index, (start, label) in enumerate(found):
        if index + 1 < len(found):
            end = found[index + 1][0]
        else:
            # 最后一节：找到下一个任意 id 锚点作为结束
            nxt = re.search(r'id="', html[start + 20 :])
            end = start + 20 + nxt.start() if nxt else min(len(html), start + 5000)
        text = clean_html(html[start:end])
        # 去掉章节标题本身
        for title in ("禁忌", "注意事项", "药物相互作用"):
            if text.startswith(title):
                text = text[len(title) :].strip()
        if text:
            result[label] = text
    return result


def find_drug_url(client: httpx.Client, name: str) -> str | None:
    """搜索药名，返回第一个药品详情页的完整 URL。"""
    url = f"{BASE}/search?keyword={urllib.parse.quote(name)}"
    response = client.get(url)
    response.raise_for_status()
    match = re.search(r'href="(/drug/\d+\.html)"', response.text)
    return f"{BASE}{match.group(1)}" if match else None


def load_done() -> set[str]:
    """读取已抓取的药名，用于续跑。"""
    if not OUTPUT.exists():
        return set()
    done = set()
    with OUTPUT.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["drug"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取药品说明书")
    parser.add_argument("--limit", type=int, default=0, help="只抓前 N 种，0 表示全部")
    parser.add_argument("--delay", type=float, default=0.4, help="每次请求间隔秒数")
    args = parser.parse_args()

    payload = json.loads(INDEX.read_text(encoding="utf-8"))
    names = [item["name"] for item in payload["drugs"]]
    if args.limit:
        names = names[: args.limit]

    done = load_done()
    todo = [n for n in names if n not in done]
    print(f"索引 {len(names)} 种，已完成 {len(done)} 种，本次待抓 {len(todo)} 种")

    ok = skipped = failed = 0
    records = 0

    with httpx.Client(headers=HEADERS, timeout=30.0, follow_redirects=True) as client:
        out = OUTPUT.open("a", encoding="utf-8")
        for number, name in enumerate(todo, start=1):
            try:
                drug_url = find_drug_url(client, name)
                time.sleep(args.delay)
                if not drug_url:
                    skipped += 1
                    print(f"  [{number}/{len(todo)}] {name}：搜不到详情页")
                    continue

                detail = client.get(drug_url)
                detail.raise_for_status()
                time.sleep(args.delay)

                sections = extract_sections(detail.text)
                if not sections:
                    skipped += 1
                    print(f"  [{number}/{len(todo)}] {name}：页面无目标章节")
                    continue

                for section, text in sections.items():
                    record = {
                        "id": f"web-{name}-{section}",
                        "drug": name,
                        "section": section,
                        "text": text[:600],  # 超长截断，避免单条过大
                        "source": f"{SOURCE}（{name}）",
                        "source_url": drug_url,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records += 1
                out.flush()
                ok += 1

                if number % 20 == 0:
                    print(f"  [{number}/{len(todo)}] 已完成 {ok} 种，{records} 条")

            except Exception as exc:  # noqa: BLE001 - 抓取失败不应中断整体流程
                failed += 1
                print(f"  [{number}/{len(todo)}] {name}：失败 {exc}")
        out.close()

    print()
    print(f"完成：{ok} 种 → {records} 条  跳过 {skipped}  失败 {failed}")
    print(f"输出：{OUTPUT}")


if __name__ == "__main__":
    main()
