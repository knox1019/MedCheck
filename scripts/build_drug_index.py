"""生成药品名称索引：官方目录 → 可识别药名清单。

数据来源：国家基本药物目录（2018 年版），国家卫生健康委员会公开文件。
    https://www.nhc.gov.cn/ewebeditor/uploadfile/2018/10/20181025183346942.pdf

为什么需要这个名字索引：

    语料只有几十种药，但用户可能问任何一种药。
    如果系统「不认识」这个药名，它不会拒答，而会拿语义上最接近的
    别的药的内容去凑答案——这是本项目最危险的一类错误。

    有了这份索引，「知道这个药但库里没有资料」就能被准确识别，
    从而给出明确拒答，而不是张冠李戴。

重要区分：
    索引里的药名  = 系统「认识」的药（663 种）
    语料里有资料的 = 系统「能回答」的药（当前 28 种）
    两者相差的部分，一律拒答。

用法：
    uv run python -m scripts.build_drug_index
"""

import json
import re
from pathlib import Path

CHEM_RAW = Path("data/raw/neml_chemical_names.json")
TCM_RAW = Path("data/raw/neml_tcm_names.json")
OUTPUT = Path("data/corpus/drug_name_index.json")

SOURCE = "国家基本药物目录（2018年版）"
SOURCE_URL = "https://www.nhc.gov.cn/ewebeditor/uploadfile/2018/10/20181025183346942.pdf"


def main_name(raw: str) -> str:
    """去掉剂型括号，取主名。

    「九味羌活丸（颗粒）」→「九味羌活丸」
    括号里是可选剂型，不是药名的一部分。
    中成药条目大量使用这种写法，不清理会让匹配失效。
    """
    return re.sub(r"[（(][^）)]*[）)]", "", raw).strip()


def main() -> None:
    chemical = json.loads(CHEM_RAW.read_text(encoding="utf-8"))
    tcm = json.loads(TCM_RAW.read_text(encoding="utf-8"))

    entries: list[dict] = []
    seen: set[str] = set()

    for raw in chemical + tcm:
        name = main_name(raw)
        if not name or name in seen:
            continue
        seen.add(name)
        aliases = [name]
        if raw != name:
            aliases.append(raw)  # 带剂型的原始写法也保留为别名
        entries.append(
            {
                "name": name,
                "aliases": aliases,
                "category": "化学药品和生物制品" if raw in chemical else "中成药",
            }
        )

    payload = {
        "source": SOURCE,
        "source_url": SOURCE_URL,
        "count": len(entries),
        "note": "本索引仅表示系统『认识』这些药名，不代表语料里有它们的说明书内容。",
        "drugs": entries,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    chem_count = sum(1 for e in entries if e["category"].startswith("化学"))
    tcm_count = len(entries) - chem_count
    print(f"已生成：{OUTPUT}")
    print(f"  化学药品和生物制品：{chem_count}")
    print(f"  中成药：{tcm_count}")
    print(f"  合计：{len(entries)}")


if __name__ == "__main__":
    main()
