"""探针脚本：把 PDF 里能抽出来的东西原样打印，用来观察真实数据结构。

写解析器之前一定要先跑这个，看清楚两件事：
    1. PyMuPDF 抽正文得到什么（元信息在不在里面）
    2. pdfplumber 抽表格得到什么（几层嵌套、有多少空列、None 长什么样）

不要凭想象写字段映射。真实的表格结构和肉眼看到的往往不一样——
比如合并单元格会让每一列横跨多个物理格，值只填在第一格，其余是 None。

用法：
    uv run python scripts/probe_pdf.py
（这个脚本不依赖 app 包，所以可以直接按路径运行）
"""

import pdfplumber
import pymupdf

path = "data/samples/report_demo.pdf"

# ---- 第一段：正文 ----
doc = pymupdf.open(path)
print("=== 页数:", doc.page_count)
for page in doc:
    print("--- PyMuPDF 正文 ---")
    # 只打印前 800 字符，够看出页眉、患者信息、日期这些元信息的排版
    print(page.get_text()[:800])

# ---- 第二段：表格 ----
# 注意：extract_tables() 返回的是三层嵌套列表
#     最外层 = 这一页有几张表
#     中间层 = 每张表有几行
#     最内层 = 每行有几个单元格
with pdfplumber.open(path) as pdf:
    for page in pdf.pages:
        print("--- pdfplumber 表格 ---")
        for table in page.extract_tables():
            for row in table:
                print(row)
