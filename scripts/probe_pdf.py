import pymupdf   
import pdfplumber

path = "data/samples/report_demo.pdf"

doc = pymupdf.open(path)
print("=== 页数:", doc.page_count)
for page in doc:
    print("--- PyMuPDF 正文 ---")
    print(page.get_text()[:800])

with pdfplumber.open(path) as pdf:
    for page in pdf.pages:
        print("--- pdfplumber 表格 ---")
        for table in page.extract_tables():
            for row in table:
                print(row)
