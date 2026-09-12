import argparse
from pathlib import Path

from pypdf import PdfReader

parser = argparse.ArgumentParser()
parser.add_argument("pdf", type=Path)
args = parser.parse_args()

reader = PdfReader(args.pdf.expanduser())
print("总页数：", len(reader.pages))
for number, page in enumerate(reader.pages, 1):
    text = page.extract_text() or ""
    print(f"\nPDF第{number}页，提取字符数：{len(text)}")
    # 输出本页完整提取文本，不截取前250个字符。
    print(text)
