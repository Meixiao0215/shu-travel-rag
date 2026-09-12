"""学习顺序 1：提取表格、合并跨页条款、生成可追溯的小块。

这里只支持已核对的 2026 年 128 号文件。新增政策应先做质量检查，
不能直接假设新 PDF 也有相同的表格和附表位置。
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

ARTICLE = re.compile(r"^(第[一二三四五六七八九十百]+条)\s*(.*)$")
CHAPTER = re.compile(r"^第[一二三四五六七八九十]+章\s*(.*)$")
PAGE_NUMBER = re.compile(r"^[—－-]\s*\d+\s*[—－-]$")
PARSER_VERSION = "shu128-v1"


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def table_records(page, table, header_rows: int) -> str:
    """用单元格坐标还原合并关系，避免对所有空值盲目向下填充。"""
    rows = table.extract()
    for r, row in enumerate(table.rows):
        for c, box in enumerate(row.cells):
            if box is not None:
                continue
            col = table.columns[c].bbox
            x, y = (col[0] + col[2]) / 2, row.bbox[1] + 0.5
            covering = next((cell for cell in table.cells
                             if cell[0] <= x < cell[2] and cell[1] <= y < cell[3]), None)
            if covering:
                rows[r][c] = page.crop(covering).extract_text()
    clean = lambda x: re.sub(r"\s+", "", x or "")
    rows = [[clean(cell) for cell in row] for row in rows]
    headers = []
    for col in range(len(rows[0])):
        parts = list(dict.fromkeys(rows[r][col] for r in range(header_rows) if rows[r][col]))
        headers.append("/".join(parts))
    return "\n".join("；".join(f"{h}：{v}" for h, v in zip(headers, row))
                     for row in rows[header_rows:])


def extract_pages(pdf: Path) -> list[dict]:
    pages = []
    with pdfplumber.open(pdf) as doc:
        if len(doc.pages) != 12:
            raise ValueError("此解析配置仅适用于已核对的 12 页上大内〔2026〕128号文件。")
        for number, page in enumerate(doc.pages, 1):
            raw = page.extract_text() or ""
            text = raw
            if number in (3, 4):
                tables = page.find_tables()
                if len(tables) != 1:
                    raise ValueError(f"第 {number} 页预期有一张表，实际 {len(tables)} 张，请人工检查。")
                table = tables[0]
                top, bottom = table.bbox[1], table.bbox[3]
                before = page.crop((0, 0, page.width, top)).extract_text() or ""
                after = page.crop((0, bottom, page.width, page.height)).extract_text() or ""
                text = before + "\n" + table_records(page, table, 1 if number == 3 else 2) + "\n" + after
            lines = [line.strip() for line in text.splitlines() if not PAGE_NUMBER.match(line.strip())]
            if len("".join(lines)) < 30:
                raise ValueError(f"第 {number} 页文字过少，先检查是否需要 OCR。")
            pages.append({"page_number": number, "text": "\n".join(lines), "raw_text": raw})
    return pages


def build_articles(pages: list[dict], doc_id: str, source: str) -> list[dict]:
    records, current, chapter = [], None, "通知"

    def flush():
        nonlocal current
        if current:
            current["text"] = "\n".join(current.pop("lines"))
            records.append(current)
        current = None

    for page in pages:
        number = page["page_number"]
        if number == 1 or number >= 11:
            flush()
            tag = "notice" if number == 1 else f"form{number - 10}"
            records.append({"id": f"{doc_id}:{tag}", "doc_id": doc_id, "source": source,
                            "article": "发文通知" if number == 1 else f"附表{number - 10}",
                            "chapter": "通知" if number == 1 else "附表",
                            "pages": [number], "text": page["text"]})
            continue
        for line in page["text"].splitlines():
            match_chapter = CHAPTER.match(line)
            match_article = ARTICLE.match(line)
            if match_chapter:
                flush()
                chapter = match_chapter.group(1).replace(" ", "")
                continue
            if line.startswith("上海大学党政办公室") or line.startswith("校对："):
                flush()
                continue
            if match_article:
                flush()
                current = {"id": "pending",
                           "doc_id": doc_id, "source": source, "article": match_article.group(1),
                           "chapter": chapter, "pages": [], "lines": []}
            if current and line:
                current["lines"].append(line)
                if number not in current["pages"]:
                    current["pages"].append(number)
    flush()
    articles = [r for r in records if r["article"].startswith("第")]
    for i, record in enumerate(articles, 1):
        record["id"] = f"{doc_id}:a{i:03d}"
    if len(articles) != 33 or articles[-1]["article"] != "第三十三条":
        raise ValueError(f"预期 33 条正文，实际 {len(articles)} 条，请检查分段。")
    return records


def build_chunks(articles: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450, chunk_overlap=80,
        separators=["\n", "。", "；", "，", "",],
    )
    chunks = []
    for parent in articles:
        for i, text in enumerate(splitter.split_text(parent["text"])):
            chunks.append({"id": f"{parent['id']}:c{i:02d}", "parent_id": parent["id"],
                           "text": f"{parent['chapter']} {parent['article']}\n{text}"})
    return chunks


def ingest(pdf: Path, root: Path):
    pdf = pdf.expanduser().resolve()
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    profile = read_json(root / "data/policy_profile.json")
    if sha != profile["sha256"]:
        raise ValueError("PDF 指纹与已核对文件不同。请先审阅新文件并更新解析配置，不要混用政策。")
    pages = extract_pages(pdf)
    doc_id = sha[:12]
    articles = build_articles(pages, doc_id, pdf.name)
    chunks = build_chunks(articles)
    out = root / "data/processed"
    raw = root / "data/raw" / pdf.name
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.resolve() != pdf:
        shutil.copy2(pdf, raw)
    for name, value in [("pages", pages), ("articles", articles), ("chunks", chunks)]:
        write_json(out / f"{name}.json", value)
    manifest = {**profile, "doc_id": doc_id, "source": pdf.name, "parser_version": PARSER_VERSION,
                "pages": len(pages), "articles_and_forms": len(articles), "chunks": len(chunks)}
    write_json(out / "manifest.json", manifest)
    return manifest
