# %% [markdown]
# # 📄 Medical Guideline PDF Parser v1.1
# Enhanced to extract rich metadata from Chinese clinical guidelines and interpretation articles.
# Distinguishes between official guidelines and expert interpretations.
# Outputs: `guidelines.parsed.jsonl` with full bibliographic & structural metadata.

# %%
from pathlib import Path
import pdfplumber 
import re
import json
from typing import List, Dict, Any

# %%
# =============================
# 🧩 1. METADATA EXTRACTION UTILS (from filename)
# =============================

def extract_year_from_filename(filename: str) -> str:
    """Extracts 4-digit year from Chinese medical guideline filenames."""
    patterns = [
        r'[（\(]([12]\d{3})[）\)]',
        r'[（\(]([12]\d{3})[年\s]*(?:修订版|版|年版|年)?[）\)]?',
        r'([12]\d{3})[年\s]*(?:修订版|版|年版|年)?(?=[\s_）\)。\.\-]|$)',
        r'[（\(]?([12]\d{3})[）\)]?\s*\.pdf$',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    return "unknown"

def extract_doc_title(filename: str) -> str:
    """Extract clean document title from filename."""
    base = re.sub(r"_[^_]*\.pdf$", "", filename)
    base = re.sub(r"\.pdf$", "", base)
    base = re.sub(r"[（\(][^）\)]*[）\)]", "", base)
    base = re.sub(r"\s*—+\s*", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base or "未命名文档"

def extract_authors_from_filename(filename: str) -> List[str]:
    """Extract author names from filename (after last underscore)."""
    match = re.search(r"_([^_\(]+?)(?:\([12]\d{3}\))?\.pdf$", filename)
    if match:
        author_str = match.group(1).strip()
        authors = re.split(r"[,，、]", author_str)
        return [a.strip() for a in authors if a.strip()]
    return []

def extract_doc_type_from_filename(filename: str) -> str:
    """Classify document type from filename."""
    if "指南" in filename and "解读" not in filename:
        return "guideline"
    elif "解读" in filename or "浅析" in filename or "解析" in filename:
        return "guideline_interpretation"
    elif "共识" in filename:
        return "consensus"
    elif "证据总结" in filename:
        return "evidence_summary"
    else:
        return "other"

# %%
# =============================
# 📚 2. METADATA EXTRACTION UTILS (from document text — PREFERRED)
# =============================

def extract_metadata_from_text(text: str) -> Dict[str, Any]:
    """Extract rich metadata from first page of document text."""
    meta = {
        "authors": [],
        "corresponding_author": "",
        "affiliations": [],
        "journal_name": "",
        "volume": "",
        "issue": "",
        "pages": "",
        "doi": "",
        "keywords": [],
        "publish_date": "",
        "original_guideline_title": "",
        "doc_type": "other"  # Will be overridden if detected
    }

    # Extract author (first non-empty line after title, before postal code or affiliation)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()] #
    for i, line in enumerate(lines[:10]):  # Look in first 10 lines
        if re.match(r"^\d{6}", line):  # Postal code → previous line is likely author
            if i > 0:
                author_line = lines[i-1]
                authors = re.split(r"[,，、]", author_line)
                meta["authors"] = [a.strip() for a in authors if len(a.strip()) >= 2]
            break
        if "通信作者" in line:
            author_match = re.search(r"通信作者[:：]\s*([^\s，,、]+)", line)
            if author_match:
                meta["corresponding_author"] = author_match.group(1).strip()
                if not meta["authors"]:
                    meta["authors"] = [meta["corresponding_author"]]

    # Extract affiliation (lines with postal code or university)
    for line in lines[:15]:
        if re.search(r"\d{6}|大学|医院|中心", line) and len(line) > 10:
            meta["affiliations"].append(line)

    # Extract DOI
    doi_match = re.search(r"DOI\s*[:：]?\s*([0-9\.\s\/a-z-]+)", text, re.IGNORECASE)
    if doi_match:
        meta["doi"] = re.sub(r"\s+", "", doi_match.group(1)).strip()

    # Extract journal, volume, issue, pages from footer pattern
    # e.g., "·396· 中国心血管杂志 2024年 10月第 29卷第 5期"
    journal_match = re.search(r"·\d+·\s*([^\s]+?杂志|学报)\s*(\d{4})年\s*\d+月第\s*(\d+)卷第\s*(\d+)期", text)
    if journal_match:
        meta["journal_name"] = journal_match.group(1).strip()
        meta["publish_date"] = journal_match.group(2).strip()  # e.g., "2024"
        meta["volume"] = journal_match.group(3).strip()
        meta["issue"] = journal_match.group(4).strip()

    # Extract pages from header/footer (e.g., "·396·")
    page_match = re.search(r"·(\d+)·", text.splitlines()[0] if text.splitlines() else "")
    if page_match:
        start_page = page_match.group(1)
        # Try to find end page (often not available, so leave as single page)
        meta["pages"] = start_page

    # Extract keywords
    kw_match = re.search(r"【关键词】\s*([^\n【】]+)", text)
    if kw_match:
        kw_str = kw_match.group(1).strip()
        meta["keywords"] = [k.strip() for k in re.split(r"[,，;；、]", kw_str) if k.strip()]

    # Detect if this is an interpretation of a guideline
    guideline_ref_match = re.search(r"《([^》]+?指南[^》]*)》", text[:500])
    if guideline_ref_match:
        meta["original_guideline_title"] = guideline_ref_match.group(1).strip()
        if "解读" in text[:200] or "浅析" in text[:200]:
            meta["doc_type"] = "guideline_interpretation"

    # If no authors found but filename has them, fallback
    # (Handled in main function)

    return meta

# %%
# =============================
# 📄 3. TEXT EXTRACTION
# =============================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf: 
            pages = [p.extract_text() or "" for p in pdf.pages] 
        return "\n".join(pages)
    except Exception as e:
        print(f"⚠️  PDF extraction error: {e}")
        return ""

# %%
# =============================
# 🧱 4. CHUNKING LOGIC
# =============================

def chunk_by_rules(
    text: str,
    source_filename: str,
    year: str,
    doc_title: str,
    authors: List[str],
    doc_type: str,
    original_guideline_title: str = "",
    journal_name: str = "",
    volume: str = "",
    issue: str = "",
    pages: str = "",
    doi: str = "",
    keywords: List[str] = [],
    publish_date: str = ""
) -> List[Dict[str, Any]]:
    """Split text into chunks by section titles, with rich metadata."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    chunks, buf = [], []
    current_title = "未命名章节"

    TITLE_KEYWORDS = [
        "章", "节", "篇", "部分", "概述", "背景", "目的", "方法", "结果", "结论",
        "推荐", "建议", "管理", "治疗", "诊断", "评估", "定义", "目标",
        "一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、", "十、",
        "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.",
        "第一", "第二", "第三", "第四", "第五",
        "【", "】", "（", "）", "(", ")", "：", ":"
    ]

    for ln in lines:
        is_title = False

        if any(kw in ln for kw in TITLE_KEYWORDS) and 3 <= len(ln) <= 100:
            is_title = True
        elif ln.endswith("：") or ln.endswith(":") or \
             (len(ln) <= 50 and (ln.startswith("【") and ln.endswith("】"))):
            is_title = True
        elif 3 <= len(ln) <= 25 and not ln.endswith("。") and not ln.endswith("."):
            is_title = True
        elif re.match(r"^[0-9]+[\.、]\s*\S{3,}", ln):
            is_title = True

        if is_title:
            if buf:
                chunk_id = f"{re.sub(r'[^a-zA-Z0-9]', '_', doc_title)}_{year}_{len(chunks):03d}"
                chunk_meta = {
                    "source_filename": source_filename,
                    "doc_title": doc_title,
                    "section_title": current_title,
                    "authors": authors,
                    "year": year,
                    "doc_type": doc_type,
                    "original_guideline_title": original_guideline_title,
                    "journal_name": journal_name,
                    "volume": volume,
                    "issue": issue,
                    "pages": pages,
                    "doi": doi,
                    "keywords": keywords,
                    "publish_date": publish_date,
                    "chunk_id": chunk_id,
                    "extraction_method": "pdfplumber + rule-based + metadata extraction"
                }
                chunks.append({
                    "content": "\n".join(buf),
                    "meta": chunk_meta
                })
                buf = []
            current_title = ln
        else:
            buf.append(ln)

    # Flush final buffer
    if buf:
        chunk_id = f"{re.sub(r'[^a-zA-Z0-9]', '_', doc_title)}_{year}_{len(chunks):03d}"
        chunk_meta = {
            "source_filename": source_filename,
            "doc_title": doc_title,
            "section_title": current_title,
            "authors": authors,
            "year": year,
            "doc_type": doc_type,
            "original_guideline_title": original_guideline_title,
            "journal_name": journal_name,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "doi": doi,
            "keywords": keywords,
            "publish_date": publish_date,
            "chunk_id": chunk_id,
            "extraction_method": "pdfplumber + rule-based + metadata extraction"
        }
        chunks.append({
            "content": "\n".join(buf),
            "meta": chunk_meta
        })

    return chunks

# %%
# =============================
# 🚀 5. MAIN PROCESSING PIPELINE
# =============================

def main():
    in_dir = Path("data/guidelines")
    out_path = Path("data/guidelines.parsed.jsonl")

    print(f"📂 Input directory: {in_dir.absolute()}")
    pdf_files = list(in_dir.glob("*.pdf"))
    print(f"📄 Found {len(pdf_files)} PDF files.")

    if not pdf_files:
        print("❌ No PDFs found. Check directory path.")
        return

    with out_path.open("w", encoding="utf-8") as f:
        for pdf in pdf_files:
            print(f"\n--- 📄 Processing: {pdf.name} ---")

            # Step 1: Extract preliminary metadata from filename
            year = extract_year_from_filename(pdf.name)
            doc_title = extract_doc_title(pdf.name)
            authors_filename = extract_authors_from_filename(pdf.name)
            doc_type = extract_doc_type_from_filename(pdf.name)

            print(f"  📅 Year (from filename): {year}")
            print(f"  🏷️ Title (from filename): {doc_title}")
            print(f"  👥 Authors (from filename): {authors_filename}")
            print(f"  📑 Type (from filename): {doc_type}")

            # Step 2: Extract text
            text = extract_text_from_pdf(pdf)
            char_count = len(text.strip())
            print(f"  🔤 Extracted {char_count} characters.")

            if char_count == 0:
                print("  ⚠️  WARNING: No text extracted. File may be scanned.")
                continue

            # Step 3: Extract rich metadata from text (overrides filename where possible)
            text_meta = extract_metadata_from_text(text)

            # Merge: Prefer text-extracted metadata, fallback to filename
            authors = text_meta["authors"] if text_meta["authors"] else authors_filename
            doc_type_final = text_meta["doc_type"] if text_meta["doc_type"] != "other" else doc_type
            original_guideline_title = text_meta["original_guideline_title"]
            journal_name = text_meta["journal_name"]
            volume = text_meta["volume"]
            issue = text_meta["issue"]
            pages = text_meta["pages"]
            doi = text_meta["doi"]
            keywords = text_meta["keywords"]
            publish_date = text_meta["publish_date"]

            print(f"  ✍️  Authors (final): {authors}")
            print(f"  📚 Journal: {journal_name} {volume}({issue}), {pages}, {publish_date}")
            print(f"  🔗 DOI: {doi}")
            print(f"  🏷️ Keywords: {keywords}")
            print(f"  🧭 Original Guideline: {original_guideline_title}")
            print(f"  📑 Doc Type (final): {doc_type_final}")

            # Step 4: Chunk text with full metadata
            chunks = chunk_by_rules(
                text=text,
                source_filename=pdf.name,
                year=year,
                doc_title=doc_title,
                authors=authors,
                doc_type=doc_type_final,
                original_guideline_title=original_guideline_title,
                journal_name=journal_name,
                volume=volume,
                issue=issue,
                pages=pages,
                doi=doi,
                keywords=keywords,
                publish_date=publish_date
            )
            print(f"  🧩 Generated {len(chunks)} chunks.")

            if not chunks:
                print("  ❌ No chunks generated. Check chunking logic.")

            # Step 5: Write to JSONL
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n✅ Output written to: {out_path.absolute()}")
    print(f"💾 File size: {out_path.stat().st_size} bytes")

# %%
# =============================
# ▶️ 6. RUN
# =============================

if __name__ == "__main__":
    main()