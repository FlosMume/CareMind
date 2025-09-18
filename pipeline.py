# -*- coding: utf-8 -*-
"""
CareMind RAG — Inference Pipeline (Ollama + Qwen2)

This file is the "glue" that ties everything together:
1) it pulls guideline snippets and drug info from the retriever,
2) builds a clean, compliance-oriented prompt,
3) calls a local LLM (Ollama with Qwen2),
4) and returns the model's answer along with the retrieved evidence.

--------------------------------------------------------------------------------
Quick Start (from project root):
    conda activate caremind
    export OLLAMA_BASE_URL=http://localhost:11434
    export LLM_MODEL=qwen2:7b-instruct
    python -m rag.pipeline --q "老年高血压合并2型糖尿病的血压与血糖目标？" --drug "氨氯地平" --k 4

Env Vars (tune generation or point to a different LLM):
    OLLAMA_BASE_URL   default http://localhost:11434
    LLM_MODEL         default qwen2:7b-instruct
    LLM_NUM_CTX       optional context window size
    LLM_TEMPERATURE   optional decoding temperature
    LLM_TOP_P         optional nucleus sampling
    LLM_SEED          optional seed for reproducibility

High-level flow:
    [retriever.search_guidelines] -> list of top-k snippets (with metadata)
                         |
                         v
    [format_guideline_snippets] -> readable "【标题 | 来源 | 年份】\n片段" text block
                         |
                         +--- [retriever.search_drugs or fetch_drug] -> drug dict
                         |                   |
                         |                   v
                         |            [format_drug_info] -> readable drug info
                         v
      [prompt.SYSTEM + prompt.USER_TEMPLATE] -> user/system messages
                         |
                         v
                    [llm_chat] -> call Ollama /api/chat (fallback /api/generate)
                         |
                         v
                    dict(output, guideline_hits, drug, prompt)
--------------------------------------------------------------------------------
"""

from __future__ import annotations
import os
import json
import time
import argparse
import requests
from typing import Any, Dict, List, Optional

# We import the retriever module (your local search/DB access code),
# and the pre-defined prompts to keep answers consistent & compliant.
from . import retriever as R
from .prompt import SYSTEM, USER_TEMPLATE

# =============================================================================
#                       LLM endpoint & generation options
# =============================================================================

# Where Ollama is running and which model to use.
# - If you use WSL or remote machine, change OLLAMA_BASE_URL accordingly.
OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
MODEL  = os.getenv("LLM_MODEL", "qwen2:7b-instruct")

def _ollama_options() -> Dict[str, Any]:
    """
    Read optional decoding parameters from environment variables.
    These are passed to Ollama under "options".
    Beginners:
      - temperature: lower (e.g., 0.1) = more deterministic, safer for medical QA
      - top_p: nucleus sampling; often keep default if unsure
      - num_ctx: context window size (tokens); bigger lets you pass more text
    """
    def _f(name: str) -> Optional[float]:
        v = os.getenv(name)
        if v is None: return None
        try: return float(v)
        except ValueError: return None

    def _i(name: str) -> Optional[int]:
        v = os.getenv(name)
        if v is None: return None
        try: return int(v)
        except ValueError: return None

    opts: Dict[str, Any] = {}
    t  = _f("LLM_TEMPERATURE")
    tp = _f("LLM_TOP_P")
    sd = _i("LLM_SEED")
    nc = _i("LLM_NUM_CTX")

    if t  is not None: opts["temperature"] = t
    if tp is not None: opts["top_p"]       = tp
    if sd is not None: opts["seed"]        = sd
    if nc is not None: opts["num_ctx"]     = nc

    # Safe default: keep generation stable for clinical style answers
    if "temperature" not in opts:
        opts["temperature"] = 0.1

    return opts

def llm_chat(system: str, user: str, timeout: int = 120, retries: int = 2) -> str:
    """
    Call the local LLM through Ollama.
    - We prefer the newer /api/chat endpoint (role-based messages).
    - If /api/chat isn't available (older Ollama), we fallback to /api/generate.

    Why a fallback? Because some environments ship older Ollama builds
    or custom servers that only support /api/generate.

    Args:
      system: system prompt that sets role & rules (e.g., compliance)
      user:   user message composed from the question + retrieved evidence
      timeout: HTTP timeout seconds
      retries: simple retry count for transient network hiccups

    Returns:
      The assistant's text content (string), or raises RuntimeError on failure.
    """
    options = _ollama_options()

    # Payload for /api/chat (messages with roles)
    chat_payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,        # we want a single JSON response
        "options": options,     # pass decoding options
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            # Try the modern endpoint first
            r = requests.post(f"{OLLAMA}/api/chat", json=chat_payload, timeout=timeout)
            if r.status_code in (404, 405):
                # Not supported on this server; trigger fallback
                raise requests.HTTPError(f"{r.status_code} Not supported", response=r)

            r.raise_for_status()
            data = r.json()
            # Standard shape: {"message": {"content": "..."}}
            content = (data.get("message") or {}).get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty content from /api/chat")
            return content

        except (requests.RequestException, ValueError, KeyError) as e:
            last_err = e

            # Fallback to /api/generate (older Ollama uses "prompt" instead of messages)
            if isinstance(e, requests.HTTPError) and getattr(e, "response", None) \
               and e.response is not None and e.response.status_code in (404, 405):
                try:
                    # We concatenate system + user in a readable way
                    prompt = (
                        "【系统角色】\n" + system.strip() +
                        "\n\n【用户】\n" + user.strip() +
                        "\n\n请严格按照系统角色与合规要求作答。"
                    )
                    gen_payload = {
                        "model": MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": options,
                    }
                    g = requests.post(f"{OLLAMA}/api/generate", json=gen_payload, timeout=timeout)
                    g.raise_for_status()
                    data = g.json()
                    content = data.get("response", "")
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("Empty content from /api/generate")
                    return content
                except Exception as ee:
                    # If fallback also fails, save the error for the final raise
                    last_err = ee

            # Simple exponential-ish backoff between attempts
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue

            # Give up after retry budget
            raise RuntimeError(
                f"Ollama request failed after {retries+1} attempts: {last_err}"
            ) from last_err

# =============================================================================
#                   Formatting helpers (make snippets readable)
# =============================================================================

def format_guideline_snippets(hits: List[Dict[str, Any]]) -> str:
    """
    Turn guideline hits into a displayable block.
    - We try hard to populate "来源(source)" and "年份(year)" even if the
      original metadata is messy, using common alternatives like journal_name
      or the filename stem.
    - We also de-duplicate similar entries lightly (title, source, year, page).

    Beginners: a "hit" here is a dict like:
      {
        "content": "...a snippet of text...",
        "meta": {"title": "...", "source": "...", "year": "...", ...},
        "score": 0.87,
        ...
      }
    """
    import os, re

    def _first(*vals):
        """Return the first non-empty string from vals."""
        for v in vals:
            if v is not None and str(v).strip():
                return str(v).strip()
        return None

    def _stem(path):
        """Get `filename` without extension from a path-like string."""
        if not path: return None
        base = os.path.basename(str(path))
        return re.sub(r'\.[^.]+$', '', base)

    def _infer_source(meta):
        """
        Try different keys to infer a readable "来源".
        If all else fails, we fall back to filename or title.
        """
        return _first(
            meta.get("source"),
            meta.get("org"), meta.get("organization"), meta.get("issuer"),
            meta.get("journal_name"), meta.get("journal"), meta.get("publisher"),
            meta.get("collection"), meta.get("website"), meta.get("book_title"),
            meta.get("conference"),
            _stem(meta.get("source_filename") or meta.get("file")),
            meta.get("title"),
        ) or "未知来源"

    def _infer_year(meta):
        """
        Try to get a 4-digit year from year/date/title/filename.
        If parsing fails, return "未知年份".
        """
        y = _first(meta.get("year"), meta.get("pub_year"),
                   meta.get("publish_date"), meta.get("date"))
        if y:
            m = re.search(r'(19|20)\d{2}', str(y))
            if m: return m.group(0)
        for f in ("title", "source_filename", "file"):
            s = meta.get(f)
            if s:
                m = re.search(r'(19|20)\d{2}', str(s))
                if m: return m.group(0)
        return "未知年份"

    if not hits:
        return "未检索到相关指南片段。"

    seen = set()
    lines: List[str] = []
    for h in hits:
        meta = h.get("meta") or {}
        src   = _infer_source(meta)
        year  = _infer_year(meta)
        title = _first(meta.get("title"), _stem(meta.get("source_filename") or meta.get("file"))) or ""
        # Use a simple key to avoid showing duplicates
        key   = (title, src, year, meta.get("page") or meta.get("pages"))
        if key in seen:
            continue
        seen.add(key)

        # Trim content to keep prompts short (LLM context is precious)
        content = (h.get("content") or "").strip()[:1200]
        title_s = f"{title} | " if title else ""
        lines.append(f"【{title_s}{src} | {year}】\n{content}")

    return "\n\n".join(lines)

def format_drug_info(drug: Optional[Dict[str, Any]]) -> str:
    """
    Convert the structured drug dict into a readable block.
    The retriever provides either:
      - search_drugs(...)[0]["meta"] as the drug record, OR
      - fetch_drug(...) as a full dict

    We render only common fields; missing fields are skipped.
    """
    if not drug:
        return "未指定药品"

    keys = [
        ("name", "药品名称"),
        ("indications", "适应症"),
        ("contraindications", "禁忌症"),
        ("interactions", "药物相互作用"),
        ("dosage", "用法用量"),
        ("pregnancy_category", "妊娠分级"),
        ("source", "来源"),
    ]
    lines = []
    for k, label in keys:
        v = drug.get(k)
        if v:
            lines.append(f"{label}: {v}")

    return "\n".join(lines) if lines else "（药品信息存在，但字段为空）"

# =============================================================================
#                    Compose prompt + call LLM (main routine)
# =============================================================================

def _pick_drug_record(drug_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Be flexible with retriever interfaces:
      1) If search_drugs exists, use top-1 hit's meta as the structured record.
      2) Else if fetch_drug exists, use that dict directly.
      3) Else return None and the prompt will say "未指定药品".
    """
    if not drug_name:
        return None
    try:
        search_drugs = getattr(R, "search_drugs", None)
        if callable(search_drugs):
            d_hits = search_drugs(drug_name, k=1) or []
            if d_hits:
                return (d_hits[0].get("meta") or {})
    except Exception:
        pass

    try:
        fetch_drug = getattr(R, "fetch_drug", None)
        if callable(fetch_drug):
            return fetch_drug(drug_name)
    except Exception:
        pass

    return None

def answer(question: str, drug_name: Optional[str] = None, k: int = 4) -> Dict[str, Any]:
    """
    The "do everything" function:
      - retrieve guideline snippets
      - retrieve (optional) drug info
      - build user prompt from template
      - call the LLM
      - return everything for UI or logging

    Args:
      question: clinical question in Chinese (recommended)
      drug_name: optional drug name (Chinese/English both OK)
      k: number of top guideline snippets to include

    Returns:
      A dict with:
        "output": the model's answer (string),
        "guideline_hits": the raw hits we used (for debugging),
        "drug": the structured drug record (or None),
        "prompt": {"system": SYSTEM, "user": the final rendered user prompt}
    """
    # 1) Retrieve guideline snippets (top-k)
    g_hits = R.search_guidelines(question, k=max(1, int(k)) if k else 4) or []
    g_text = format_guideline_snippets(g_hits)

    # 2) (Optional) drug info
    drug = _pick_drug_record(drug_name)
    d_text = format_drug_info(drug)

    # 3) Build user message via template (keeps formatting consistent)
    user = USER_TEMPLATE.format(
        question=question.strip(),
        guideline_snippets=g_text,
        drug_info=d_text,
        k=k,
    )

    # 4) Call LLM through Ollama
    output = llm_chat(SYSTEM, user)

    return {
        "output": output,
        "guideline_hits": g_hits,
        "drug": drug,
        "prompt": {"system": SYSTEM, "user": user},
    }

# =============================================================================
#                                   CLI
# =============================================================================

def _build_cli() -> argparse.ArgumentParser:
    """
    Simple command-line interface so you can test the pipeline without a UI.
    Example:
      python -m rag.pipeline --q "...问题..." --drug "氨氯地平" --k 4 --print-prompt
    """
    p = argparse.ArgumentParser(
        prog="CareMind-RAG-Pipeline",
        description="Run Q&A over Chinese medical guidelines + structured drug table via Ollama Qwen2."
    )
    p.add_argument("--q", "--question", dest="question", required=True,
                   help="临床问题（中文推荐）")
    p.add_argument("--drug", dest="drug", default=None,
                   help="药品名称（可选）")
    p.add_argument("--k", dest="k", type=int, default=4,
                   help="检索到的指南片段数量（Top-k）")
    p.add_argument("--print-prompt", action="store_true",
                   help="调试：打印拼接后的 user prompt")
    p.add_argument("--json", action="store_true",
                   help="以 JSON 格式输出完整结果")
    return p

def main() -> None:
    """
    Entry point for `python -m rag.pipeline`.
    Prints either:
      - the model's plain answer, or
      - (with --print-prompt) the system prompt, user prompt, and the answer,
      - (with --json) a JSON blob with everything (handy for logging).
    """
    args = _build_cli().parse_args()
    res = answer(args.question, drug_name=args.drug, k=args.k)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    if args.print_prompt:
        print("====== SYSTEM ======")
        print(res["prompt"]["system"])
        print("\n====== USER ======")
        print(res["prompt"]["user"])
        print("\n====== OUTPUT ======")

    print(res["output"])

if __name__ == "__main__":
    main()
