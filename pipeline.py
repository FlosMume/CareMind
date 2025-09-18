# -*- coding: utf-8 -*-
"""
CareMind RAG — Inference Pipeline (Ollama + Qwen2)

Usage (from project root, with your venv activated):
    python -m rag.pipeline --q "2型糖尿病合并高血压的血压目标？" --drug "氨氯地平" --k 4

Environment (.env loaded by your launcher or shell):
    OLLAMA_BASE_URL=http://localhost:11434
    LLM_MODEL=qwen2:7b-instruct
    LLM_NUM_CTX=16384        # optional
    LLM_TEMPERATURE=0.1      # optional
    LLM_TOP_P=0.9            # optional
    LLM_SEED=42              # optional

This module depends on:
    - rag.retriever.search_guidelines(question, k)
      -> list[{"content": str, "meta": {...}}]
    - rag.retriever.fetch_drug(drug_name)
      -> dict with keys like name/indications/contraindications/interactions/...
"""

from __future__ import annotations
import os
import json
import time
import argparse
import requests
from typing import Any, Dict, List, Optional

from .retriever import search_guidelines, fetch_drug
from .prompt import SYSTEM, USER_TEMPLATE

# -------- LLM endpoint & defaults --------
OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
MODEL  = os.getenv("LLM_MODEL", "qwen2:7b-instruct")

def _ollama_options() -> Dict[str, Any]:
    """
    Collect optional generation parameters from environment variables.
    Returned dict is passed as "options" to /api/chat.
    """
    def _maybe_float(key: str) -> Optional[float]:
        v = os.getenv(key)
        if v is None: return None
        try:
            return float(v)
        except ValueError:
            return None

    def _maybe_int(key: str) -> Optional[int]:
        v = os.getenv(key)
        if v is None: return None
        try:
            return int(v)
        except ValueError:
            return None

    opts: Dict[str, Any] = {}
    t = _maybe_float("LLM_TEMPERATURE")
    if t is not None: opts["temperature"] = t
    tp = _maybe_float("LLM_TOP_P")
    if tp is not None: opts["top_p"] = tp
    seed = _maybe_int("LLM_SEED")
    if seed is not None: opts["seed"] = seed
    nctx = _maybe_int("LLM_NUM_CTX")
    if nctx is not None: opts["num_ctx"] = nctx
    # Keep responses concise and deterministic by default
    if "temperature" not in opts:
        opts["temperature"] = 0.1
    return opts

# -------- LLM call --------
def llm_chat(system: str, user: str, timeout: int = 120, retries: int = 2) -> str:
    """
    Call Ollama /api/chat with (system, user). Retries on transient failures.
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": _ollama_options(),
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # Standard Ollama chat response shape:
            # { "message": {"role":"assistant","content":"..."}, ... }
            msg = data.get("message", {})
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty content from LLM.")
            return content
        except (requests.RequestException, ValueError, KeyError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise RuntimeError(f"Ollama chat failed after {retries+1} attempts: {e}") from e
    # Unreachable
    raise RuntimeError(f"Ollama chat failed: {last_err}")

# -------- Formatting helpers --------
def format_guideline_snippets(hits: List[Dict[str, Any]]) -> str:
    """
    Convert guideline hits into a compact, readable block for the prompt.
    """
    if not hits:
        return "未检索到相关指南片段。"
    lines: List[str] = []
    for h in hits:
        meta = h.get("meta", {}) or {}
        src = meta.get("source", "未知来源")
        year = meta.get("year", "未知年份")
        title = meta.get("title", "").strip()
        title_str = f"{title} | " if title else ""
        content = (h.get("content") or "").strip()
        # Trim very long content for prompt safety
        content = content[:1200]
        lines.append(f"【{title_str}{src} | {year}】\n{content}")
    return "\n\n".join(lines)

def format_drug_info(drug: Optional[Dict[str, Any]]) -> str:
    """
    Convert structured drug dict into a stable text block for the prompt.
    """
    if not drug:
        return "未指定药品"
    # Only include non-empty known keys in a consistent order
    keys = [
        ("name", "药品名称"),
        ("indications", "适应症"),
        ("contraindications", "禁忌症"),
        ("interactions", "药物相互作用"),
        ("dosage", "用法用量"),
        ("pregnancy_category", "妊娠分级"),
        ("source", "来源"),
    ]
    lines: List[str] = []
    for k, label in keys:
        v = drug.get(k)
        if v:
            lines.append(f"{label}: {v}")
    if not lines:
        return "（药品信息存在，但字段为空）"
    return "\n".join(lines)

# -------- Public API --------
def answer(question: str, drug_name: Optional[str] = None, k: int = 4) -> Dict[str, Any]:
    """
    Main entry: retrieve, compose prompt, call LLM, and return rich result.
    Returns:
        {
          "output": str,
          "guideline_hits": list[...],
          "drug": dict|None,
          "prompt": {"system": str, "user": str}
        }
    """
    # 1) Retrieve guideline snippets
    g_hits = search_guidelines(question, k=max(1, int(k)) if k else 4) or []
    g_text = format_guideline_snippets(g_hits)

    # 2) Fetch structured drug info (optional)
    d = fetch_drug(drug_name) if drug_name else None
    d_text = format_drug_info(d)

    # 3) Compose user message
    user = USER_TEMPLATE.format(
        question=question.strip(),
        guideline_snippets=g_text,
        drug_info=d_text,
        k=k,
    )

    # 4) Call LLM
    output = llm_chat(SYSTEM, user)

    return {
        "output": output,
        "guideline_hits": g_hits,
        "drug": d,
        "prompt": {"system": SYSTEM, "user": user},
    }

# -------- CLI for quick testing --------
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="CareMind-RAG-Pipeline",
        description="Run Q&A over Chinese medical guidelines + structured drug table via Ollama Qwen2."
    )
    p.add_argument("--q", "--question", dest="question", required=True, help="临床问题（中文推荐）")
    p.add_argument("--drug", dest="drug", default=None, help="药品名称（可选）")
    p.add_argument("--k", dest="k", type=int, default=4, help="检索到的指南片段数量（Top-k）")
    p.add_argument("--print-prompt", action="store_true", help="调试：打印拼接后的 user prompt")
    p.add_argument("--json", action="store_true", help="以 JSON 格式输出完整结果")
    return p

def main() -> None:
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
