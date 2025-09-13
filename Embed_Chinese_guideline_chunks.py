#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Embed Chinese guideline chunks → ChromaDB (OOM-safe, idempotent, metadata-robust)
===============================================================================

What this script does (high-level)
----------------------------------
1) Streams a JSONL file that contains one chunk per line:
     {
       "content": "分章文本……",
       "meta": {
         "source": "中国高血压防治指南(2024年修订版).pdf",
         "year": 2024,
         "section": "3.2 降压目标",
         "chunk_id": 57,
         "authors": ["秦煜", "张三"]    # lists allowed; we sanitize below
       }
     }

2) Uses a Chinese-capable SentenceTransformer (defaults to BAAI/bge-small-zh)
   to embed "content" with **GPU if available**.

3) Writes embeddings, documents, and **sanitized metadata** into a persistent
   **Chroma** collection on disk, in an **idempotent** way:
   - Uses **upsert** when available (Chromadb>=0.5) to overwrite duplicates.
   - Strengthens IDs to minimize collisions.
   - De-duplicates IDs **inside each batch**.
   - Final safety net: per-item upsert/repair if a batch raises an error.

Inputs (files & environment variables)
-------------------------------------
• JSONL file (default `data/guidelines.parsed.jsonl`)
  - Set via env var `CAREMIND_DATA`.

• Environment variables (optional, with sensible defaults):
  CHROMA_PERSIST_DIR  : Directory for Chroma persistence (default './chroma_store')
  CHROMA_COLLECTION   : Collection name (default 'guideline_chunks')
  CAREMIND_DATA       : Input JSONL path (default 'data/guidelines.parsed.jsonl')
  EMBEDDING_MODEL     : Model id (default 'BAAI/bge-small-zh')
  EMBED_BATCH_SIZE    : Starting batch size (default '16')
  EMBED_FP16          : '1' to allow fp16 autocast on CUDA (default '1')
  EMBED_PROGRESS      : '1' to show progress bars (default '1')
  EMBED_MAX_LEN       : Max sequence length for encoder (default '384')
  OOM_CPU_FALLBACK    : '1' to allow CPU fallback when BS=1 still OOM (default '1')

Outputs
-------
• A persistent Chroma database on disk containing:
    - ids: stable, low-collision per-chunk IDs
    - embeddings: float vectors
    - documents: original text chunks
    - metadatas: scalar/JSON-safe metadata
  Location and collection are controlled by CHROMA_PERSIST_DIR / CHROMA_COLLECTION.

Run
---
$ python embed_to_chroma.py
(or set envs first, e.g., EMBED_MAX_LEN=256 EMBED_BATCH_SIZE=8 for tighter VRAM)
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Iterable, List, Dict, Any

# NOTE: We intentionally do NOT set PYTORCH_CUDA_ALLOC_CONF here because some
# PyTorch builds require a specific format and may crash. All OOM tactics below
# (batch backoff, fp16, truncation, cache clears, CPU fallback) work without it.

from dotenv import load_dotenv
load_dotenv()

import torch
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb import errors as chroma_errors  # for DuplicateIDError handling
from tqdm import tqdm

# ---------------------------
# Config (safe defaults; override via env vars)
# ---------------------------
PERSIST_DIR      = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION_NAME  = os.getenv("CHROMA_COLLECTION", "guideline_chunks")
DATA_PATH        = os.getenv("CAREMIND_DATA", "data/guidelines.parsed.jsonl")

# Chinese-capable model; bge-* are strong + efficient. Start small on 12GB VRAM.
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh")

# Start conservatively; dynamic backoff will reduce further on OOM.
START_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

# fp16 autocast reduces memory on CUDA and is safe for sentence-transformers inference
USE_FP16         = os.getenv("EMBED_FP16", "1") == "1"

# Lower seq length cuts memory on long Chinese paragraphs; 384 or even 256 works well
MAX_SEQ_LEN      = int(os.getenv("EMBED_MAX_LEN", "384"))

# nice progress bars
SHOW_PROGRESS    = os.getenv("EMBED_PROGRESS", "1") == "1"

# If bs=1 still OOMs (rare), push that batch to CPU to finish and keep going
CPU_FALLBACK     = os.getenv("OOM_CPU_FALLBACK", "1") == "1"

# Ensure output dir exists early
Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

# ---------------------------
# Device and model load
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    # Not a memory tactic, but improves throughput on Ada/RTX40
    torch.backends.cuda.matmul.allow_tf32 = True

embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
# Reduce max sequence length to save memory; long texts will be truncated
embed_model.max_seq_length = MAX_SEQ_LEN

# ---------------------------
# ID generation (stable + low collision)
# ---------------------------
def stable_id(meta: Dict[str, Any], content: str) -> str:
    """
    Stable, low-collision ID:
    • Prefer source|chunk_id|sha12(content) when source+chunk_id exist.
    • Fallback binds to source-hash + content-hash so identical text in different
      files remains separate.
    This keeps re-runs idempotent and minimizes accidental collisions.
    """
    src = str(meta.get("source", "")).strip()
    cid = meta.get("chunk_id", None)
    ch  = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    if src and cid is not None:
        return f"{src}|{cid}|{ch}"
    sh  = hashlib.sha1(src.encode("utf-8")).hexdigest()[:8]
    return f"g_{sh}_{hashlib.sha1(content.encode('utf-8')).hexdigest()[:16]}"

# ---------------------------
# JSONL streaming (robust to bad lines)
# ---------------------------
def jsonl_iter(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping bad JSON at line {ln}: {e}")
                continue

# ---------------------------
# Metadata sanitizer (prevents list/dict errors in Chroma)
# ---------------------------
def sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma requires scalar metadata values: str, int, float, bool, or None.
    This function converts:
        • List/Tuple/Set of scalars → "a, b, c"
        • Complex lists (mixed/nested) → JSON string
        • Dicts → JSON string (preserve Chinese with ensure_ascii=False)
        • Other types → str(v)
    Example fix:
        ["秦煜"]  → "秦煜"
    """
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        elif isinstance(v, (list, tuple, set)):
            seq = list(v)
            if all(isinstance(x, (str, int, float, bool)) or x is None for x in seq):
                clean[k] = ", ".join("" if x is None else str(x) for x in seq)
            else:
                clean[k] = json.dumps(seq, ensure_ascii=False)
        elif isinstance(v, dict):
            clean[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
        else:
            clean[k] = str(v)
    return clean

# ---------------------------
# Optional VRAM stats (debug)
# ---------------------------
def cuda_mem_summary(prefix: str = ""):
    if not torch.cuda.is_available():
        return
    try:
        free, total = torch.cuda.mem_get_info()  # bytes
        used = total - free
        gb = 1024**3
        print(f"[VRAM] {prefix} used={used/gb:.2f} GB / total={total/gb:.2f} GB (free={free/gb:.2f} GB)")
    except Exception:
        pass

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ---------------------------
# OOM-resilient encoder with dynamic batch backoff + optional CPU fallback
# ---------------------------
@torch.no_grad()
def encode_with_backoff(texts: List[str], start_bs: int, use_fp16: bool, model: SentenceTransformer, cpu_fallback: bool):
    """
    Try encoding with the given batch size. On CUDA OOM:
      - Halve batch size and retry.
      - If batch size is 1 and still OOM, optionally move to CPU for that batch.
    Returns a List[List[float]].
    """
    bs = max(1, start_bs)
    current_device = model.device  # torch.device('cuda') or ('cpu')

    while True:
        try:
            if use_fp16 and current_device.type == "cuda":
                # AMP saves memory; sentence-transformers is autocast-aware
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    vecs = model.encode(
                        texts,
                        batch_size=bs,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
            else:
                vecs = model.encode(
                    texts,
                    batch_size=bs,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            return vecs.tolist()

        except RuntimeError as e:
            msg = str(e)
            if "CUDA out of memory" in msg or "CUBLAS" in msg:
                print(f"[OOM] CUDA OOM at batch_size={bs} (len={len(texts)}). Backing off…")
                clear_cuda_cache()
                if bs > 1:
                    bs = max(1, bs // 2)
                    continue
                # bs==1 and still OOM → optional CPU fallback
                if cpu_fallback and current_device.type == "cuda":
                    print("[OOM] Switching this batch to CPU to complete.")
                    model.to("cpu")
                    current_device = torch.device("cpu")
                    vecs = model.encode(
                        texts,
                        batch_size=1,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    # Move back to CUDA for subsequent batches if available
                    if torch.cuda.is_available():
                        model.to("cuda")
                    return vecs.tolist()
                else:
                    raise
            else:
                # Not an OOM error; surface it
                raise

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    # Resolve & validate input
    data_path = Path(DATA_PATH).expanduser().resolve()
    if not data_path.exists():
        raise SystemExit(f"❌ JSONL not found: {data_path}")

    # Prepare Chroma
    client = PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Count lines for progress bar (optional; ok to skip on huge files)
    try:
        num_lines = sum(1 for _ in data_path.open("r", encoding="utf-8"))
    except Exception:
        num_lines = None

    iterator = jsonl_iter(data_path)
    if SHOW_PROGRESS:
        iterator = tqdm(iterator, total=num_lines, desc="Reading JSONL")

    BUFFER: List[Dict[str, Any]] = []
    total = 0
    current_bs = START_BATCH_SIZE

    print(f"📦 Persist dir: {PERSIST_DIR}")
    print(f"🗃️  Collection: {COLLECTION_NAME}")
    print(f"🧠 Model: {EMBEDDING_MODEL} | Device: {device} | fp16: {USE_FP16} | max_seq_len: {MAX_SEQ_LEN}")
    print(f"📑 Input: {data_path}")
    print(f"⚙️  Start batch size: {START_BATCH_SIZE} | CPU fallback: {CPU_FALLBACK}")
    if torch.cuda.is_available():
        cuda_mem_summary("start")

    # Inner helper: flush a batch to Chroma robustly
    def flush(batch: List[Dict[str, Any]]):
        nonlocal total, current_bs
        if not batch:
            return

        # Extract and sanitize
        docs  = [b["content"] for b in batch]
        metas = [sanitize_meta(b["meta"]) for b in batch]
        ids   = [stable_id(b["meta"], b["content"]) for b in batch]

        # Embed with OOM backoff
        try:
            vecs = encode_with_backoff(
                docs,
                start_bs=current_bs,
                use_fp16=USE_FP16,
                model=embed_model,
                cpu_fallback=CPU_FALLBACK
            )
        except RuntimeError as e:
            raise RuntimeError(f"Embedding failed for a batch of size {len(docs)}: {e}") from e
        finally:
            clear_cuda_cache()

        # Intra-batch de-duplication (keeps only first occurrence of each ID)
        seen = set()
        keep = []
        for i, _id in enumerate(ids):
            if _id in seen:
                continue
            seen.add(_id)
            keep.append(i)
        if len(keep) != len(ids):
            print(f"[dedupe] removed {len(ids) - len(keep)} duplicate id(s) in the same batch)")

        ids   = [ids[i] for i in keep]
        docs  = [docs[i] for i in keep]
        vecs  = [vecs[i] for i in keep]
        metas = [metas[i] for i in keep]

        # Write to Chroma, preferring upsert for idempotency
        try:
            if hasattr(collection, "upsert"):
                collection.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)
            else:
                # Older Chroma: emulate upsert with add+update split
                existing = set(collection.get(ids=ids).get("ids", []) or [])
                new_idx = [i for i, _id in enumerate(ids) if _id not in existing]
                upd_idx = [i for i, _id in enumerate(ids) if _id in existing]
                if new_idx:
                    collection.add(
                        ids=[ids[i] for i in new_idx],
                        embeddings=[vecs[i] for i in new_idx],
                        documents=[docs[i] for i in new_idx],
                        metadatas=[metas[i] for i in new_idx],
                    )
                if upd_idx:
                    collection.update(
                        ids=[ids[i] for i in upd_idx],
                        embeddings=[vecs[i] for i in upd_idx],
                        documents=[docs[i] for i in upd_idx],
                        metadatas=[metas[i] for i in upd_idx],
                    )

        except (chroma_errors.DuplicateIDError, ValueError) as e:
            # Final safety net: heal per item (rare edge cases)
            print(f"[warn] batch write hit {type(e).__name__}: {e}\n[repair] trying per-item upsert/repair")
            for i in range(len(ids)):
                try:
                    if hasattr(collection, "upsert"):
                        collection.upsert(
                            ids=[ids[i]],
                            embeddings=[vecs[i]],
                            documents=[docs[i]],
                            metadatas=[metas[i]],
                        )
                    else:
                        if collection.get(ids=[ids[i]]).get("ids"):
                            collection.update(
                                ids=[ids[i]],
                                embeddings=[vecs[i]],
                                documents=[docs[i]],
                                metadatas=[metas[i]],
                            )
                        else:
                            collection.add(
                                ids=[ids[i]],
                                embeddings=[vecs[i]],
                                documents=[docs[i]],
                                metadatas=[metas[i]],
                            )
                except Exception as inner:
                    # As a last resort, stringify all non-scalar meta fields and retry once
                    fallback_meta = {
                        k: (v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v, ensure_ascii=False))
                        for k, v in metas[i].items()
                    }
                    if hasattr(collection, "upsert"):
                        collection.upsert(
                            ids=[ids[i]],
                            embeddings=[vecs[i]],
                            documents=[docs[i]],
                            metadatas=[fallback_meta],
                        )
                    else:
                        if collection.get(ids=[ids[i]]).get("ids"):
                            collection.update(
                                ids=[ids[i]],
                                embeddings=[vecs[i]],
                                documents=[docs[i]],
                                metadatas=[fallback_meta],
                            )
                        else:
                            collection.add(
                                ids=[ids[i]],
                                embeddings=[vecs[i]],
                                documents=[docs[i]],
                                metadatas=[fallback_meta],
                            )

        total += len(ids)
        if SHOW_PROGRESS:
            tqdm.write(f"✅ Flushed {len(ids)} items (total={total})")
        if torch.cuda.is_available() and total % max(64, START_BATCH_SIZE) == 0:
            cuda_mem_summary(f"after {total}")

    # Stream → buffer → flush
    for obj in iterator:
        # Minimal schema validation
        if not isinstance(obj, dict) or "content" not in obj or "meta" not in obj:
            continue
        BUFFER.append(obj)
        if len(BUFFER) >= current_bs:
            flush(BUFFER)
            BUFFER.clear()

    # Flush remainder
    if BUFFER:
        flush(BUFFER)
        BUFFER.clear()

    print(f"\n🎉 Done. Inserted {total} chunks into '{COLLECTION_NAME}' at '{PERSIST_DIR}'.")
    print(f"Model: {EMBEDDING_MODEL} | Device: {device} | Start batch: {START_BATCH_SIZE} | max_seq_len: {MAX_SEQ_LEN}")
    if torch.cuda.is_available():
        cuda_mem_summary("end")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    main()
