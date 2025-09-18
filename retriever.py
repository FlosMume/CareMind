#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CareMind | 混合检索（Chroma + SQLite）
File: rag/retriever.py

与当前目录结构匹配：
- 向量库目录: ./chroma_store
- SQLite 路径: ./db/drugs.sqlite
- （可选）原始 JSONL: ./data/guidelines.parsed.jsonl

新增/改动要点
-----------
1) Chroma 集合自动识别：
   - 首选环境变量 CHROMA_COLLECTION（默认 'guideline_chunks'）
   - 若该集合不存在或为空 → 自动扫描已有集合，选择**非空**集合
   - 若没有任何集合或集合为空 → 友好报错并给出诊断建议

2) 安全 where 处理：绝不向 Chroma 传入空字典（避免 “Expected where to have exactly one operator” 报错）

3) 诊断模式：--diagnose 打印集合列表、每个集合的计数、SQLite FTS 状态等

4) SQLite：
   - 优先使用 FTS5（drugs_fts + bm25），否则回退 LIKE
"""

from __future__ import annotations
import os
import sqlite3
from typing import Any, Dict, List, Optional

from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# ---- Env / Defaults (match your screenshots)
CHROMA_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
# COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION", "guideline_chunks") 
COLLECTION    = os.getenv("CHROMA_COLLECTION", "guideline_chunks_1024_v2") # preferred name
SQLITE_PATH: str = os.getenv("SQLITE_PATH", "./db/drugs.sqlite")
PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
# EMBED_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh")
EMBED_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")


try:
    import torch
    _dev = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    _dev = "cpu"
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL,
    # device="cuda"  # or "cpu" if no GPU
    device=_dev
    # normalize=True  # if your version supports; otherwise leave off
)

# ---- Lazy singletons
_embedder: Optional[SentenceTransformer] = None
# _chroma_client: Optional[PersistentClient] = None
_chroma_client = PersistentClient(path=PERSIST_DIR)
_chroma_collection = None

# client = PersistentClient(path=PERSIST_DIR)
# col = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

# =========================
# Embeddings
# =========================
def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def embed_text(text: str) -> List[float]:
    model = get_embedder()
    return model.encode([text], normalize_embeddings=True).tolist()[0]


# =========================
# Chroma helpers
# =========================
def _ensure_chroma_dir():
    if not os.path.isdir(CHROMA_DIR):
        raise FileNotFoundError(
            f"Chroma dir not found: {CHROMA_DIR}\n"
            "请确认向量库已写入（例如运行你的向量化脚本），或修改 CHROMA_PERSIST_DIR。"
        )


def _open_client() -> PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _ensure_chroma_dir()
        _chroma_client = PersistentClient(path=CHROMA_DIR)
    return _chroma_client


def _pick_nonempty_collection(client: PersistentClient, preferred: str): # -> Any:
    """
    选择一个可用集合：
      1) 如果 preferred 存在且非空，返回它
      2) 否则扫描所有集合，选择第一个 count()>0 的
      3) 如果所有集合都为空或不存在 → 抛错并给出诊断建议
    """

    def get(name):
        return client.get_collection(name=name, embedding_function=embed_fn)

    # Try preferred
    try:
        # col = client.get_collection(preferred)
        col = get(preferred)
        try:
            if col.count() > 0:
                return col
        except Exception:
            pass
            # some older chroma may not support count() well; fall back to querying 1
            # try:
            #    test = col.query(query_embeddings=[[0.0]], n_results=1)
            #    if test and test.get("ids"):
            #        return col
            #except Exception:
            #    pass
        for c in client.list_collections():
            try:
                col = get(c.name)
                if col.count() > 0:
                    return col
            except Exception:
                continue        
    except Exception:
        # preferred not found
        pass

    # Fallback: list all and choose a non-empty one
    candidates = []
    try:
        candidates = client.list_collections()
    except Exception:
        pass

    for c in candidates:
        try:
            col = client.get_collection(c.name)
            if col.count() > 0:
                return col
        except Exception:
            # try a tiny query to see if it exists
            try:
                test = col.query(query_embeddings=[[0.0]], n_results=1)
                if test and test.get("ids"):
                    return col
            except Exception:
                continue

    # If we are here, no non-empty collections
    names = [c.name for c in candidates] if candidates else []
    raise RuntimeError(
        "未找到可用的 Chroma 集合（所有集合不存在或为空）。\n"
        f"已检测集合: {names}\n"
        "请先运行你的指南向量化/写入脚本，将 ./data/guidelines.parsed.jsonl 写入向量库，"
        f"或检查 CHROMA_PERSIST_DIR={CHROMA_DIR} 是否正确。"
    )


# def get_chroma_collection():
#    global _chroma_collection
#    if _chroma_collection is None:
#        client = _open_client()
#        _chroma_collection = _pick_nonempty_collection(client, COLLECTION_NAME)
#    return _chroma_collection

def get_chroma_collection():
    client = PersistentClient(path=PERSIST_DIR)
    # BIND the embedding function here
    # return client.get_collection(name=COLLECTION, embedding_function=embed_fn)
    return client.get_collection(name=COLLECTION)


# =========================
# Guideline search (Chroma)
# =========================
"""
    语义检索（cosine 距离→相似度 1 - distance）
    返回：[{id, content, meta, score, source}]
"""
def search_guidelines(query: str, k: int = 6, where: Optional[Dict[str, Any]] = None):
    col = get_chroma_collection()
    enc = get_embedder()  # loads BAAI/bge-large-zh-v1.5 on cuda/cpu
    qvec = enc.encode([query], normalize_embeddings=True).tolist()

    # kwargs = dict(query_texts=[query], n_results=k,
    #               include=["documents", "metadatas", "distances"])
    kwargs = dict(
        query_embeddings=qvec,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )   
    if where:
        kwargs["where"] = where  # never pass empty dict

    try:
        res = col.query(**kwargs)
    except ValueError as e:
        # guard against malformed where
        if "Expected where to have exactly one operator" in str(e) and "where" in kwargs:
            kwargs.pop("where", None)
            res = col.query(**kwargs)
        else:
            raise

    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out = []
    for i in range(len(ids)):
        distance = float(dists[i]) if dists else 1.0
        sim = max(0.0, min(1.0, 1.0 - distance))
        out.append({
            "id": ids[i],
            "content": docs[i],
            "meta": metas[i],
            "score": sim,
            "source": "guideline",
        })
    return out

# =========================
# Drug search (SQLite)
# =========================
def _connect_sqlite() -> sqlite3.Connection:
    if not os.path.isfile(SQLITE_PATH):
        raise FileNotFoundError(
            f"SQLite DB not found: {SQLITE_PATH}\n"
            "请确认已运行 caremind/ingest/load_drugs.py 生成 db/drugs.sqlite。"
        )
    con = sqlite3.connect(SQLITE_PATH)
    con.row_factory = sqlite3.Row
    return con


def _has_fts(con: sqlite3.Connection) -> bool:
    cur = con.cursor()
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='drugs_fts'")
    return cur.fetchone() is not None


def trim(text: Optional[str], n: int = 120) -> str:
    if not text:
        return "-"
    t = str(text).replace("\n", " ").strip()
    return (t[:n] + "…") if len(t) > n else t


def search_drugs(query: str, k: int = 6) -> List[Dict[str, Any]]:
    con = _connect_sqlite()
    try:
        if _has_fts(con):
            sql = """
                SELECT d.*, bm25(drugs_fts) AS rank
                FROM drugs_fts JOIN drugs d ON d.rowid = drugs_fts.rowid
                WHERE drugs_fts MATCH ?
                ORDER BY rank ASC
                LIMIT ?
            """
            cur = con.cursor()
            cur.execute(sql, (query, k))
            rows = cur.fetchall()
            out = []
            for r in rows:
                meta = {k: r[k] for k in r.keys()}           # dict copy
                fields = ["name","generic_name","indications","contraindications","interactions"]
                hay = " ".join((meta.get(f) or "") for f in fields)
                hits = sum(1 for f in fields if meta.get(f) and (query in meta.get(f, "")))
                sim = max(0.3, min(1.0, (hits + (1 if query in hay else 0)) / (len(fields) + 1)))
                content = f"{meta.get('name','?')}（{meta.get('generic_name','-')}）\n适应症: {trim(meta.get('indications'))}"
                out.append({
                    "id": f"drug:{r['id']}",
                    "content": content,
                    "meta": meta,
                    "score": sim,
                    "source": "drug"
                })
            return out
        else:
            # LIKE fallback
            cur = con.cursor()
            kw = f"%{query}%"
            sql = """
                SELECT *
                FROM drugs
                WHERE name LIKE ? OR generic_name LIKE ?
                   OR indications LIKE ? OR contraindications LIKE ?
                   OR interactions LIKE ?
                LIMIT ?
            """
            cur.execute(sql, (kw, kw, kw, kw, kw, k))
            rows = cur.fetchall()
            out = []
            for r in rows:
                meta = {k: r[k] for k in r.keys()}           # dict copy
                fields = ["name", "generic_name", "indications", "contraindications", "interactions"]
                hay = " ".join((meta.get(f) or "") for f in fields)
                hits = sum(1 for f in fields if meta.get(f) and (query in (meta.get(f) or "")))
                sim = max(0.3, min(1.0, (hits + (1 if query in hay else 0)) / (len(fields) + 1)))
                content = f"{meta.get('name','?')}（{meta.get('generic_name','-')}）\n适应症: {trim(meta.get('indications'))}"
                out.append({
                    "id": f"drug:{meta.get('id','')}",
                    "content": content,
                    "meta": meta,
                    "score": sim,
                    "source": "drug",
                })
            return out
        
    finally:
        con.close()


# =========================
# Fusion
# =========================
def linear_fusion(guideline_hits: List[Dict[str, Any]],
                  drug_hits: List[Dict[str, Any]],
                  alpha: float = 0.6,
                  topn: int = 8) -> List[Dict[str, Any]]:
    fused: List[Dict[str, Any]] = []
    for h in guideline_hits:
        fused.append({**h, "fused_score": alpha * float(h["score"])})
    for h in drug_hits:
        fused.append({**h, "fused_score": (1.0 - alpha) * float(h["score"])})
    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused[:topn]


def rrf_fusion(guideline_hits: List[Dict[str, Any]],
               drug_hits: List[Dict[str, Any]],
               k: float = 60.0,
               topn: int = 8) -> List[Dict[str, Any]]:
    fused_map: Dict[str, Dict[str, Any]] = {}

    def add_list(hits: List[Dict[str, Any]]):
        for rank, h in enumerate(sorted(hits, key=lambda x: x["score"], reverse=True), start=1):
            key = f"{h['source']}::{h['id']}"
            if key not in fused_map:
                fused_map[key] = {**h, "fused_score": 0.0}
            fused_map[key]["fused_score"] += 1.0 / (k + rank)

    add_list(guideline_hits)
    add_list(drug_hits)

    fused = list(fused_map.values())
    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused[:topn]


# =========================
# High-level API
# =========================
def hybrid_search(query: str,
                  k_guideline: int = 6,
                  k_drug: int = 6,
                  method: str = "linear",
                  alpha: float = 0.6,
                  topn: int = 8) -> Dict[str, Any]:
    g_hits = search_guidelines(query, k=k_guideline)
    d_hits = search_drugs(query, k=k_drug)

    if method == "rrf":
        fused = rrf_fusion(g_hits, d_hits, topn=topn)
    else:
        fused = linear_fusion(g_hits, d_hits, alpha=alpha, topn=topn)

    return {"query": query, "guidelines": g_hits, "drugs": d_hits, "fused": fused}


# =========================
# Diagnostics & CLI
# =========================
def diagnose() -> None:
    print("=== Diagnosing Chroma ===")
    try:
        client = _open_client()
        cols = client.list_collections()
        col = get_chroma_collection()
        print("Using collection:", col.name, "| count:", col.count())    
        if not cols:
            print("No collections found.")
        else:
            for c in cols:
                try:
                    col = client.get_collection(c.name)
                    cnt = col.count()
                except Exception:
                    cnt = -1
                print(f" - {c.name}  count={cnt}")
    except Exception as e:
        print("Chroma error:", repr(e))

    print("\n=== Diagnosing SQLite ===")
    try:
        con = _connect_sqlite()
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='drugs'")
        has = cur.fetchone()[0] > 0
        print(" - drugs table:", "OK" if has else "MISSING")
        print(" - FTS (drugs_fts):", "ENABLED" if _has_fts(con) else "disabled")
        if has:
            cur.execute("SELECT COUNT(*) FROM drugs")
            print(" - drugs rows:", cur.fetchone()[0])
        con.close()
    except Exception as e:
        print("SQLite error:", repr(e))


def _print_hit(h: Dict[str, Any], idx: int):
    head = f"[{idx:02d}] {h['source'].upper()}  score={h.get('score'):.3f}  fused={h.get('fused_score', float('nan')):.3f}"
    print(head)
    if h["source"] == "guideline":
        meta = h.get("meta", {})
        year = meta.get("year", "?")
        title = meta.get("title", meta.get("file", ""))
        print(f"     年份: {year} | 标题: {title}")
    else:
        m = h.get("meta", {})
        print(f"     药名: {m.get('name','?')} | 通用名: {m.get('generic_name','-')}")
    snippet = trim(h.get("content", ""), 160)
    print(f"     摘要: {snippet}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CareMind 混合检索（Chroma + SQLite）")
    parser.add_argument("--q", "--query", dest="query", type=str, help="查询文本")
    parser.add_argument("--k-guide", type=int, default=6, help="指南检索返回数")
    parser.add_argument("--k-drug", type=int, default=6, help="药品检索返回数")
    parser.add_argument("--method", type=str, default="linear", choices=["linear", "rrf"], help="融合方式")
    parser.add_argument("--alpha", type=float, default=0.6, help="linear 模式下指南权重")
    parser.add_argument("--topn", type=int, default=8, help="融合后返回数")
    parser.add_argument("--diagnose", action="store_true", help="打印 Chroma/SQLite 诊断信息并退出")
    args = parser.parse_args()

    print(f"Embedding model: {EMBED_MODEL}")
    print(f"Chroma dir:     {CHROMA_DIR} | preferred collection={COLLECTION}")
    print(f"SQLite path:    {SQLITE_PATH}")

    if args.diagnose:
        diagnose()
        return

    if not args.query:
        parser.error("--q/--query is required unless --diagnose is used")

    print(f"Query:          {args.query}\n")

    result = hybrid_search(
        query=args.query,
        k_guideline=args.k_guide,
        k_drug=args.k_drug,
        method=args.method,
        alpha=args.alpha,
        topn=args.topn
    )

    print("==== 混合结果（Fused） ====")
    for i, h in enumerate(result["fused"], 1):
        _print_hit(h, i)


if __name__ == "__main__":
    main()
