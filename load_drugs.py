#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CareMind Drug Excel → SQLite Ingestor
=====================================

Reads an Excel workbook (default: caremind/data/drugs.xlsx), normalizes column names
(including Chinese → English mappings), and upserts rows into a SQLite database
(default: caremind/db/drugs.sqlite).

Key features
------------
- Auto-detects common column names in English or Chinese:
  - 药品名称 / 药物名称 / 通用名 / drug / drug_name → drug_name (required)
  - 适应症 → indications
  - 禁忌症 → contraindications
  - 药物相互作用 / 相互作用 → interactions
  - 妊娠分级 / 妊娠分类 / 妊娠用药分级 → pregnancy_category
  - 来源 / 信息来源 / 引用 → source
- Creates the SQLite DB and a `drugs` table if missing.
- Idempotent upsert on `drug_name` (unique), updates only non-empty fields.
- Optional FTS5 full‑text search virtual table and triggers (enable with --with-fts).
- Works both as a CLI script and inside Jupyter.

Usage (CLI)
-----------
# From repo root (WSL/Windows paths both supported)
python caremind/ingest/load_drugs.py \
  --in   data/drugs.xlsx \
  --out  db/drugs.sqlite \
  --sheet 0 \
  --with-fts

Usage (Jupyter)
---------------
from caremind.ingest.load_drugs import main, IngestConfig
main(IngestConfig(in_path="caremind/data/drugs.xlsx",
                  out_path="caremind/db/drugs.sqlite",
                  sheet=0,
                  with_fts=True))

Dependencies
------------
- Python 3.10.18
- pandas
- openpyxl  (Excel reader)
- (optional) sqlite3 is in the standard library

Schema
------
drugs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  drug_name TEXT UNIQUE NOT NULL,
  indications TEXT,
  contraindications TEXT,
  interactions TEXT,
  pregnancy_category TEXT,
  source TEXT,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

If --with-fts is set, also creates drugs_fts (FTS5) and sync triggers.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


# -----------------------------
# Configuration
# -----------------------------
@dataclass

class IngestConfig:
    """Configuration for the ingest job."""
    in_path: str = "data/drugs.xlsx"        # Input Excel file
    out_path: str = "db/drugs.sqlite"       # Output SQLite DB file
    sheet: Optional[str | int] = 0          # Excel sheet (default: first)
    with_fts: bool = False                  # Enable FTS5 indexing
    if_exists: str = "append"               # How to handle existing DB? append via upsert
    fail_on_missing_drug_name: bool = True  # Stop if no drug_name column


# Canonical column names we care about
CANONICAL = {
    "drug_name",
    "indications",
    "contraindications",
    "interactions",
    "pregnancy_category",
    "source",
}

# Heuristic mappings (lowercased + stripped + snake_cased keys)
COLUMN_MAP: Dict[str, str] = {
    # drug_name
    "药品名称": "drug_name",
    "药物名称": "drug_name",
    "通用名": "drug_name",
    "通用名称": "drug_name",
    "商品名": "drug_name",
    "英文名": "drug_name",
    "drug": "drug_name",
    "drugname": "drug_name",
    "drug_name": "drug_name",
    "name": "drug_name",
    "品名": "drug_name",
    # indications
    "适应症": "indications",
    "indication": "indications",
    "indications": "indications",
    # contraindications
    "禁忌症": "contraindications",
    "禁忌": "contraindications",
    "contraindication": "contraindications",
    "contraindications": "contraindications",
    # interactions
    "药物相互作用": "interactions",
    "相互作用": "interactions",
    "interactions": "interactions",
    "drug_interactions": "interactions",
    # pregnancy_category
    "妊娠分级": "pregnancy_category",
    "妊娠分类": "pregnancy_category",
    "妊娠用药分级": "pregnancy_category",
    "pregnancy": "pregnancy_category",
    "pregnancy_category": "pregnancy_category",
    "pregnancy category": "pregnancy_category",
    # source
    "来源": "source",
    "信息来源": "source",
    "引用": "source",
    "参考文献": "source",
    "source": "source",
    "ref": "source",
    "reference": "source",
}


# -----------------------------
# Helpers
# -----------------------------
def snake(s: str) -> str:
    s = (s or "").strip()
    # Replace spaces and weird separators
    for ch in (" ", "-", "（", "）", "(", ")", "：", ":", "/", "\\", "。", "，", ","):
        s = s.replace(ch, "_")
    s = "_".join(filter(None, s.split("_")))
    return s.lower()


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Map dataframe columns into canonical names; return df and a mapping used."""
    mapping_used: Dict[str, str] = {}
    new_cols = []
    for c in df.columns:
        key = snake(str(c))
        mapped = COLUMN_MAP.get(key, key)  # default to normalized key
        new_cols.append(mapped)
        mapping_used[str(c)] = mapped
    df = df.copy()
    df.columns = new_cols
    return df, mapping_used


def ensure_dirs(db_path: Path) -> None:
    """Ensure the directory for the DB exists."""    
    db_path.parent.mkdir(parents=True, exist_ok=True)


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with safe pragmas."""
    con = sqlite3.connect(db_path)
    # Enable foreign keys and WAL for better concurrency
    con.execute("PRAGMA foreign_keys = ON;")
    con.execute("PRAGMA journal_mode = WAL;")
    return con

# -----------------------------
# Schema creation
# -----------------------------
def create_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS drugs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_name TEXT UNIQUE NOT NULL,
            indications TEXT,
            contraindications TEXT,
            interactions TEXT,
            pregnancy_category TEXT,
            source TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


def create_fts(con: sqlite3.Connection) -> None:
    # Requires SQLite built with FTS5 (standard in modern SQLite)
    con.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS drugs_fts USING fts5(
            drug_name, indications, contraindications, interactions, pregnancy_category, source,
            content='drugs', content_rowid='id'
        );

        -- Initial sync
        INSERT INTO drugs_fts(rowid, drug_name, indications, contraindications, interactions, pregnancy_category, source)
        SELECT id, drug_name, indications, contraindications, interactions, pregnancy_category, source FROM drugs
        WHERE NOT EXISTS (SELECT 1 FROM drugs_fts LIMIT 1);

        -- Triggers to keep FTS in sync
        CREATE TRIGGER IF NOT EXISTS drugs_ai AFTER INSERT ON drugs BEGIN
            INSERT INTO drugs_fts(rowid, drug_name, indications, contraindications, interactions, pregnancy_category, source)
            VALUES (new.id, new.drug_name, new.indications, new.contraindications, new.interactions, new.pregnancy_category, new.source);
        END;
        CREATE TRIGGER IF NOT EXISTS drugs_ad AFTER DELETE ON drugs BEGIN
            INSERT INTO drugs_fts(drugs_fts, rowid, drug_name, indications, contraindications, interactions, pregnancy_category, source)
            VALUES ('delete', old.id, old.drug_name, old.indications, old.contraindications, old.interactions, old.pregnancy_category, old.source);
        END;
        CREATE TRIGGER IF NOT EXISTS drugs_au AFTER UPDATE ON drugs BEGIN
            INSERT INTO drugs_fts(drugs_fts, rowid, drug_name, indications, contraindications, interactions, pregnancy_category, source)
            VALUES ('delete', old.id, old.drug_name, old.indications, old.contraindications, old.interactions, old.pregnancy_category, old.source);
            INSERT INTO drugs_fts(rowid, drug_name, indications, contraindications, interactions, pregnancy_category, source)
            VALUES (new.id, new.drug_name, new.indications, new.contraindications, new.interactions, new.pregnancy_category, new.source);
        END;
        """
    )

# -----------------------------
# Insertion logic
# -----------------------------
def upsert_row(con: sqlite3.Connection, row: dict) -> None:
    """Upsert by drug_name. Only overwrite columns where incoming value is non-empty."""
    # Build dynamic set clause that preserves existing non-empty values when new is empty
    columns = ["indications", "contraindications", "interactions", "pregnancy_category", "source"]
    params = {
        "drug_name": row.get("drug_name", "").strip(),
        **{c: (None if pd.isna(row.get(c)) else str(row.get(c)).strip()) for c in columns},
    }
    if not params["drug_name"]:
        raise ValueError("Missing required 'drug_name' for a row")

    con.execute(
        """
        INSERT INTO drugs (drug_name, indications, contraindications, interactions, pregnancy_category, source)
        VALUES (:drug_name, :indications, :contraindications, :interactions, :pregnancy_category, :source)
        ON CONFLICT(drug_name) DO UPDATE SET
            indications = COALESCE(excluded.indications, drugs.indications),
            contraindications = COALESCE(excluded.contraindications, drugs.contraindications),
            interactions = COALESCE(excluded.interactions, drugs.interactions),
            pregnancy_category = COALESCE(excluded.pregnancy_category, drugs.pregnancy_category),
            source = COALESCE(excluded.source, drugs.source),
            updated_at = CURRENT_TIMESTAMP;
        """,
        params,
    )

# -----------------------------
# Main ingestion pipeline
# -----------------------------
def ingest_excel(cfg: IngestConfig) -> Tuple[int, int]:
    """Main workflow: read Excel, normalize, insert into SQLite."""
    in_path = Path(cfg.in_path)
    out_path = Path(cfg.out_path)
    if not in_path.exists():
        raise SystemExit(f"Excel not found: {in_path}")

    ensure_dirs(out_path)
    logging.info("Reading Excel: %s (sheet=%s)", in_path, str(cfg.sheet))

    # Read using pandas; allow either sheet index or name
    df = pd.read_excel(in_path, sheet_name=cfg.sheet, engine="openpyxl")
    logging.info("Loaded %d rows, %d columns from sheet %s", len(df), len(df.columns), str(cfg.sheet))

    # Normalize columns
    df, mapping = normalize_columns(df)
    logging.info("Column mapping: %s", mapping)

    # Check required, ensure drug_name exists
    if "drug_name" not in df.columns:
        msg = "No 'drug_name' column found after normalization. Please include a column like 药品名称/通用名/drug."
        if cfg.fail_on_missing_drug_name:
            raise SystemExit(msg)
        else:
            logging.warning(msg + " Proceeding with empty rows filtered.")

    # Keep only canonical columns (others are ignored safely), fill missing canonical columns with None
    for col in CANONICAL:
        if col not in df.columns:
            df[col] = None

    # Filter empty drug_name rows: Remove rows with empty drug_name
    df["drug_name"] = df["drug_name"].astype(str).str.strip()
    df = df[df["drug_name"].astype(bool)].copy()

    # Ingest rows, Connect DB and create schema
    con = connect(out_path)
    create_schema(con)
    if cfg.with_fts:
        create_fts(con)

    # Upsert rows
    inserted = 0
    for _, r in df.iterrows():
        upsert_row(con, r.to_dict())
        inserted += 1
    con.commit()

    # Basic indices, Add index for faster lookup
    con.execute("CREATE INDEX IF NOT EXISTS idx_drugs_drug_name ON drugs(drug_name);")
    con.commit()
    con.close()

    logging.info("Upserted %d rows into %s", inserted, out_path)
    return len(df), inserted

# -----------------------------
# CLI parsing
# -----------------------------
def parse_args(argv=None) -> IngestConfig:
    p = argparse.ArgumentParser(description="Ingest drugs Excel → SQLite for CareMind.")
    p.add_argument("--in", dest="in_path", default="caremind/data/drugs.xlsx",
                   help="Path to input Excel file (default: caremind/data/drugs.xlsx)")
    p.add_argument("--out", dest="out_path", default="caremind/db/drugs.sqlite",
                   help="Path to output SQLite database (default: caremind/db/drugs.sqlite)")
    p.add_argument("--sheet", dest="sheet", default="0",
                   help="Sheet index or name (default: 0)")
    p.add_argument("--with-fts", dest="with_fts", action="store_true",
                   help="Create and sync an FTS5 full-text search table (drugs_fts).")
    p.add_argument("--no-fail-on-missing-name", dest="fail_on_missing_drug_name",
                   action="store_false", help="Do not abort if 'drug_name' is missing.")

    args = p.parse_args(argv)
    # try to cast sheet to int if it's numeric
    sheet: str | int
    try:
        sheet = int(args.sheet)
    except (TypeError, ValueError):
        sheet = args.sheet

    return IngestConfig(
        in_path=args.in_path,
        out_path=args.out_path,
        sheet=sheet,
        with_fts=args.with_fts,
        fail_on_missing_drug_name=args.fail_on_missing_drug_name,
    )

# -----------------------------
# Entry point
# -----------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(cfg: Optional[IngestConfig] = None) -> None:
    setup_logging()
    if cfg is None:
        cfg = parse_args()

    try:
        total, upserted = ingest_excel(cfg)
        logging.info("Done. Processed %d rows; upserted %d rows.", total, upserted)
    except Exception as e:
        logging.exception("Ingest failed: %s", e)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
