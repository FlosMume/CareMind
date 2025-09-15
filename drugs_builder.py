#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
drugs_builder.py

功能
-----
从 NMPA 数据查询站（在线检索）与 DrugBank API（可选），以及本地 NMPA 说明书文件（PDF/HTML，兜底）
抽取下列字段，构建药品知识表 Excel：

字段：
- 药品名称
- 适应症
- 禁忌症
- 药物相互作用
- 妊娠分级
- 来源（NMPA说明书 / DrugBank / 本地NMPA）

运行示例
--------
python drugs_builder.py \
  --in ~/caremind/data/drug_list.txt \
  --out ~/caremind/data/drugs.xlsx \
  --nmpa-offline-dir ~/caremind/data/nmpa_labels

环境
----
- Python 3.10+
- 依赖: requests, beautifulsoup4, lxml, pandas, openpyxl, tenacity, pdfminer.six, PyPDF2, chardet, python-dotenv

注意
----
- 尊重 NMPA / DrugBank 使用条款与 robots.txt，控制请求频率。
- DrugBank：需要授权 API key（.env）。
- 医学用途免责声明：本工具仅作信息整合，非医疗建议。
"""

from __future__ import annotations
import os
import re
import time
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pdfminer.high_level import extract_text as pdf_extract_text
from PyPDF2 import PdfReader
import chardet
from dotenv import load_dotenv

# ---------------------------
# 常量与正则
# ---------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CareMindDrugBot/1.0; +https://example.org)"
}
SECTION_PATTERNS = [
    # 常见中文说明书节标题（带书名号样式）
    ("适应症", r"(?:【适应症】|适\s*应\s*症|适应证|适应症状)[：:\s]*"),
    ("禁忌症", r"(?:【禁忌】|禁\s*忌|禁忌症)[：:\s]*"),
    ("药物相互作用", r"(?:【药物相互作用】|相互作用|药物-药物相互作用)[：:\s]*"),
    # 妊娠/哺乳期用药可能以多种形式出现
    ("妊娠分级", r"(?:【孕妇及哺乳期用药】|孕妇及哺乳期用药|孕期用药|妊娠用药)[：:\s]*"),
]
# 若文本中找不到明确分段，则用关键词粗抽取（截断到下一个节标题）
NEXT_SECTION_RE = re.compile(r"【[^】]{1,20}】")

# ---------------------------
# 数据结构
# ---------------------------
@dataclass
class DrugRecord:
    name: str
    indications: Optional[str] = None
    contraindications: Optional[str] = None
    interactions: Optional[str] = None
    pregnancy_category: Optional[str] = None
    source: Optional[str] = None

# ---------------------------
# 工具函数
# ---------------------------
def ensure_utf8(text_bytes: bytes) -> str:
    """尽量检测编码并转为 UTF-8 字符串"""
    if not isinstance(text_bytes, (bytes, bytearray)):
        return str(text_bytes)
    det = chardet.detect(text_bytes)
    enc = det.get("encoding") or "utf-8"
    try:
        return text_bytes.decode(enc, errors="ignore")
    except Exception:
        return text_bytes.decode("utf-8", errors="ignore")


def slice_section(text: str, start_pat: str) -> Optional[str]:
    """
    从说明书全文中按节标题粗抽取内容：
    - 定位 start_pat
    - 截断到下一个 '【...】' 标题
    """
    m = re.search(start_pat, text, flags=re.IGNORECASE)
    if not m:
        return None
    start = m.end()
    tail = text[start:]
    nxt = NEXT_SECTION_RE.search(tail)
    if nxt:
        body = tail[:nxt.start()]
    else:
        body = tail
    # 清理多余空白
    body = re.sub(r"\s+", " ", body).strip()
    return body if body else None


def parse_cn_label_text(full_text: str) -> Dict[str, Optional[str]]:
    """从中文说明书全文中解析目标字段"""
    out = {
        "适应症": None,
        "禁忌症": None,
        "药物相互作用": None,
        "妊娠分级": None,
    }
    text = re.sub(r"\s+", " ", full_text)
    for key, pat in SECTION_PATTERNS:
        chunk = slice_section(text, pat)
        if chunk and key != "妊娠分级":
            out[key] = chunk

    # 妊娠分级：大陆说明书通常不提供 A/B/C/D/X；若文本中出现类似“孕妇禁用/慎用”，可粗映射；否则“未标注”
    preg_txt = slice_section(text, SECTION_PATTERNS[-1][1])
    if preg_txt:
        # 简单启发式映射（可按需强化）
        if re.search(r"(禁用|绝对禁用|禁止使用)", preg_txt):
            out["妊娠分级"] = "禁用（未标注分级）"
        elif re.search(r"(慎用|权衡利弊|风险.*收益)", preg_txt):
            out["妊娠分级"] = "慎用（未标注分级）"
        else:
            out["妊娠分级"] = "未标注"
    else:
        out["妊娠分级"] = "未标注"

    return out


# ---------------------------
# NMPA 在线检索（轻量实现）
# ---------------------------
class NMPAClient:
    """
    说明：
    NMPA 数据查询站前端经常有动态脚本与分页接口，且可能变动。
    这里提供一个“轻量检索 + 详情页抓取”的实现思路：
      1) 使用站内搜索（GET 查询字符串）定位“说明书”详情页 URL
      2) 抓取详情页 HTML
      3) 解析说明书全文（若为 PDF/图片，可结合 pdfminer/PyPDF2 或 OCR 自行扩展）
    如遇到前端结构变化、反爬增强、需要 POST token 等情况，请根据实际页面更新 `search_endpoint` 与 CSS 选择器。
    """

    def __init__(self, rate_sec: float = 1.2):
        self.sess = requests.Session()
        self.rate_sec = rate_sec
        self.base = "https://www.nmpa.gov.cn/datasearch/"

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((requests.RequestException,))
    )
    def _get(self, url: str, params: Optional[dict] = None) -> requests.Response:
        resp = self.sess.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code != 200:
            raise requests.RequestException(f"HTTP {resp.status_code}")
        return resp

    def search_label_urls(self, drug_name: str) -> List[str]:
        """
        粗检索：尝试在站内搜索“药名 + 说明书”，返回候选详情页 URL 列表。
        注意：此处接口/参数是“占位策略”，可能需根据当前网站结构调整。
        """
        # 占位（常见为 /home-index.html + 前端异步接口）；这里尝试一个简化的页面搜索回退方案。
        # 方案：搜索站内关键字页，然后从搜索结果中抓取包含 “说明书” 的链接
        search_url = self.base + "home-index.html"
        # 实际通常需要调用站内 search API，这里作为通用回退：请求主页→抓涨落内容→很可能拿不到
        # 因此在实际项目中建议针对当前 NMPA 站点调试具体接口（如 /search?keyword=xxx）。
        try:
            _ = self._get(search_url)
            time.sleep(self.rate_sec)
        except Exception:
            return []

        # 由于无公开稳定 search API，这里直接返回空，让调用方走离线/DrugBank 兜底或自行定制。
        return []

    def fetch_label_text(self, url: str) -> Optional[str]:
        """抓取说明书详情页全文文本（HTML → 文本）。"""
        try:
            r = self._get(url)
            time.sleep(self.rate_sec)
            html = ensure_utf8(r.content)
            soup = BeautifulSoup(html, "lxml")

            # 通用提取：找正文容器
            # 说明书常见在 <div class="article"> 或者 id="article", 根据实际页面结构调整
            container = soup.find("div", class_="article") or soup.find(id="article") or soup
            text = container.get_text(separator="\n")
            text = re.sub(r"\n+", "\n", text)
            return text.strip()
        except Exception:
            return None


# ---------------------------
# DrugBank API（可选）
# ---------------------------
class DrugBankClient:
    """
    说明：
    - 需在 .env 中配置 DRUGBANK_API_KEY 与 DRUGBANK_BASE
    - 具体 API 端点与字段名会因你的许可证版本不同而异。此处演示一种常见 REST 风格：
        GET /v1/drugs?name=<query>
        GET /v1/drugs/<id>
    - 若你的接口是 GraphQL，请自行替换实现。
    """

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("DRUGBANK_API_KEY")
        self.base = os.getenv("DRUGBANK_BASE", "https://api.drugbank.com/v1")
        self.sess = requests.Session()
        if self.api_key:
            self.sess.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": HEADERS["User-Agent"]
            })

    def available(self) -> bool:
        return bool(self.api_key)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((requests.RequestException,))
    )
    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = self.base.rstrip("/") + "/" + path.lstrip("/")
        resp = self.sess.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            raise requests.RequestException(f"DrugBank HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def query_by_name(self, name: str) -> Optional[dict]:
        """
        名称搜索 → 取第一个匹配项 → 拉详情。
        实际字段名与路径需按你的 API 文档调整。
        """
        try:
            j = self._get("drugs", params={"name": name})
            if not j:
                return None
            # 假设返回列表
            first = j[0] if isinstance(j, list) else j
            drug_id = first.get("id") or first.get("drugbank_id")
            if not drug_id:
                return first  # 已含详情
            detail = self._get(f"drugs/{drug_id}")
            return detail
        except Exception:
            return None

    @staticmethod
    def pick_fields(detail: dict) -> Dict[str, Optional[str]]:
        """
        从 DrugBank 详情中抽取近似字段。
        注意：具体路径/键名需依你的 API 实际响应调整。
        """
        def get_nested(d, *keys, default=None):
            cur = d
            for k in keys:
                if cur is None:
                    return default
                cur = cur.get(k)
            return cur if cur is not None else default

        indications = get_nested(detail, "indication", default=None) or get_nested(detail, "indications", default=None)
        contraindications = get_nested(detail, "contraindications", default=None)
        interactions = None

        # interactions 可能在 "drug_interactions" 数组中
        intr_list = get_nested(detail, "drug_interactions", default=None)
        if isinstance(intr_list, list) and intr_list:
            # 合并简要描述
            snippets = []
            for it in intr_list[:30]:  # 截前 N 条以免过长
                desc = it.get("description") or it.get("text")
                partner = it.get("name") or it.get("drug") or ""
                if desc:
                    snippets.append(f"{partner}: {desc}")
            if snippets:
                interactions = "；".join(snippets)

        # 妊娠分级（若提供）。若无，留空，后续由 NMPA/启发式补全
        preg = get_nested(detail, "pregnancy_category", default=None) or \
               get_nested(detail, "fda_pregnancy_category", default=None)

        return {
            "适应症": indications,
            "禁忌症": contraindications,
            "药物相互作用": interactions,
            "妊娠分级": preg
        }


# ---------------------------
# 离线说明书解析（PDF/HTML）
# ---------------------------
def read_pdf_text(path: str) -> str:
    try:
        # 优先 pdfminer（版式更稳定）
        txt = pdf_extract_text(path) or ""
        if not txt:
            # 回退 PyPDF2
            reader = PdfReader(path)
            buf = []
            for page in reader.pages:
                buf.append(page.extract_text() or "")
            txt = "\n".join(buf)
        return txt
    except Exception:
        return ""


def read_html_text(path: str) -> str:
    try:
        with open(path, "rb") as f:
            raw = f.read()
        html = ensure_utf8(raw)
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n+", "\n", text)
        return text.strip()
    except Exception:
        return ""


def scan_offline_label(drug_name: str, folder: Optional[str]) -> Optional[str]:
    if not folder or not os.path.isdir(folder):
        return None
    # 以药名包含匹配
    cand = []
    for fn in os.listdir(folder):
        if re.search(re.escape(drug_name), fn, flags=re.IGNORECASE):
            cand.append(os.path.join(folder, fn))
    # 优先 HTML，其次 PDF
    cand_sorted = sorted(cand, key=lambda p: (0 if p.lower().endswith((".html", ".htm")) else 1, p))
    for p in cand_sorted:
        if p.lower().endswith((".html", ".htm")):
            txt = read_html_text(p)
        elif p.lower().endswith(".pdf"):
            txt = read_pdf_text(p)
        else:
            continue
        if txt and len(txt) > 50:
            return txt
    return None


# ---------------------------
# 主流程
# ---------------------------
def build_records(
    names: List[str],
    use_nmpa_online: bool = True,
    nmpa_offline_dir: Optional[str] = None,
    use_drugbank: bool = True
) -> List[DrugRecord]:

    nmpa = NMPAClient()
    db = DrugBankClient()
    if use_drugbank and not db.available():
        use_drugbank = False

    records: List[DrugRecord] = []

    for name in names:
        rec = DrugRecord(name=name)
        # 优先：NMPA 在线（若可定制到有效搜索接口）
        if use_nmpa_online:
            # 这里的 search 是占位，默认返回空（需你根据当前站点接口定制）
            urls = nmpa.search_label_urls(name)
            text = None
            for u in urls:
                text = nmpa.fetch_label_text(u)
                if text:
                    break
            if text:
                parsed = parse_cn_label_text(text)
                rec.indications = parsed["适应症"]
                rec.contraindications = parsed["禁忌症"]
                rec.interactions = parsed["药物相互作用"]
                rec.pregnancy_category = parsed["妊娠分级"]
                rec.source = "NMPA说明书（在线）"

        # 其次：DrugBank（若可用且仍有缺口）
        if use_drugbank and not all([rec.indications, rec.contraindications, rec.interactions, rec.pregnancy_category]):
            detail = db.query_by_name(name)
            if detail:
                picked = db.pick_fields(detail)
                # 仅填补缺口（保留已从 NMPA 得到的内容）
                rec.indications = rec.indications or picked["适应症"]
                rec.contraindications = rec.contraindications or picked["禁忌症"]
                rec.interactions = rec.interactions or picked["药物相互作用"]
                rec.pregnancy_category = rec.pregnancy_category or picked["妊娠分级"]
                rec.source = (rec.source + " + DrugBank") if rec.source else "DrugBank"

        # 兜底：离线 NMPA 文件夹
        if not any([rec.indications, rec.contraindications, rec.interactions, rec.pregnancy_category]):
            text = scan_offline_label(name, nmpa_offline_dir)
            if text:
                parsed = parse_cn_label_text(text)
                rec.indications = parsed["适应症"]
                rec.contraindications = parsed["禁忌症"]
                rec.interactions = parsed["药物相互作用"]
                rec.pregnancy_category = parsed["妊娠分级"]
                rec.source = "NMPA说明书（离线）"

        # 若仍缺字段，填上“未标注”
        if not rec.indications: rec.indications = "未标注"
        if not rec.contraindications: rec.contraindications = "未标注"
        if not rec.interactions: rec.interactions = "未标注"
        if not rec.pregnancy_category: rec.pregnancy_category = "未标注"
        if not rec.source: rec.source = "未获取（请补充源）"

        records.append(rec)

    return records


def save_to_excel(records: List[DrugRecord], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for r in records:
        rows.append({
            "药品名称": r.name,
            "适应症": r.indications,
            "禁忌症": r.contraindications,
            "药物相互作用": r.interactions,
            "妊娠分级": r.pregnancy_category,
            "来源": r.source
        })
    df = pd.DataFrame(rows, columns=["药品名称","适应症","禁忌症","药物相互作用","妊娠分级","来源"])
    df.to_excel(out_path, index=False)
    print(f"✅ 已写出 {len(df)} 条记录 → {out_path}")


def load_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_file", required=True, help="药品清单（每行一个名称）")
    ap.add_argument("--out", dest="out_file", required=True, help="输出 Excel 路径，例如 ~/caremind/data/drugs.xlsx")
    ap.add_argument("--nmpa-offline-dir", dest="nmpa_offline_dir", default=None, help="离线 NMPA 说明书目录（可选）")
    ap.add_argument("--no-nmpa-online", action="store_true", help="禁用 NMPA 在线检索")
    ap.add_argument("--no-drugbank", action="store_true", help="禁用 DrugBank API")
    args = ap.parse_args()

    names = load_names(os.path.expanduser(args.in_file))
    if not names:
        raise SystemExit("❌ 药品清单为空")

    use_nmpa_online = not args.no_nmpa_online
    use_drugbank = not args.no_drugbank

    records = build_records(
        names,
        use_nmpa_online=use_nmpa_online,
        nmpa_offline_dir=os.path.expanduser(args.nmpa_offline_dir) if args.nmpa_offline_dir else None,
        use_drugbank=use_drugbank
    )
    save_to_excel(records, os.path.expanduser(args.out_file))


if __name__ == "__main__":
    main()
