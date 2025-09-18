# -*- coding: utf-8 -*-
"""
CareMind · MVP CDSS · 前端 (Streamlit)
依赖：
  - streamlit>=1.32
项目约定：
  - 后端推理入口：rag.pipeline.answer(question: str, drug_name: Optional[str], k: int) -> dict
返回字典示例：
  {
    "output": "...模型生成建议（带引用与合规说明）...",
    "guideline_hits": [
        {"content": "...片段...", "meta": {"source": "...", "year": 2024, "title": "...", "id": "..."}},
        ...
    ],
    "drug": {"药品名称": "...", "适应症": "...", ...}  # 若有，则为结构化信息
  }
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import streamlit as st
from pipeline import answer  # 你已实现的后端入口

# ---------------------------
# 基础页面配置 & 轻量样式
# ---------------------------
st.set_page_config(
    page_title="CareMind · MVP CDSS",
    layout="wide",
    page_icon="💊",
)

# 轻量 CSS：更紧凑的卡片风格 & 引用徽章
st.markdown(
    """
    <style>
    .cm-badge {
        display:inline-block;
        padding:2px 8px;
        border-radius:12px;
        font-size:12px;
        background:#eef2ff;
        border:1px solid #c7d2fe;
        margin-right:6px;
        white-space:nowrap;
    }
    .cm-chip {
        display:inline-block;
        padding:2px 8px;
        border-radius:8px;
        font-size:12px;
        background:#f1f5f9;
        border:1px solid #e2e8f0;
        margin:0 6px 6px 0;
    }
    .cm-card {
        border:1px solid #e5e7eb;
        background:#ffffff;
        border-radius:12px;
        padding:12px 14px;
        margin-bottom:10px;
    }
    .cm-muted {
        color:#64748b;
        font-size:13px;
    }
    .cm-output {
        line-height:1.6;
        font-size:16px;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Sidebar：检索与显示设置
# ---------------------------
with st.sidebar:
    st.header("⚙️ 设置")
    k = st.slider("检索片段数（Top-K）", min_value=2, max_value=8, value=4, step=1)
    show_meta = st.toggle("显示片段元数据行", value=True)
    expand_hits = st.toggle("展开所有片段", value=False)
    st.divider()
    st.markdown(
        "#### 🧪 使用建议\n"
        "- 临床问题尽量**具体**，可包含人群限定、并发症与药名\n"
        "- 可选输入药品名，用于**结构化比对**（如：阿司匹林）\n"
    )
    st.divider()
    st.caption("版本：MVP • 本工具仅供临床决策参考，不替代医师诊断与处方。")

# ---------------------------
# Session State：简单历史记录
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []

# ---------------------------
# 页面主区
# ---------------------------
st.title("CareMind · 临床决策支持（MVP）")

# 两列布局：左侧输入/结果，右侧为命中与药品信息
col_left, col_right = st.columns([1.3, 1.0])

with col_left:
    # —— 输入表单
    with st.form("cm_query"):
        q = st.text_input(
            "输入临床问题",
            placeholder="例如：慢性肾病（CKD）合并高血压患者使用 ACEI/ARB 有何监测要点？",
        )
        drug = st.text_input(
            "（可选）指定药品名，用于结构化比对",
            placeholder="例如：阿司匹林 / 氨氯地平 / 依那普利...",
        )
        submitted = st.form_submit_button("生成建议", use_container_width=True)

    # —— 调用后端
    res: Optional[Dict[str, Any]] = None
    if submitted:
        if not q or not q.strip():
            st.warning("请输入临床问题。")
        else:
            with st.spinner("检索与生成中…"):
                try:
                    res = answer(q.strip(), drug_name=(drug.strip() or None), k=int(k))
                except Exception as e:
                    st.error("后端推理出现异常，请查看后端日志或稍后重试。")
                    st.exception(e)
                    res = None

    # —— 渲染输出
    if res:
        # 存历史
        st.session_state.history.insert(
            0,
            {
                "q": q.strip(),
                "drug": (drug.strip() or None),
                "k": k,
                "res": res,
            },
        )

        st.subheader("🧭 建议（含引用与合规声明）")
        output_text = res.get("output") or "（无生成内容）"
        st.markdown(f"<div class='cm-output'>{output_text}</div>", unsafe_allow_html=True)

        # 复制 & 下载
        col_btn1, col_btn2, _ = st.columns([0.25, 0.25, 0.5])
        with col_btn1:
            st.code(output_text, language="markdown")
        with col_btn2:
            download_payload = json.dumps(res, ensure_ascii=False, indent=2)
            st.download_button(
                "下载本次结果（JSON）",
                data=download_payload.encode("utf-8"),
                file_name="caremind_response.json",
                mime="application/json",
                use_container_width=True,
            )

        st.caption("⚠️ 本工具仅供临床决策参考，不替代医师诊断与处方。")

with col_right:
    # —— Top-K 命中展示
    if res:
        hits: List[Dict[str, Any]] = res.get("guideline_hits") or []
        st.subheader(f"📚 检索片段（Top-{k}）")

        if not hits:
            st.info("未检索到相关指南/共识片段。请尝试调整问题或增大 Top-K。")
        else:
            # 顶部来源统计 chips
            sources = {}
            for h in hits:
                meta = (h.get("meta") or {})
                src = str(meta.get("source") or "未知来源").strip()
                sources[src] = sources.get(src, 0) + 1
            st.markdown(
                " ".join([f"<span class='cm-chip'>{s} × {n}</span>" for s, n in sources.items()]),
                unsafe_allow_html=True,
            )

            for i, h in enumerate(hits, 1):
                meta = h.get("meta") or {}
                source = str(meta.get("source") or "")
                year = str(meta.get("year") or "")
                title = str(meta.get("title") or "")
                doc_id = str(meta.get("id") or "")

                label = f"#{i} · {title}" if title else f"#{i} · 无标题片段"
                with st.expander(label, expanded=expand_hits):
                    if show_meta:
                        st.markdown(
                            f"<div class='cm-muted'>"
                            f"<span class='cm-badge'>来源：{source or '未知'}</span>"
                            f"<span class='cm-badge'>年份：{year or '—'}</span>"
                            f"<span class='cm-badge'>ID：{doc_id or '—'}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(h.get("content") or "（空片段）")

        # —— 结构化药品信息
        drug_obj = res.get("drug")
        st.divider()
        st.subheader("💊 药品结构化信息（SQLite）")
        if drug_obj:
            st.json(drug_obj, expanded=False)
        else:
            st.caption("未提供或未检索到对应药品的结构化信息。")

# ---------------------------
# 历史记录（本会话）
# ---------------------------
with st.expander("🗂️ 本会话历史（仅本地会话内可见）", expanded=False):
    if not st.session_state.history:
        st.caption("暂无历史记录。")
    else:
        for idx, item in enumerate(st.session_state.history, 1):
            st.markdown(
                f"**{idx}.** Q: `{item['q']}` | 药品: `{item['drug'] or '—'}` | Top-K: `{item['k']}`"
            )

# ---------------------------
# 页脚提示
# ---------------------------
st.caption("© CareMind · MVP CDSS | 本工具仅供临床决策参考，不替代医师诊断与处方。")
