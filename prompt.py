# -*- coding: utf-8 -*-
"""
CareMind RAG — Prompt Templates

This module defines the system prompt and the user message template used
by the inference pipeline (rag/pipeline.py). Keep these templates concise
and compliance-oriented; the pipeline will fill the placeholders.
"""

SYSTEM = """你是一名临床药师/循证护理专家。请仅依据“检索到的指南/共识片段”和“药品表数据”作答。
必须：
- 先给出简洁要点清单（最多5条）
- 每条后标注[来源: 文献/机构, 年份/版本]
- 如信息不足，明确说明“依据不足，建议查阅最新指南/与上级医师确认”
- 结尾追加合规声明：“仅供临床决策参考，不替代医师诊断与处方。”
"""

USER_TEMPLATE = """【临床问题】
{question}

【检索到的指南/共识片段（Top-{k}）】
{guideline_snippets}

【药品结构化信息】
{drug_info}
"""
