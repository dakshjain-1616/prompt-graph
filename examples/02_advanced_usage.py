#!/usr/bin/env python3
"""
02_advanced_usage.py — Advanced Prompt-Graph features.

Demonstrates:
  - Parsing a real JSONL log file
  - Multiple layout algorithms
  - Re-ranker rank badges (top-3 nodes)
  - Cluster detection and colour-coding
  - Exporting a JSON stats report alongside the SVG
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import tempfile
from pathlib import Path
from prompt_graph_visuali import (
    parse_log_file,
    build_graph,
    compute_layout,
    compute_stats,
    export_report,
    render_svg,
)

# ── Build a synthetic JSONL log in memory ────────────────────────────────────

events = [
    {"event": "query", "query": "How does cross-encoder re-ranking improve RAG?"},
]
chunk_data = [
    ("c001", 0.92, "Cross-encoders jointly encode query and passage.", "papers/cross_encoder.pdf"),
    ("c002", 0.85, "Bi-encoders are faster but less accurate than cross-encoders.", "docs/bi_encoder.md"),
    ("c003", 0.78, "Re-ranking reorders an initial candidate set by relevance.", "wiki/reranking.md"),
    ("c004", 0.61, "BM25 provides a strong sparse retrieval baseline.", "docs/bm25.md"),
    ("c005", 0.54, "Dense retrieval uses vector similarity in embedding space.", "docs/dense.md"),
    ("c006", 0.43, "Hybrid retrieval combines sparse and dense scores via RRF.", "blog/hybrid.md"),
]
for rank, (cid, score, content, source) in enumerate(chunk_data, 1):
    events.append({"event": "retrieve", "chunk_id": cid, "score": score,
                   "content": content, "source": source})
    if rank <= 3:
        events.append({"event": "rerank", "chunk_id": cid, "rank": rank})

# Chunk-to-chunk connections
events += [
    {"event": "connect", "from": "c001", "to": "c002", "weight": 0.80},
    {"event": "connect", "from": "c002", "to": "c003", "weight": 0.65},
    {"event": "connect", "from": "c004", "to": "c005", "weight": 0.72},
    {"event": "connect", "from": "c005", "to": "c006", "weight": 0.58},
]

with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
    for ev in events:
        f.write(json.dumps(ev) + "\n")
    log_path = f.name

# ── Parse → Build → Layout → Render ─────────────────────────────────────────

data = parse_log_file(log_path)
G = build_graph(data, cluster=True)          # cluster=True → community colour-coding
pos = compute_layout(G, width=1400, height=900)

Path("outputs").mkdir(exist_ok=True)

svg_path = "outputs/advanced_clustered.svg"
render_svg(G, pos, svg_path, width=1400, height=900,
           title="Advanced Usage — Clustered RAG Graph")
print(f"SVG  → {svg_path}")

# Export JSON stats report
report_path = "outputs/advanced_clustered.json"
export_report(G, data, report_path)
stats = json.loads(Path(report_path).read_text())
print(f"Report → {report_path}")
print(f"  chunks={stats['chunk_count']}  edges={stats['edge_count']}"
      f"  avg_score={stats['scores']['mean']:.3f}")

# Try a circular layout too
pos_circ = compute_layout(G, width=1000, height=1000)
render_svg(G, pos_circ, "outputs/advanced_circular.svg",
           width=1000, height=1000, title="Advanced Usage — Circular Layout")
print("SVG  → outputs/advanced_circular.svg")

os.unlink(log_path)
