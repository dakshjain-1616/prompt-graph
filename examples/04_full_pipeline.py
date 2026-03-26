#!/usr/bin/env python3
"""
04_full_pipeline.py — End-to-end Prompt-Graph workflow.

Simulates a realistic two-run RAG experiment:

  Run A  — baseline retrieval (8 chunks, spring layout)
  Run B  — improved retrieval (same query, different scores/chunks)

Pipeline steps:
  1. Write two synthetic JSONL log files (run_a.log, run_b.log)
  2. Visualize each run individually with stats export
  3. Diff the two runs → highlight added / removed / score-changed chunks
  4. Print a side-by-side stats comparison to the console
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
    diff_logs,
    visualize,
    visualize_diff,
)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ── Step 1: Build two JSONL log files ────────────────────────────────────────

QUERY = "What chunking strategies work best for long documents?"

RUN_A_CHUNKS = [
    ("c001", 0.91, "Fixed-size chunking splits text into N-token windows.", "docs/chunking.md"),
    ("c002", 0.83, "Sentence-boundary chunking preserves semantic units.", "wiki/nlp.md"),
    ("c003", 0.74, "Overlapping windows reduce context loss at boundaries.", "blog/chunking.md"),
    ("c004", 0.62, "Recursive character splitting adapts to nested structure.", "docs/langchain.md"),
    ("c005", 0.55, "Semantic chunking groups by embedding similarity.", "papers/chunking.pdf"),
    ("c006", 0.48, "Document-level chunking is simple but loses fine detail.", "docs/basics.md"),
    ("c007", 0.41, "Token-budget chunking respects LLM context windows.", "docs/llm.md"),
    ("c008", 0.33, "Paragraph splitting uses whitespace as natural dividers.", "wiki/text.md"),
]

RUN_B_CHUNKS = [
    # c001 improved, c002 same, c003 removed, c004 improved, c005 dropped,
    # c006 removed, new c009 and c010 added
    ("c001", 0.95, "Fixed-size chunking splits text into N-token windows.", "docs/chunking.md"),
    ("c002", 0.83, "Sentence-boundary chunking preserves semantic units.", "wiki/nlp.md"),
    ("c004", 0.78, "Recursive character splitting adapts to nested structure.", "docs/langchain.md"),
    ("c005", 0.41, "Semantic chunking groups by embedding similarity.", "papers/chunking.pdf"),
    ("c007", 0.44, "Token-budget chunking respects LLM context windows.", "docs/llm.md"),
    ("c008", 0.36, "Paragraph splitting uses whitespace as natural dividers.", "wiki/text.md"),
    ("c009", 0.88, "Hierarchical chunking indexes at multiple granularities.", "papers/hier.pdf"),
    ("c010", 0.71, "Late chunking embeds full document then splits.", "papers/late.pdf"),
]

def _build_log(chunks, rerank_top=3):
    events = [{"event": "query", "query": QUERY}]
    for rank, (cid, score, content, source) in enumerate(chunks, 1):
        events.append({"event": "retrieve", "chunk_id": cid, "score": score,
                       "content": content, "source": source})
        if rank <= rerank_top:
            events.append({"event": "rerank", "chunk_id": cid, "rank": rank})
    # Add a few connections
    ids = [c[0] for c in chunks]
    for i in range(min(4, len(ids) - 1)):
        events.append({"event": "connect", "from": ids[i], "to": ids[i+1],
                       "weight": round(0.5 + i * 0.07, 2)})
    return events

log_a = OUT / "run_a.log"
log_b = OUT / "run_b.log"
log_a.write_text("\n".join(json.dumps(e) for e in _build_log(RUN_A_CHUNKS)))
log_b.write_text("\n".join(json.dumps(e) for e in _build_log(RUN_B_CHUNKS)))
print(f"Logs written: {log_a}, {log_b}")

# ── Step 2: Visualize each run with stats report ──────────────────────────────

svg_a = visualize(str(log_a), output_path=str(OUT / "run_a.svg"),
                  title="Run A — Baseline Retrieval",
                  export_report_path=str(OUT / "run_a.json"))
svg_b = visualize(str(log_b), output_path=str(OUT / "run_b.svg"),
                  title="Run B — Improved Retrieval",
                  export_report_path=str(OUT / "run_b.json"))
print(f"SVG A → {svg_a}")
print(f"SVG B → {svg_b}")

# ── Step 3: Diff the two runs ─────────────────────────────────────────────────

svg_diff = visualize_diff(
    str(log_a), str(log_b),
    output_path=str(OUT / "run_diff.svg"),
    title="Diff: Run A vs Run B",
    change_threshold=0.05,
)
print(f"Diff  → {svg_diff}")

# ── Step 4: Side-by-side stats comparison ────────────────────────────────────

stats_a = json.loads((OUT / "run_a.json").read_text())
stats_b = json.loads((OUT / "run_b.json").read_text())

print("\n── Stats comparison ──────────────────────────────")
fmt = "{:<22} {:>10} {:>10}"
print(fmt.format("Metric", "Run A", "Run B"))
print("─" * 44)
print(fmt.format("Chunks", stats_a["chunk_count"], stats_b["chunk_count"]))
print(fmt.format("Edges", stats_a["edge_count"], stats_b["edge_count"]))
print(fmt.format("Avg score", f"{stats_a['scores']['mean']:.3f}",
                 f"{stats_b['scores']['mean']:.3f}"))
print(fmt.format("Max score", f"{stats_a['scores']['max']:.3f}",
                 f"{stats_b['scores']['max']:.3f}"))
print(fmt.format("High (≥0.75)", stats_a["scores"]["high_count"],
                 stats_b["scores"]["high_count"]))
print(fmt.format("Graph density", stats_a["graph_density"],
                 stats_b["graph_density"]))
print("\nFull pipeline complete — check outputs/ for SVGs and JSON reports.")
