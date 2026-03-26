#!/usr/bin/env python3
"""
demo.py — Prompt-Graph demonstration script.

Generates sample RAG logs and produces SVG visualizations in outputs/.
Runs completely offline with no API keys required.

Demos:
  1. Basic 5-chunk retrieval (spring layout)
  2. Large 10-chunk retrieval (shell layout)
  3. Hybrid retrieval (circular layout)
  4. Mock/dry-run mode — no log file needed
  5. Graph diff — compare two retrieval runs
  6. Cluster detection — community color-coding
  7. Stats report export — JSON alongside SVG
"""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich import box as rich_box

from prompt_graph_visuali import (
    VERSION,
    build_graph,
    compute_stats,
    export_report,
    diff_logs,
    make_mock_data,
    parse_log_file,
    render_svg,
    visualize,
    visualize_diff,
    visualize_mock,
    compute_layout,
)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
LAYOUT_SEED = int(os.getenv("LAYOUT_SEED", "42"))

console = Console()

# ── Sample data generators ───────────────────────────────────────────────────

SAMPLE_QUERIES = [
    "What is Retrieval Augmented Generation and how does it differ from fine-tuning?",
    "How does vector similarity search work in embeddings space?",
    "Explain the trade-offs between BM25 and dense retrieval methods.",
]

SAMPLE_SOURCES = [
    "papers/attention_is_all_you_need.pdf",
    "docs/rag_survey_2023.pdf",
    "wiki/vector_databases.md",
    "blog/retrieval_strategies.md",
    "docs/embedding_models.md",
    "papers/dense_passage_retrieval.pdf",
    "docs/bm25_explained.md",
    "wiki/knn_search.md",
    "papers/fusion_retrieval.pdf",
    "docs/re-ranking_guide.md",
]

SAMPLE_CONTENTS = [
    "RAG combines parametric memory (model weights) with non-parametric memory (retrieved docs).",
    "Dense retrieval uses bi-encoder models to embed queries and passages into shared vector space.",
    "BM25 is a bag-of-words retrieval function that ranks documents based on term frequency.",
    "Cross-encoders re-rank retrieved candidates by jointly encoding query and passage.",
    "Vector similarity is typically measured via cosine distance or dot-product.",
    "Chunking strategies include fixed-size, sentence-based, and semantic chunking.",
    "Hybrid retrieval combines sparse (BM25) and dense (embedding) scores via RRF.",
    "The retrieval step selects the top-k most relevant passages from a knowledge base.",
    "Embeddings from contrastive training outperform generative embeddings on retrieval tasks.",
    "Multi-hop retrieval iteratively queries to gather evidence across multiple documents.",
]


def make_simple_rag_log(n_chunks: int = 5,
                        n_connections: int = 3,
                        query_idx: int = 0,
                        seed: int = LAYOUT_SEED) -> list[dict]:
    """Generate a synthetic JSONL RAG log with n_chunks retrieved chunks."""
    import datetime
    rng = random.Random(seed)
    events = []

    query = SAMPLE_QUERIES[query_idx % len(SAMPLE_QUERIES)]
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    events.append({"event": "query", "query": query, "timestamp": ts})

    chunk_ids = [f"chunk_{i+1:03d}" for i in range(n_chunks)]
    scores = sorted([round(rng.uniform(0.45, 0.99), 3)
                     for _ in range(n_chunks)], reverse=True)

    for i, (cid, score) in enumerate(zip(chunk_ids, scores)):
        events.append({
            "event": "retrieve",
            "chunk_id": cid,
            "score": score,
            "content": SAMPLE_CONTENTS[i % len(SAMPLE_CONTENTS)],
            "source": SAMPLE_SOURCES[i % len(SAMPLE_SOURCES)],
            "timestamp": ts,
        })

    # Re-rank top chunks
    for rank, cid in enumerate(chunk_ids[:3], start=1):
        events.append({"event": "rerank", "chunk_id": cid, "rank": rank})

    # Add cross-chunk connections (similarity links)
    pairs = list(zip(chunk_ids, chunk_ids[1:]))
    rng.shuffle(pairs)
    for src, dst in pairs[:n_connections]:
        weight = round(rng.uniform(0.3, 0.85), 3)
        events.append({"event": "connect", "from": src, "to": dst, "weight": weight})

    return events


def write_log(events: list[dict], path: Path) -> None:
    """Write a list of RAG event dicts as JSONL to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    console.print(f"  [dim]Wrote log  → {path}[/dim]")


def _print_basic_stats(G, scores) -> None:
    """Print node/edge count and score range for a graph."""
    console.print(
        f"  [cyan]Nodes:[/cyan] {G.number_of_nodes()}  "
        f"[cyan]Edges:[/cyan] {G.number_of_edges()}"
    )
    if scores:
        lo, hi = min(scores), max(scores)
        console.print(f"  [cyan]Score range:[/cyan] {lo:.3f} – {hi:.3f}")


def run_demo() -> None:
    """Run all 7 demo scenarios and write outputs to the outputs/ directory."""
    console.print(Panel.fit(
        f"[bold #7B61FF]Prompt-Graph[/bold #7B61FF]  [dim]v{VERSION}[/dim]\n"
        "[cyan]Visualize RAG retrieval paths as SVG graphs[/cyan]\n"
        "[dim italic]Made autonomously by NEO · heyneo.so[/dim italic]",
        border_style="#7B61FF",
        padding=(0, 2),
    ))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("[bold cyan]Running demos", total=7)

        # ── Demo 1: Basic 5-chunk retrieval ──────────────────────────────────
        progress.update(task, description="[1/7] Basic 5-chunk retrieval (spring)")
        log1 = OUTPUT_DIR / "demo_basic.log"
        events1 = make_simple_rag_log(n_chunks=5, n_connections=3, query_idx=0)
        write_log(events1, log1)
        out1 = visualize(str(log1),
                         output_path=str(OUTPUT_DIR / "retrieval_graph_basic.svg"),
                         layout="spring",
                         title="RAG Retrieval — Basic Example")
        console.print(f"  [green]✓[/green] SVG → {out1}")
        data1 = parse_log_file(str(log1))
        G1 = build_graph(data1)
        scores1 = [d["score"] for _, d in G1.nodes(data=True)
                   if d.get("node_type") == "chunk"]
        _print_basic_stats(G1, scores1)
        progress.advance(task)

        # ── Demo 2: 10-chunk large retrieval (shell layout) ──────────────────
        progress.update(task, description="[2/7] Large 10-chunk retrieval (shell)")
        log2 = OUTPUT_DIR / "demo_large.log"
        events2 = make_simple_rag_log(n_chunks=10, n_connections=6, query_idx=1)
        write_log(events2, log2)
        out2 = visualize(str(log2),
                         output_path=str(OUTPUT_DIR / "retrieval_graph_large.svg"),
                         layout="shell",
                         title="RAG Retrieval — 10 Chunks")
        console.print(f"  [green]✓[/green] SVG → {out2}")
        data2 = parse_log_file(str(log2))
        G2 = build_graph(data2)
        scores2 = [d["score"] for _, d in G2.nodes(data=True)
                   if d.get("node_type") == "chunk"]
        _print_basic_stats(G2, scores2)
        progress.advance(task)

        # ── Demo 3: Hybrid retrieval (circular layout) ────────────────────────
        progress.update(task, description="[3/7] Hybrid retrieval (circular)")
        log3 = OUTPUT_DIR / "demo_hybrid.log"
        events3 = make_simple_rag_log(n_chunks=7, n_connections=8, query_idx=2)
        write_log(events3, log3)
        out3 = visualize(str(log3),
                         output_path=str(OUTPUT_DIR / "retrieval_graph_hybrid.svg"),
                         layout="circular",
                         title="RAG Retrieval — Hybrid Strategy")
        console.print(f"  [green]✓[/green] SVG → {out3}")
        data3 = parse_log_file(str(log3))
        G3 = build_graph(data3)
        _print_basic_stats(G3, [])
        progress.advance(task)

        # ── Demo 4: Mock / dry-run mode (no log file) ─────────────────────────
        progress.update(task, description="[4/7] Mock mode — no log file needed")
        out4 = visualize_mock(
            output_path=str(OUTPUT_DIR / "retrieval_graph_mock.svg"),
            layout="spring",
            title="RAG Retrieval — Mock (Dry-run)",
            n_chunks=8,
            n_connections=5,
            query_idx=2,
        )
        console.print(f"  [green]✓[/green] SVG → {out4}")
        mock_data = make_mock_data(n_chunks=8, n_connections=5, query_idx=2)
        Gm = build_graph(mock_data)
        _print_basic_stats(Gm, [d["score"] for _, d in Gm.nodes(data=True)
                                if d.get("node_type") == "chunk"])
        progress.advance(task)

        # ── Demo 5: Diff two retrieval runs ──────────────────────────────────
        progress.update(task, description="[5/7] Graph diff — v1 vs v2")
        log_base = OUTPUT_DIR / "demo_diff_v1.log"
        events_base = make_simple_rag_log(n_chunks=5, n_connections=2,
                                          query_idx=0, seed=10)
        write_log(events_base, log_base)

        log_updated = OUTPUT_DIR / "demo_diff_v2.log"
        events_updated = make_simple_rag_log(n_chunks=6, n_connections=3,
                                             query_idx=0, seed=99)
        write_log(events_updated, log_updated)

        out5 = visualize_diff(
            str(log_base),
            str(log_updated),
            output_path=str(OUTPUT_DIR / "retrieval_graph_diff.svg"),
            layout="spring",
            title="RAG Diff — v1 vs v2",
        )
        console.print(f"  [green]✓[/green] Diff SVG → {out5}")
        d_base = parse_log_file(str(log_base))
        d_upd = parse_log_file(str(log_updated))
        _, diff_map, score_delta = diff_logs(d_base, d_upd)
        added   = sum(1 for v in diff_map.values() if v == "added")
        removed = sum(1 for v in diff_map.values() if v == "removed")
        changed = sum(1 for v in diff_map.values() if v == "changed")
        console.print(
            f"  [green]+{added} added[/green]  "
            f"[red]-{removed} removed[/red]  "
            f"[yellow]~{changed} score-changed[/yellow]"
        )
        progress.advance(task)

        # ── Demo 6: Cluster detection ─────────────────────────────────────────
        progress.update(task, description="[6/7] Cluster detection")
        out6 = visualize(
            str(log2),
            output_path=str(OUTPUT_DIR / "retrieval_graph_clustered.svg"),
            layout="spring",
            title="RAG Retrieval — Cluster View",
            cluster=True,
        )
        console.print(f"  [green]✓[/green] SVG → {out6}")
        G_cluster = build_graph(parse_log_file(str(log2)), cluster=True)
        cluster_ids = {d["cluster"] for _, d in G_cluster.nodes(data=True)
                       if d.get("node_type") == "chunk" and d.get("cluster") is not None}
        n_chunks_total = sum(
            1 for _, d in G_cluster.nodes(data=True) if d.get("node_type") == "chunk"
        )
        console.print(
            f"  [cyan]Detected {len(cluster_ids)} cluster(s)[/cyan] "
            f"across {n_chunks_total} chunks"
        )
        progress.advance(task)

        # ── Demo 7: Stats report export ───────────────────────────────────────
        progress.update(task, description="[7/7] Stats report export")
        report_path = str(OUTPUT_DIR / "retrieval_graph_report.json")
        out7 = visualize(
            str(log2),
            output_path=str(OUTPUT_DIR / "retrieval_graph_with_report.svg"),
            layout="shell",
            title="RAG Retrieval — With Report",
            export_report_path=report_path,
        )
        console.print(f"  [green]✓[/green] SVG → {out7}")
        console.print(f"  [green]✓[/green] JSON report → {report_path}")
        with open(report_path) as f:
            report = json.load(f)
        s = report["scores"]
        console.print(
            f"  [cyan]mean=[/cyan]{s['mean']:.3f}  "
            f"[cyan]stdev=[/cyan]{s['stdev']:.3f}  "
            f"[green]high={s['high_count']}[/green]/"
            f"[yellow]med={s['med_count']}[/yellow]/"
            f"[red]low={s['low_count']}[/red]"
        )
        console.print(
            f"  [cyan]Density:[/cyan] {report['graph_density']}  "
            f"[cyan]Sources:[/cyan] {len(report['sources'])}"
        )
        progress.advance(task)

    # ── Summary table ─────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold #7B61FF]Demo complete — output files[/bold #7B61FF]")

    svg_tbl = Table(box=rich_box.ROUNDED, border_style="dim", show_header=True)
    svg_tbl.add_column("SVG file", style="cyan", no_wrap=True)
    svg_tbl.add_column("Size", style="white", justify="right")
    for p in sorted(OUTPUT_DIR.glob("*.svg")):
        size_kb = p.stat().st_size / 1024
        svg_tbl.add_row(str(p), f"{size_kb:.1f} KB")
    console.print(svg_tbl)

    extra_tbl = Table(box=rich_box.SIMPLE, border_style="dim", show_header=True)
    extra_tbl.add_column("File", style="dim", no_wrap=True)
    extra_tbl.add_column("Info", style="dim")
    for p in sorted(OUTPUT_DIR.glob("*.json")):
        extra_tbl.add_row(str(p), "stats report")
    for p in sorted(OUTPUT_DIR.glob("*.log")):
        lines = sum(1 for _ in open(p, encoding="utf-8"))
        extra_tbl.add_row(str(p), f"{lines} events")
    console.print(extra_tbl)

    console.print()
    console.print("[dim]Open any .svg file in your browser to view the graph.[/dim]")


if __name__ == "__main__":
    run_demo()
