#!/usr/bin/env python3
"""
Prompt-Graph: Visualize RAG retrieval paths as SVG graphs.
Parses RAG log files and outputs a retrieval_graph.svg showing
which chunks were retrieved and how they connect.

New in v2:
  - Mock/dry-run mode (no log file needed)
  - JSON stats report export
  - Graph diff between two log files
  - Cluster detection + color-coding
  - Score threshold env vars
  - --export-report, --mock, --diff, --cluster CLI flags
"""

import argparse
import datetime
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import svgwrite
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box as rich_box

VERSION = "2.0.0"

_console = Console()         # stdout — success / info
_err = Console(stderr=True)  # stderr — warnings / errors

# ── Configuration via environment variables ──────────────────────────────────
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
DEFAULT_SVG_WIDTH = int(os.getenv("SVG_WIDTH", "1200"))
DEFAULT_SVG_HEIGHT = int(os.getenv("SVG_HEIGHT", "800"))
NODE_RADIUS = int(os.getenv("NODE_RADIUS", "40"))
FONT_FAMILY = os.getenv("FONT_FAMILY", "monospace")
COLOR_QUERY = os.getenv("COLOR_QUERY", "#7B61FF")
COLOR_CHUNK = os.getenv("COLOR_CHUNK", "#00B4D8")
COLOR_SECONDARY = os.getenv("COLOR_SECONDARY", "#48CAE4")
COLOR_EDGE = os.getenv("COLOR_EDGE", "#ADB5BD")
COLOR_EDGE_STRONG = os.getenv("COLOR_EDGE_STRONG", "#6C757D")
COLOR_BG = os.getenv("COLOR_BG", "#0D1117")
COLOR_TEXT = os.getenv("COLOR_TEXT", "#E6EDF3")
COLOR_SCORE_HIGH = os.getenv("COLOR_SCORE_HIGH", "#3FB950")
COLOR_SCORE_MED = os.getenv("COLOR_SCORE_MED", "#D29922")
COLOR_SCORE_LOW = os.getenv("COLOR_SCORE_LOW", "#F85149")
LAYOUT_ALGO = os.getenv("LAYOUT_ALGO", "spring")
LAYOUT_SEED = int(os.getenv("LAYOUT_SEED", "42"))
GITHUB_REPO = os.getenv("GITHUB_REPO", "github.com/dakshjain-1616/prompt-graph")

# Score thresholds (configurable so teams can tune to their scoring model)
SCORE_HIGH_THRESHOLD = float(os.getenv("SCORE_HIGH_THRESHOLD", "0.75"))
SCORE_MED_THRESHOLD = float(os.getenv("SCORE_MED_THRESHOLD", "0.50"))

# Cluster color palette — cycled for graphs with >8 clusters
_CLUSTER_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#F0A04B", "#A29BFE",
]

# Diff annotation colors
_DIFF_COLORS = {
    "added":     "#39D353",  # bright green ring
    "removed":   "#F85149",  # red ring
    "changed":   "#E3B341",  # amber ring
    "unchanged": None,       # no special ring
}


# ── Log Parsing ──────────────────────────────────────────────────────────────

def parse_log_file(log_path: str) -> dict[str, Any]:
    """
    Parse a RAG log file and extract graph data.

    Supports two formats:
      1. JSONL (one JSON object per line) — structured events
      2. Plain text  — heuristic extraction of chunk IDs and scores

    JSONL event types:
      {"event": "query",    "query": "...", "timestamp": "..."}
      {"event": "retrieve", "chunk_id": "c1", "score": 0.95,
                            "content": "...", "source": "doc.txt"}
      {"event": "connect",  "from": "c1", "to": "c2", "weight": 0.7}
      {"event": "rerank",   "chunk_id": "c1", "rank": 1}
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Log file not found: {log_path}\n"
            f"  Tip: run with --mock to generate a synthetic graph without a log file."
        )

    raw = path.read_text(encoding="utf-8")
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    query_text = ""
    nodes: dict[str, dict] = {}   # chunk_id → attrs
    edges: list[dict] = []
    reranks: dict[str, int] = {}
    parse_errors = 0

    jsonl_count = sum(1 for l in lines if l.startswith("{"))
    use_jsonl = jsonl_count >= len(lines) * 0.5

    if use_jsonl:
        for line in lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            event = obj.get("event", "")

            if event == "query" or "query" in obj:
                query_text = obj.get("query", query_text)

            elif event == "retrieve":
                cid = obj.get("chunk_id", f"chunk_{len(nodes)}")
                nodes[cid] = {
                    "chunk_id": cid,
                    "score": float(obj.get("score", 0.0)),
                    "content": obj.get("content", ""),
                    "source": obj.get("source", ""),
                    "metadata": {k: v for k, v in obj.items()
                                 if k not in {"event", "chunk_id", "score",
                                              "content", "source"}},
                }

            elif event == "connect":
                src = obj.get("from", "")
                dst = obj.get("to", "")
                if src and dst:
                    edges.append({
                        "from": src,
                        "to": dst,
                        "weight": float(obj.get("weight", 0.5)),
                    })

            elif event == "rerank":
                cid = obj.get("chunk_id", "")
                if cid:
                    reranks[cid] = int(obj.get("rank", 0))

            # Also handle flat retrieve-like dicts (no "event" key)
            elif "chunk_id" in obj:
                cid = obj["chunk_id"]
                nodes[cid] = {
                    "chunk_id": cid,
                    "score": float(obj.get("score", 0.0)),
                    "content": obj.get("content", ""),
                    "source": obj.get("source", ""),
                    "metadata": {},
                }

        if parse_errors > 0:
            _err.print(
                f"  [yellow]Warning:[/yellow] {parse_errors}/{len(lines)} lines "
                f"failed JSON parsing (skipped)."
            )

    else:
        # Heuristic plain-text parsing
        import re
        for line in lines:
            # Query lines: "Query: ..." or "Q: ..."
            m = re.match(r"(?:query|q)\s*:\s*(.+)", line, re.IGNORECASE)
            if m:
                query_text = m.group(1).strip()
                continue

            # Chunk lines: "chunk_001 score=0.95 ..."
            m = re.match(
                r"(chunk[_\-]?\w+)\s+(?:score[=:\s]+)([0-9.]+)(.*)",
                line, re.IGNORECASE)
            if m:
                cid, score, rest = m.group(1), float(m.group(2)), m.group(3)
                src_m = re.search(r"source[=:\s]+(\S+)", rest, re.IGNORECASE)
                nodes[cid] = {
                    "chunk_id": cid,
                    "score": score,
                    "content": rest.strip(),
                    "source": src_m.group(1) if src_m else "",
                    "metadata": {},
                }

    # Apply rerank info
    for cid, rank in reranks.items():
        if cid in nodes:
            nodes[cid]["rank"] = rank

    return {
        "query": query_text,
        "nodes": list(nodes.values()),
        "edges": edges,
    }


# ── Mock Data Generation ──────────────────────────────────────────────────────

_MOCK_QUERIES = [
    "What is Retrieval Augmented Generation and how does it differ from fine-tuning?",
    "How does vector similarity search work in embeddings space?",
    "Explain the trade-offs between BM25 and dense retrieval methods.",
    "What chunking strategies work best for long documents?",
    "How do cross-encoders improve retrieval quality?",
]

_MOCK_SOURCES = [
    "papers/attention_is_all_you_need.pdf",
    "docs/rag_survey_2023.pdf",
    "wiki/vector_databases.md",
    "blog/retrieval_strategies.md",
    "docs/embedding_models.md",
    "papers/dense_passage_retrieval.pdf",
    "docs/bm25_explained.md",
    "wiki/knn_search.md",
]

_MOCK_CONTENTS = [
    "RAG combines parametric memory (model weights) with non-parametric memory (retrieved docs).",
    "Dense retrieval uses bi-encoder models to embed queries and passages into shared vector space.",
    "BM25 is a bag-of-words retrieval function that ranks documents based on term frequency.",
    "Cross-encoders re-rank retrieved candidates by jointly encoding query and passage.",
    "Vector similarity is typically measured via cosine distance or dot-product.",
    "Chunking strategies include fixed-size, sentence-based, and semantic chunking.",
    "Hybrid retrieval combines sparse (BM25) and dense (embedding) scores via RRF.",
    "The retrieval step selects the top-k most relevant passages from a knowledge base.",
    "Embeddings from contrastive training outperform generative embeddings on retrieval.",
    "Multi-hop retrieval iteratively queries to gather evidence across multiple documents.",
]


def make_mock_data(
    n_chunks: int = 6,
    n_connections: int = 4,
    query_idx: int = 0,
    seed: int = LAYOUT_SEED,
) -> dict[str, Any]:
    """
    Generate synthetic RAG data without needing a log file.

    Returns the same structure as parse_log_file(), so it can be passed
    directly to build_graph() and render_svg().
    """
    rng = random.Random(seed)

    query = _MOCK_QUERIES[query_idx % len(_MOCK_QUERIES)]
    chunk_ids = [f"chunk_{i+1:03d}" for i in range(n_chunks)]
    scores = sorted(
        [round(rng.uniform(0.40, 0.99), 3) for _ in range(n_chunks)],
        reverse=True,
    )

    nodes = []
    for i, (cid, score) in enumerate(zip(chunk_ids, scores)):
        node: dict[str, Any] = {
            "chunk_id": cid,
            "score": score,
            "content": _MOCK_CONTENTS[i % len(_MOCK_CONTENTS)],
            "source": _MOCK_SOURCES[i % len(_MOCK_SOURCES)],
        }
        if i < 3:
            node["rank"] = i + 1
        nodes.append(node)

    pairs = list(zip(chunk_ids, chunk_ids[1:]))
    rng.shuffle(pairs)
    edges = [
        {"from": src, "to": dst, "weight": round(rng.uniform(0.3, 0.85), 3)}
        for src, dst in pairs[:min(n_connections, len(pairs))]
    ]

    return {"query": query, "nodes": nodes, "edges": edges}


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(
    G: nx.DiGraph,
    data: dict[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """
    Compute rich statistics about the retrieval graph.

    Includes: timestamp, score distribution, top chunks, graph density,
    source breakdown, and degree centrality per chunk.
    """
    if generated_at is None:
        generated_at = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    chunk_nodes = [(n, d) for n, d in G.nodes(data=True)
                   if d.get("node_type") == "chunk"]
    scores = [d["score"] for _, d in chunk_nodes]

    centrality = nx.degree_centrality(G)

    sources: dict[str, int] = {}
    for _, d in chunk_nodes:
        src = d.get("source") or "(unknown)"
        sources[src] = sources.get(src, 0) + 1

    top_chunks = sorted(
        [
            {
                "chunk_id": n,
                "score": d["score"],
                "source": d.get("source", ""),
                "rank": d.get("rank"),
                "centrality": round(centrality.get(n, 0.0), 4),
                "content_preview": (d.get("content") or "")[:80],
            }
            for n, d in chunk_nodes
        ],
        key=lambda x: x["score"],
        reverse=True,
    )[:5]

    return {
        "generated_at": generated_at,
        "query": data.get("query", ""),
        "chunk_count": len(chunk_nodes),
        "edge_count": G.number_of_edges(),
        "graph_density": round(nx.density(G), 4),
        "scores": {
            "mean": round(statistics.mean(scores), 4) if scores else 0.0,
            "median": round(statistics.median(scores), 4) if scores else 0.0,
            "stdev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
            "min": round(min(scores), 4) if scores else 0.0,
            "max": round(max(scores), 4) if scores else 0.0,
            "high_count": sum(1 for s in scores if s >= SCORE_HIGH_THRESHOLD),
            "med_count": sum(
                1 for s in scores if SCORE_MED_THRESHOLD <= s < SCORE_HIGH_THRESHOLD
            ),
            "low_count": sum(1 for s in scores if s < SCORE_MED_THRESHOLD),
            "high_threshold": SCORE_HIGH_THRESHOLD,
            "med_threshold": SCORE_MED_THRESHOLD,
        },
        "sources": sources,
        "top_chunks": top_chunks,
        "layout_algo": LAYOUT_ALGO,
    }


def export_report(G: nx.DiGraph, data: dict[str, Any], report_path: str) -> str:
    """
    Export a JSON statistics report alongside the SVG.

    Returns the path written to.
    """
    stats = compute_stats(G, data)
    Path(report_path).write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_path


# ── Graph Diff ────────────────────────────────────────────────────────────────

def diff_logs(
    data1: dict[str, Any],
    data2: dict[str, Any],
    change_threshold: float = 0.05,
) -> tuple[dict[str, Any], dict[str, str], dict[str, float]]:
    """
    Compare two parsed log datasets and compute diff annotations.

    Args:
        data1: Earlier/baseline log (from parse_log_file or make_mock_data)
        data2: Newer log to compare against
        change_threshold: Min score delta to classify a node as "changed"

    Returns:
        merged_data  — all nodes/edges from both logs combined (uses data2 values)
        diff_map     — {chunk_id: "added" | "removed" | "changed" | "unchanged"}
        score_delta  — {chunk_id: float}  (positive = score improved in data2)
    """
    nodes1 = {n["chunk_id"]: n for n in data1.get("nodes", [])}
    nodes2 = {n["chunk_id"]: n for n in data2.get("nodes", [])}

    all_ids = set(nodes1) | set(nodes2)
    diff_map: dict[str, str] = {}
    score_delta: dict[str, float] = {}
    merged_nodes = []

    for cid in sorted(all_ids):
        if cid in nodes2 and cid not in nodes1:
            diff_map[cid] = "added"
            score_delta[cid] = nodes2[cid]["score"]
            merged_nodes.append(nodes2[cid])
        elif cid in nodes1 and cid not in nodes2:
            diff_map[cid] = "removed"
            score_delta[cid] = -nodes1[cid]["score"]
            merged_nodes.append(nodes1[cid])
        else:
            delta = round(nodes2[cid]["score"] - nodes1[cid]["score"], 4)
            score_delta[cid] = delta
            diff_map[cid] = "changed" if abs(delta) >= change_threshold else "unchanged"
            merged_nodes.append(nodes2[cid])

    merged_data = {
        "query": data2.get("query") or data1.get("query", ""),
        "nodes": merged_nodes,
        "edges": data2.get("edges", []),
    }
    return merged_data, diff_map, score_delta


# ── Graph Building ───────────────────────────────────────────────────────────

def build_graph(data: dict[str, Any], *, cluster: bool = False) -> nx.DiGraph:
    """Build a NetworkX directed graph from parsed log data."""
    G = nx.DiGraph()

    # Query node
    query = data.get("query", "Query")
    G.add_node(
        "__query__",
        label=query[:40],
        node_type="query",
        score=1.0,
        source="",
        content=query,
        cluster=None,
    )

    # Chunk nodes
    for node in data.get("nodes", []):
        cid = node["chunk_id"]
        label = node.get("content", cid)[:30].replace("\n", " ")
        G.add_node(
            cid,
            label=label,
            node_type="chunk",
            score=node.get("score", 0.0),
            source=node.get("source", ""),
            content=node.get("content", ""),
            rank=node.get("rank", None),
            cluster=None,
        )
        # Edge from query to every retrieved chunk
        G.add_edge("__query__", cid, weight=node.get("score", 0.5))

    # Explicit chunk-to-chunk edges
    for edge in data.get("edges", []):
        src, dst = edge["from"], edge["to"]
        if G.has_node(src) and G.has_node(dst):
            G.add_edge(src, dst, weight=edge.get("weight", 0.5))

    if cluster:
        _assign_clusters(G)

    return G


def _assign_clusters(G: nx.DiGraph) -> None:
    """
    Assign cluster IDs to chunk nodes using greedy modularity communities.
    Clustering is best-effort: silently skips if the graph has no edges.
    """
    chunk_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "chunk"]
    if len(chunk_nodes) < 2:
        return

    subgraph = G.subgraph(chunk_nodes).to_undirected()
    if subgraph.number_of_edges() == 0:
        # No inter-chunk edges — nothing to cluster
        return

    try:
        communities = list(nx.community.greedy_modularity_communities(subgraph))
        for cluster_id, community in enumerate(communities):
            for node in community:
                if G.has_node(node):
                    G.nodes[node]["cluster"] = cluster_id
    except Exception:
        pass  # Clustering is best-effort; never crash the pipeline


# ── Layout ──────────────────────────────────────────────────────────────────

def compute_layout(G: nx.DiGraph, width: int, height: int) -> dict[str, tuple]:
    """Compute node positions scaled to SVG canvas."""
    algo = LAYOUT_ALGO.lower()
    pad = NODE_RADIUS * 3

    if algo == "spring":
        pos = nx.spring_layout(G, seed=LAYOUT_SEED, k=2.5, iterations=60)
    elif algo == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif algo == "circular":
        pos = nx.circular_layout(G)
    elif algo == "shell":
        # Query at center, chunks on outer shell
        query_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "query"]
        chunk_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") != "query"]
        pos = nx.shell_layout(G, nlist=[query_nodes, chunk_nodes])
    else:
        pos = nx.spring_layout(G, seed=LAYOUT_SEED)

    # Normalise to [pad, width-pad] × [pad, height-pad]
    xs = [v[0] for v in pos.values()]
    ys = [v[1] for v in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    rx = (max_x - min_x) or 1
    ry = (max_y - min_y) or 1

    scaled = {}
    for node, (x, y) in pos.items():
        sx = pad + (x - min_x) / rx * (width - 2 * pad)
        sy = pad + (y - min_y) / ry * (height - 2 * pad)
        scaled[node] = (sx, sy)

    return scaled


# ── SVG Rendering ────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= SCORE_HIGH_THRESHOLD:
        return COLOR_SCORE_HIGH
    if score >= SCORE_MED_THRESHOLD:
        return COLOR_SCORE_MED
    return COLOR_SCORE_LOW


def _truncate(text: str, max_chars: int = 22) -> str:
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"


def render_svg(
    G: nx.DiGraph,
    pos: dict,
    output_path: str,
    width: int = DEFAULT_SVG_WIDTH,
    height: int = DEFAULT_SVG_HEIGHT,
    title: str = "RAG Retrieval Graph",
    diff_map: dict[str, str] | None = None,
    score_delta: dict[str, float] | None = None,
) -> str:
    """
    Render the graph to an SVG file and return the output path.

    Args:
        diff_map    — optional {chunk_id: "added"|"removed"|"changed"|"unchanged"}
        score_delta — optional {chunk_id: float} delta labels for diff mode
    """
    dwg = svgwrite.Drawing(output_path, size=(width, height), profile="full")

    # ── Defs: arrow marker, drop shadow ─────────────────────────────────────
    defs = dwg.defs
    marker = dwg.marker(insert=(10, 5), size=(10, 10), orient="auto", id="arrow")
    marker.add(dwg.path(d="M 0 0 L 10 5 L 0 10 z", fill=COLOR_EDGE_STRONG))
    defs.add(marker)

    # Drop shadow filter
    filt = dwg.filter(id="shadow", x="-20%", y="-20%",
                      width="140%", height="140%")
    filt.feGaussianBlur(in_="SourceAlpha", stdDeviation="4", result="blur")
    filt.feOffset(in_="blur", dx="2", dy="2", result="offsetBlur")
    filt.feFlood(**{"flood-color": "#000000", "flood-opacity": "0.5", "result": "color"})
    filt.feComposite(in_="color", in2="offsetBlur", operator="in", result="shadow")
    filt.feBlend(in_="SourceGraphic", in2="shadow", mode="normal")
    defs.add(filt)

    # ── Background ───────────────────────────────────────────────────────────
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill=COLOR_BG))

    # ── Title ────────────────────────────────────────────────────────────────
    dwg.add(dwg.text(
        title,
        insert=(width / 2, 28),
        text_anchor="middle",
        font_size="18px",
        font_family=FONT_FAMILY,
        font_weight="bold",
        fill=COLOR_TEXT,
        opacity="0.85",
    ))

    chunk_count = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "chunk")
    edge_count = G.number_of_edges()
    mode_tag = " · DIFF" if diff_map else ""
    subtitle = f"{chunk_count} chunks · {edge_count} connections{mode_tag}"
    dwg.add(dwg.text(
        subtitle,
        insert=(width / 2, 48),
        text_anchor="middle",
        font_size="11px",
        font_family=FONT_FAMILY,
        fill=COLOR_TEXT,
        opacity="0.5",
    ))

    # Timestamp (bottom-left of subtitle area)
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    dwg.add(dwg.text(
        f"Generated {ts}",
        insert=(width - 12, 48),
        text_anchor="end",
        font_size="9px",
        font_family=FONT_FAMILY,
        fill=COLOR_TEXT,
        opacity="0.3",
    ))

    # ── Edges ────────────────────────────────────────────────────────────────
    edge_group = dwg.g(id="edges")
    for src, dst, edata in G.edges(data=True):
        if src not in pos or dst not in pos:
            continue
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        weight = edata.get("weight", 0.5)

        # Dim edges involving removed nodes in diff mode
        is_removed_edge = (
            diff_map is not None
            and (diff_map.get(src) == "removed" or diff_map.get(dst) == "removed")
        )
        edge_opacity = 0.2 if is_removed_edge else max(0.3, min(1.0, weight))

        # Shorten line so it doesn't overlap nodes
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy) or 1
        r = NODE_RADIUS + 4
        x1s = x1 + dx / dist * r
        y1s = y1 + dy / dist * r
        x2e = x2 - dx / dist * (r + 12)
        y2e = y2 - dy / dist * (r + 12)

        stroke_w = max(1.0, weight * 3.5)
        stroke_c = COLOR_EDGE_STRONG if weight >= 0.7 else COLOR_EDGE

        line = dwg.line(
            start=(x1s, y1s),
            end=(x2e, y2e),
            stroke=stroke_c,
            stroke_width=str(stroke_w),
            opacity=str(edge_opacity),
            marker_end="url(#arrow)",
        )
        edge_group.add(line)

        # Weight label on edge midpoint
        mx, my = (x1s + x2e) / 2, (y1s + y2e) / 2
        edge_group.add(dwg.text(
            f"{weight:.2f}",
            insert=(mx, my - 5),
            text_anchor="middle",
            font_size="9px",
            font_family=FONT_FAMILY,
            fill=COLOR_TEXT,
            opacity="0.55",
        ))
    dwg.add(edge_group)

    # ── Nodes ────────────────────────────────────────────────────────────────
    node_group = dwg.g(id="nodes")
    for node_id, ndata in G.nodes(data=True):
        if node_id not in pos:
            continue
        cx, cy = pos[node_id]
        ntype = ndata.get("node_type", "chunk")
        score = ndata.get("score", 0.0)
        label = ndata.get("label", node_id)
        source = ndata.get("source", "")
        rank = ndata.get("rank", None)
        cluster_id = ndata.get("cluster", None)

        # Diff annotations
        diff_status = diff_map.get(node_id) if diff_map else None
        delta = (score_delta or {}).get(node_id, 0.0)
        node_opacity = "0.35" if diff_status == "removed" else "1.0"

        if ntype == "query":
            fill = COLOR_QUERY
            stroke = "#B39DFF"
            radius = NODE_RADIUS + 8
        else:
            # Cluster mode overrides score color
            if cluster_id is not None:
                fill = _CLUSTER_COLORS[cluster_id % len(_CLUSTER_COLORS)]
            else:
                fill = _score_color(score)
            stroke = "#FFFFFF"
            radius = NODE_RADIUS

        # Diff glow ring (drawn before node so it appears behind)
        diff_ring_color = _DIFF_COLORS.get(diff_status) if diff_status else None
        if diff_ring_color:
            node_group.add(dwg.circle(
                center=(cx, cy),
                r=radius + 10,
                fill="none",
                stroke=diff_ring_color,
                stroke_width="3",
                opacity="0.9",
            ))

        # Outer glow ring (standard)
        node_group.add(dwg.circle(
            center=(cx, cy),
            r=radius + 6,
            fill="none",
            stroke=fill,
            stroke_width="1",
            opacity="0.25",
        ))

        # Main circle
        node_group.add(dwg.circle(
            center=(cx, cy),
            r=radius,
            fill=fill,
            stroke=stroke,
            stroke_width="1.5",
            filter="url(#shadow)",
            opacity=node_opacity,
        ))

        # Score arc background (thin ring) — skip for removed nodes
        if ntype == "chunk" and diff_status != "removed":
            arc_r = radius - 4
            angle = score * 360
            rad = math.radians(angle - 90)
            end_x = cx + arc_r * math.cos(rad)
            end_y = cy + arc_r * math.sin(rad)
            large = 1 if angle > 180 else 0
            if score > 0.01:
                arc_path = (
                    f"M {cx} {cy - arc_r} "
                    f"A {arc_r} {arc_r} 0 {large} 1 "
                    f"{end_x:.1f} {end_y:.1f} "
                    f"L {cx} {cy} Z"
                )
                node_group.add(dwg.path(d=arc_path, fill="#FFFFFF", opacity="0.12"))

        # Node text label
        node_label_str = "QUERY" if ntype == "query" else _truncate(label, 16)
        node_group.add(dwg.text(
            node_label_str,
            insert=(cx, cy - 4),
            text_anchor="middle",
            font_size="9px",
            font_family=FONT_FAMILY,
            font_weight="bold",
            fill="#FFFFFF",
            opacity=node_opacity,
        ))

        # Score sub-label
        if ntype == "chunk":
            node_group.add(dwg.text(
                f"score:{score:.2f}",
                insert=(cx, cy + 8),
                text_anchor="middle",
                font_size="8px",
                font_family=FONT_FAMILY,
                fill="#FFFFFF",
                opacity="0.8" if diff_status != "removed" else "0.4",
            ))

        # Score delta label (diff mode)
        if diff_status in ("added", "removed", "changed") and delta != 0.0:
            sign = "+" if delta > 0 else ""
            delta_color = "#39D353" if delta > 0 else "#F85149"
            node_group.add(dwg.text(
                f"{sign}{delta:.2f}",
                insert=(cx, cy + 20),
                text_anchor="middle",
                font_size="8px",
                font_family=FONT_FAMILY,
                fill=delta_color,
                opacity="0.95",
            ))

        # Rank badge
        if rank is not None:
            badge_x, badge_y = cx + radius - 8, cy - radius + 8
            node_group.add(dwg.circle(center=(badge_x, badge_y), r=9, fill="#E36209"))
            node_group.add(dwg.text(
                f"#{rank}",
                insert=(badge_x, badge_y + 4),
                text_anchor="middle",
                font_size="8px",
                font_family=FONT_FAMILY,
                font_weight="bold",
                fill="#FFFFFF",
            ))

        # Cluster badge
        if cluster_id is not None and ntype == "chunk":
            badge_x2 = cx - radius + 8
            badge_y2 = cy - radius + 8
            node_group.add(dwg.circle(
                center=(badge_x2, badge_y2),
                r=8,
                fill=_CLUSTER_COLORS[cluster_id % len(_CLUSTER_COLORS)],
                stroke="#FFFFFF",
                stroke_width="1",
            ))
            node_group.add(dwg.text(
                str(cluster_id),
                insert=(badge_x2, badge_y2 + 4),
                text_anchor="middle",
                font_size="8px",
                font_family=FONT_FAMILY,
                font_weight="bold",
                fill="#000000",
            ))

        # Source label below node
        if source:
            node_group.add(dwg.text(
                _truncate(source, 20),
                insert=(cx, cy + radius + 14),
                text_anchor="middle",
                font_size="8px",
                font_family=FONT_FAMILY,
                fill=COLOR_TEXT,
                opacity="0.55",
            ))

        # Chunk ID below node
        if ntype == "chunk":
            node_group.add(dwg.text(
                _truncate(node_id, 18),
                insert=(cx, cy + radius + 24),
                text_anchor="middle",
                font_size="7px",
                font_family=FONT_FAMILY,
                fill=COLOR_TEXT,
                opacity="0.4",
            ))

    dwg.add(node_group)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend = dwg.g(id="legend")
    lx, ly = 20, height - 100

    if diff_map:
        # Diff legend
        legend.add(dwg.rect(
            insert=(lx - 8, ly - 16),
            size=(200, 100),
            fill="#161B22",
            stroke="#30363D",
            rx="6",
        ))
        legend.add(dwg.text("Diff Legend", insert=(lx, ly),
                            font_size="10px", font_family=FONT_FAMILY,
                            font_weight="bold", fill=COLOR_TEXT))
        diff_items = [
            ("#39D353", "Added chunk"),
            ("#F85149", "Removed chunk"),
            ("#E3B341", "Score changed ≥5%"),
            (COLOR_TEXT, "Unchanged"),
        ]
        for i, (color, label) in enumerate(diff_items):
            iy = ly + 14 + i * 15
            legend.add(dwg.circle(center=(lx + 6, iy - 4), r=5, fill=color))
            legend.add(dwg.text(label, insert=(lx + 16, iy),
                                font_size="9px", font_family=FONT_FAMILY,
                                fill=COLOR_TEXT, opacity="0.75"))
    else:
        # Standard legend
        legend.add(dwg.rect(
            insert=(lx - 8, ly - 16),
            size=(200, 90),
            fill="#161B22",
            stroke="#30363D",
            rx="6",
        ))
        legend.add(dwg.text("Legend", insert=(lx, ly),
                            font_size="10px", font_family=FONT_FAMILY,
                            font_weight="bold", fill=COLOR_TEXT))
        items = [
            (COLOR_QUERY, "Query node"),
            (COLOR_SCORE_HIGH, f"Score ≥ {SCORE_HIGH_THRESHOLD}"),
            (COLOR_SCORE_MED, f"Score {SCORE_MED_THRESHOLD}–{SCORE_HIGH_THRESHOLD}"),
            (COLOR_SCORE_LOW, f"Score < {SCORE_MED_THRESHOLD}"),
        ]
        for i, (color, label) in enumerate(items):
            iy = ly + 14 + i * 15
            legend.add(dwg.circle(center=(lx + 6, iy - 4), r=5, fill=color))
            legend.add(dwg.text(label, insert=(lx + 16, iy),
                                font_size="9px", font_family=FONT_FAMILY,
                                fill=COLOR_TEXT, opacity="0.75"))
    dwg.add(legend)

    # ── Footer ───────────────────────────────────────────────────────────────
    dwg.add(dwg.text(
        f"Generated by Prompt-Graph · {GITHUB_REPO}",
        insert=(width / 2, height - 8),
        text_anchor="middle",
        font_size="9px",
        font_family=FONT_FAMILY,
        fill=COLOR_TEXT,
        opacity="0.3",
    ))

    dwg.save()
    return output_path


# ── High-level API ───────────────────────────────────────────────────────────

_DEFAULT_TITLE = "RAG Retrieval Graph"


def visualize(
    log_path: str,
    output_path: str | None = None,
    width: int = DEFAULT_SVG_WIDTH,
    height: int = DEFAULT_SVG_HEIGHT,
    layout: str = LAYOUT_ALGO,
    title: str = _DEFAULT_TITLE,
    export_report_path: str | None = None,
    cluster: bool = False,
) -> str:
    """
    Parse a RAG log, build a graph, and render to SVG.

    Args:
        log_path          — path to the RAG log file (JSONL or plain text)
        output_path       — SVG output path (auto-derived from log_path if None)
        export_report_path — if set, write a JSON stats report to this path
        cluster           — detect and color-code chunk communities

    Returns the path to the output SVG file.
    """
    global LAYOUT_ALGO
    LAYOUT_ALGO = layout

    data = parse_log_file(log_path)
    G = build_graph(data, cluster=cluster)

    if output_path is None:
        stem = Path(log_path).stem
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{stem}_graph.svg")

    pos = compute_layout(G, width, height)
    query_preview = data.get("query", "")
    if query_preview and title == _DEFAULT_TITLE:
        title = f'RAG Retrieval Graph · "{query_preview[:50]}"'

    render_svg(G, pos, output_path, width=width, height=height, title=title)

    if export_report_path:
        export_report(G, data, export_report_path)

    return output_path


def visualize_mock(
    output_path: str | None = None,
    width: int = DEFAULT_SVG_WIDTH,
    height: int = DEFAULT_SVG_HEIGHT,
    layout: str = LAYOUT_ALGO,
    title: str = "RAG Retrieval Graph · Mock",
    n_chunks: int = 6,
    n_connections: int = 4,
    query_idx: int = 0,
    seed: int = LAYOUT_SEED,
    export_report_path: str | None = None,
    cluster: bool = False,
) -> str:
    """
    Generate a synthetic RAG graph without needing a log file.

    All parameters have sensible defaults — useful for demos, CI, and testing.
    Returns the path to the output SVG file.
    """
    global LAYOUT_ALGO
    LAYOUT_ALGO = layout

    data = make_mock_data(
        n_chunks=n_chunks,
        n_connections=n_connections,
        query_idx=query_idx,
        seed=seed,
    )
    G = build_graph(data, cluster=cluster)

    if output_path is None:
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "mock_graph.svg")

    pos = compute_layout(G, width, height)
    render_svg(G, pos, output_path, width=width, height=height, title=title)

    if export_report_path:
        export_report(G, data, export_report_path)

    return output_path


def visualize_diff(
    log1_path: str,
    log2_path: str,
    output_path: str | None = None,
    width: int = DEFAULT_SVG_WIDTH,
    height: int = DEFAULT_SVG_HEIGHT,
    layout: str = LAYOUT_ALGO,
    title: str | None = None,
    change_threshold: float = 0.05,
) -> str:
    """
    Compare two RAG logs and render a diff SVG showing what changed.

    Added chunks get a green glow, removed get a red glow (dimmed),
    and chunks whose score changed by ≥ change_threshold get an amber glow
    with the score delta printed inside the node.

    Returns the path to the output diff SVG.
    """
    global LAYOUT_ALGO
    LAYOUT_ALGO = layout

    data1 = parse_log_file(log1_path)
    data2 = parse_log_file(log2_path)

    merged_data, diff_map, score_delta = diff_logs(data1, data2, change_threshold)
    G = build_graph(merged_data)

    if output_path is None:
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem1 = Path(log1_path).stem
        stem2 = Path(log2_path).stem
        output_path = str(out_dir / f"diff_{stem1}_vs_{stem2}.svg")

    if title is None:
        q = merged_data.get("query", "")[:40]
        title = f'RAG Diff · "{q}"' if q else "RAG Retrieval Diff"

    pos = compute_layout(G, width, height)
    render_svg(
        G, pos, output_path,
        width=width, height=height, title=title,
        diff_map=diff_map, score_delta=score_delta,
    )
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt-Graph: Visualize RAG retrieval paths as SVG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"Prompt-Graph {VERSION}",
    )

    # Input source — mutually exclusive: real log or mock
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--log",
        default=None,
        help="Path to the RAG log file (JSONL or plain text)",
    )
    input_group.add_argument(
        "--mock",
        action="store_true",
        help="Generate a synthetic graph without a real log file",
    )

    parser.add_argument(
        "--diff",
        metavar="LOG2",
        default=None,
        help="Compare --log against this second log and render a diff SVG",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output SVG path (default: outputs/<stem>_graph.svg)",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_SVG_WIDTH,
                        help="SVG canvas width in pixels")
    parser.add_argument("--height", type=int, default=DEFAULT_SVG_HEIGHT,
                        help="SVG canvas height in pixels")
    parser.add_argument(
        "--layout",
        choices=["spring", "kamada_kawai", "circular", "shell"],
        default=LAYOUT_ALGO,
        help="Graph layout algorithm",
    )
    parser.add_argument("--title", default="RAG Retrieval Graph", help="Graph title")
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print graph statistics to stdout",
    )
    parser.add_argument(
        "--export-report",
        action="store_true",
        help="Write a JSON stats report alongside the SVG",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Detect chunk communities and color-code by cluster",
    )
    # Mock-mode options
    parser.add_argument("--n-chunks", type=int, default=6,
                        help="Number of chunks in mock mode")
    parser.add_argument("--n-connections", type=int, default=4,
                        help="Number of chunk-to-chunk connections in mock mode")
    parser.add_argument("--query-idx", type=int, default=0,
                        help="Which mock query to use (0–4)")

    args = parser.parse_args()

    if not args.mock and not args.log:
        parser.error("Provide --log <file> or --mock to generate a synthetic graph.")

    # Startup banner
    _console.print(Panel.fit(
        f"[bold #7B61FF]Prompt-Graph[/]  [dim]v{VERSION}[/]\n"
        "[cyan]Visualize RAG retrieval paths as SVG · 100% offline[/]\n"
        "[dim italic]Made autonomously by NEO · heyneo.so[/]",
        border_style="#7B61FF",
    ))

    try:
        # ── Diff mode ─────────────────────────────────────────────────────────
        if args.diff:
            if not args.log:
                parser.error("--diff requires --log to specify the baseline log.")
            out = visualize_diff(
                log1_path=args.log,
                log2_path=args.diff,
                output_path=args.output,
                width=args.width,
                height=args.height,
                layout=args.layout,
                title=args.title if args.title != "RAG Retrieval Graph" else None,
            )
            _console.print(f"[green]✓[/green] Diff SVG: {out}")

        # ── Mock mode ─────────────────────────────────────────────────────────
        elif args.mock:
            report_path = None
            if args.export_report:
                svg_out = args.output or str(Path(OUTPUT_DIR) / "mock_graph.svg")
                report_path = str(Path(svg_out).with_suffix(".json"))

            out = visualize_mock(
                output_path=args.output,
                width=args.width,
                height=args.height,
                layout=args.layout,
                title=args.title,
                n_chunks=args.n_chunks,
                n_connections=args.n_connections,
                query_idx=args.query_idx,
                export_report_path=report_path,
                cluster=args.cluster,
            )
            _console.print(f"[green]✓[/green] SVG: {out}")
            if report_path:
                _console.print(f"[green]✓[/green] Report: {report_path}")

        # ── Normal mode ───────────────────────────────────────────────────────
        else:
            report_path = None
            if args.export_report:
                svg_out = args.output
                if svg_out is None:
                    stem = Path(args.log).stem
                    svg_out = str(Path(OUTPUT_DIR) / f"{stem}_graph.svg")
                report_path = str(Path(svg_out).with_suffix(".json"))

            out = visualize(
                log_path=args.log,
                output_path=args.output,
                width=args.width,
                height=args.height,
                layout=args.layout,
                title=args.title,
                export_report_path=report_path,
                cluster=args.cluster,
            )
            _console.print(f"[green]✓[/green] SVG: {out}")
            if report_path:
                _console.print(f"[green]✓[/green] Report: {report_path}")

    except FileNotFoundError as exc:
        _err.print(f"[red]ERROR:[/red] {exc}")
        sys.exit(1)
    except Exception as exc:
        _err.print(f"[red]ERROR:[/red] {exc}")
        sys.exit(1)

    if args.print_stats and not args.diff:
        if args.mock:
            data = make_mock_data(
                n_chunks=args.n_chunks,
                n_connections=args.n_connections,
                query_idx=args.query_idx,
            )
        else:
            data = parse_log_file(args.log)
        G = build_graph(data, cluster=args.cluster)
        stats = compute_stats(G, data)
        s = stats["scores"]

        tbl = Table(title="Graph Statistics", box=rich_box.ROUNDED, border_style="dim")
        tbl.add_column("Metric", style="cyan", no_wrap=True)
        tbl.add_column("Value", style="white")
        tbl.add_row("Query", stats["query"][:80])
        tbl.add_row("Chunks", str(stats["chunk_count"]))
        tbl.add_row("Edges", str(stats["edge_count"]))
        tbl.add_row("Density", str(stats["graph_density"]))
        tbl.add_row("Avg score", f"{s['mean']:.3f}  (σ={s['stdev']:.3f})")
        tbl.add_row("Score range", f"{s['min']:.3f} – {s['max']:.3f}")
        tbl.add_row(
            "High / Med / Low",
            f"[green]{s['high_count']}[/]  /  [yellow]{s['med_count']}[/]  /  [red]{s['low_count']}[/]",
        )
        _console.print(tbl)

        if stats["top_chunks"]:
            top_tbl = Table(
                title="Top Chunks", box=rich_box.SIMPLE, border_style="dim"
            )
            top_tbl.add_column("Chunk ID", style="cyan")
            top_tbl.add_column("Score", style="white")
            top_tbl.add_column("Rank", style="yellow")
            top_tbl.add_column("Centrality", style="dim")
            for c in stats["top_chunks"]:
                rank_str = f"#{c['rank']}" if c["rank"] else "—"
                top_tbl.add_row(
                    c["chunk_id"],
                    f"{c['score']:.3f}",
                    rank_str,
                    f"{c['centrality']:.4f}",
                )
            _console.print(top_tbl)


if __name__ == "__main__":
    main()

# Make this module callable so that `import prompt_graph_visuali; prompt_graph_visuali.visualize(...)`
# works while still allowing monkeypatching of module-level constants like SCORE_HIGH_THRESHOLD.
import sys as _sys

_visualize_func = visualize  # capture the function before replacing the module class


class _CallableModule(_sys.modules[__name__].__class__):
    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return _visualize_func(*args, **kwargs)


_sys.modules[__name__].__class__ = _CallableModule
