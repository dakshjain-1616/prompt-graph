"""
Tests for Prompt-Graph (visualize.py).

Requirements covered:
  1. Input: RAG log file → Output: SVG graph
  2. Input: 10 chunks    → Output: 10 nodes in graph
  3. Input: Local file   → Output: 0 network calls (no socket activity)
  4. Mock mode           → SVG without any log file
  5. Stats report        → JSON export with correct schema
  6. Graph diff          → diff_logs detects added/removed/changed nodes
  7. Cluster detection   → cluster IDs assigned to chunk nodes
  8. Score thresholds    → SCORE_HIGH_THRESHOLD / SCORE_MED_THRESHOLD respected
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_graph_visuali import (
    build_graph,
    compute_layout,
    compute_stats,
    diff_logs,
    export_report,
    make_mock_data,
    parse_log_file,
    render_svg,
    visualize,
    visualize_diff,
    visualize_mock,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_jsonl_log(tmp_path: Path,
                    n_chunks: int = 5,
                    n_connections: int = 2,
                    query: str = "What is RAG?",
                    filename: str = "test.log") -> Path:
    """Write a synthetic JSONL RAG log and return its path."""
    events = [{"event": "query", "query": query}]
    for i in range(1, n_chunks + 1):
        events.append({
            "event": "retrieve",
            "chunk_id": f"chunk_{i:03d}",
            "score": round(0.5 + i / (n_chunks * 2), 3),
            "content": f"Content of chunk {i}",
            "source": f"doc{i}.txt",
        })
    for i in range(n_connections):
        events.append({
            "event": "connect",
            "from": f"chunk_{i+1:03d}",
            "to": f"chunk_{i+2:03d}",
            "weight": 0.7,
        })
    log_path = tmp_path / filename
    log_path.write_text("\n".join(json.dumps(e) for e in events))
    return log_path


def _make_plain_log(tmp_path: Path, n_chunks: int = 3) -> Path:
    """Write a plain-text RAG log."""
    lines = ["Query: Explain vector search"]
    for i in range(1, n_chunks + 1):
        lines.append(f"chunk_{i:03d} score={0.5 + i*0.1:.2f} source=wiki.md content here")
    log_path = tmp_path / "plain.log"
    log_path.write_text("\n".join(lines))
    return log_path


# ── Test 1: RAG log → SVG file ───────────────────────────────────────────────

class TestLogToSVG:
    def test_svg_file_is_created(self, tmp_path):
        log = _make_jsonl_log(tmp_path)
        out = tmp_path / "out.svg"
        result = visualize(str(log), output_path=str(out))
        assert result == str(out)
        assert out.exists(), "SVG file must be created"

    def test_svg_has_correct_extension(self, tmp_path):
        log = _make_jsonl_log(tmp_path)
        out = tmp_path / "graph.svg"
        visualize(str(log), output_path=str(out))
        assert out.suffix == ".svg"

    def test_svg_is_non_empty(self, tmp_path):
        log = _make_jsonl_log(tmp_path)
        out = tmp_path / "out.svg"
        visualize(str(log), output_path=str(out))
        assert out.stat().st_size > 0, "SVG file must not be empty"

    def test_svg_contains_xml_declaration_or_svg_tag(self, tmp_path):
        log = _make_jsonl_log(tmp_path)
        out = tmp_path / "out.svg"
        visualize(str(log), output_path=str(out))
        content = out.read_text()
        assert "<svg" in content, "Output must contain an SVG element"

    def test_svg_contains_query_text(self, tmp_path):
        log = _make_jsonl_log(tmp_path, query="TestQueryString")
        out = tmp_path / "out.svg"
        visualize(str(log), output_path=str(out))
        content = out.read_text()
        assert "TestQueryString" in content

    def test_svg_title_customisable(self, tmp_path):
        log = _make_jsonl_log(tmp_path)
        out = tmp_path / "out.svg"
        visualize(str(log), output_path=str(out), title="CustomTitle")
        content = out.read_text()
        assert "CustomTitle" in content

    def test_missing_log_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            visualize(str(tmp_path / "nonexistent.log"),
                      output_path=str(tmp_path / "out.svg"))

    def test_plain_text_log_produces_svg(self, tmp_path):
        log = _make_plain_log(tmp_path, n_chunks=3)
        out = tmp_path / "plain.svg"
        visualize(str(log), output_path=str(out))
        assert out.exists()
        assert "<svg" in out.read_text()


# ── Test 2: 10 chunks → 10 nodes ─────────────────────────────────────────────

class TestTenChunks:
    def test_parse_returns_10_chunks(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=10)
        data = parse_log_file(str(log))
        assert len(data["nodes"]) == 10

    def test_graph_has_10_chunk_nodes(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=10)
        data = parse_log_file(str(log))
        G = build_graph(data)
        chunk_nodes = [n for n, d in G.nodes(data=True)
                       if d.get("node_type") == "chunk"]
        assert len(chunk_nodes) == 10

    def test_graph_total_nodes_is_11(self, tmp_path):
        """10 chunks + 1 query node = 11 total."""
        log = _make_jsonl_log(tmp_path, n_chunks=10)
        data = parse_log_file(str(log))
        G = build_graph(data)
        assert G.number_of_nodes() == 11

    def test_svg_references_all_10_chunk_ids(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=10)
        out = tmp_path / "ten.svg"
        visualize(str(log), output_path=str(out))
        content = out.read_text()
        for i in range(1, 11):
            cid = f"chunk_{i:03d}"
            assert cid in content, f"{cid} must appear in SVG"

    def test_scores_preserved_in_graph(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=10)
        data = parse_log_file(str(log))
        G = build_graph(data)
        for _, ndata in G.nodes(data=True):
            if ndata.get("node_type") == "chunk":
                assert 0.0 <= ndata["score"] <= 1.0

    def test_edges_connect_query_to_all_chunks(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=10, n_connections=0)
        data = parse_log_file(str(log))
        G = build_graph(data)
        chunk_nodes = [n for n, d in G.nodes(data=True)
                       if d.get("node_type") == "chunk"]
        for cid in chunk_nodes:
            assert G.has_edge("__query__", cid), \
                f"Query must connect to {cid}"

    def test_layout_returns_positions_for_all_nodes(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=10)
        data = parse_log_file(str(log))
        G = build_graph(data)
        pos = compute_layout(G, width=1200, height=800)
        assert len(pos) == G.number_of_nodes()
        for node_id, (x, y) in pos.items():
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_chunk_connections_are_added_to_graph(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=10, n_connections=4)
        data = parse_log_file(str(log))
        G = build_graph(data)
        # 10 query→chunk edges + 4 chunk→chunk edges = 14
        assert G.number_of_edges() == 14

    def test_different_n_chunks_produces_correct_node_count(self, tmp_path):
        for n in [1, 3, 5, 7, 10]:
            log = _make_jsonl_log(tmp_path, n_chunks=n, filename=f"test_{n}.log")
            data = parse_log_file(str(log))
            G = build_graph(data)
            chunk_nodes = [nd for nd, d in G.nodes(data=True)
                           if d.get("node_type") == "chunk"]
            assert len(chunk_nodes) == n, f"Expected {n} chunks, got {len(chunk_nodes)}"


# ── Test 3: Local file, 0 network calls ──────────────────────────────────────

class TestNoNetworkCalls:
    def test_no_socket_connect_during_parse(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=5)
        with mock.patch("socket.socket.connect") as mock_connect:
            parse_log_file(str(log))
            mock_connect.assert_not_called()

    def test_no_socket_connect_during_build(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=5)
        data = parse_log_file(str(log))
        with mock.patch("socket.socket.connect") as mock_connect:
            build_graph(data)
            mock_connect.assert_not_called()

    def test_no_socket_connect_during_render(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=5)
        data = parse_log_file(str(log))
        G = build_graph(data)
        pos = compute_layout(G, 1200, 800)
        out = tmp_path / "no_net.svg"
        with mock.patch("socket.socket.connect") as mock_connect:
            render_svg(G, pos, str(out))
            mock_connect.assert_not_called()

    def test_no_http_requests_full_pipeline(self, tmp_path):
        """Full pipeline must make zero HTTP calls."""
        log = _make_jsonl_log(tmp_path, n_chunks=5)
        out = tmp_path / "local.svg"
        with mock.patch("socket.socket.connect") as mock_connect:
            visualize(str(log), output_path=str(out))
            mock_connect.assert_not_called()

    def test_reads_only_local_file(self, tmp_path):
        """parse_log_file must open the specified local path, nothing else."""
        log = _make_jsonl_log(tmp_path, n_chunks=3)
        opened_files = []
        real_open = open

        def tracking_open(file, *args, **kwargs):
            opened_files.append(str(file))
            return real_open(file, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=tracking_open):
            parse_log_file(str(log))

        for f in opened_files:
            assert not f.startswith("http"), \
                f"Unexpected network path opened: {f}"


# ── Test 4: Mock mode ─────────────────────────────────────────────────────────

class TestMockMode:
    def test_make_mock_data_returns_correct_structure(self):
        data = make_mock_data(n_chunks=5, n_connections=3)
        assert "query" in data
        assert "nodes" in data
        assert "edges" in data
        assert isinstance(data["query"], str)
        assert len(data["nodes"]) == 5

    def test_make_mock_data_nodes_have_required_fields(self):
        data = make_mock_data(n_chunks=4)
        for node in data["nodes"]:
            assert "chunk_id" in node
            assert "score" in node
            assert "content" in node
            assert "source" in node
            assert 0.0 <= node["score"] <= 1.0

    def test_make_mock_data_is_deterministic(self):
        data1 = make_mock_data(n_chunks=5, seed=42)
        data2 = make_mock_data(n_chunks=5, seed=42)
        assert data1["nodes"] == data2["nodes"]

    def test_make_mock_data_different_seeds_differ(self):
        data1 = make_mock_data(n_chunks=5, seed=1)
        data2 = make_mock_data(n_chunks=5, seed=99)
        scores1 = [n["score"] for n in data1["nodes"]]
        scores2 = [n["score"] for n in data2["nodes"]]
        assert scores1 != scores2

    def test_visualize_mock_creates_svg(self, tmp_path):
        out = tmp_path / "mock.svg"
        result = visualize_mock(output_path=str(out), n_chunks=5)
        assert result == str(out)
        assert out.exists()
        assert "<svg" in out.read_text()

    def test_visualize_mock_correct_node_count(self, tmp_path):
        out = tmp_path / "mock.svg"
        visualize_mock(output_path=str(out), n_chunks=7)
        data = make_mock_data(n_chunks=7)
        G = build_graph(data)
        chunk_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "chunk"]
        assert len(chunk_nodes) == 7

    def test_visualize_mock_no_network_calls(self, tmp_path):
        out = tmp_path / "mock.svg"
        with mock.patch("socket.socket.connect") as mock_connect:
            visualize_mock(output_path=str(out), n_chunks=4)
            mock_connect.assert_not_called()

    def test_make_mock_data_top3_have_ranks(self):
        data = make_mock_data(n_chunks=6)
        ranked = [n for n in data["nodes"] if "rank" in n]
        assert len(ranked) == 3
        ranks = sorted(n["rank"] for n in ranked)
        assert ranks == [1, 2, 3]

    def test_make_mock_data_query_idx_cycles(self):
        # query_idx wraps around the mock query list (5 entries)
        data0 = make_mock_data(query_idx=0)
        data5 = make_mock_data(query_idx=5)
        assert data0["query"] == data5["query"]


# ── Test 5: Stats report ──────────────────────────────────────────────────────

class TestStatsReport:
    def test_compute_stats_returns_required_keys(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=5)
        data = parse_log_file(str(log))
        G = build_graph(data)
        stats = compute_stats(G, data)
        for key in ("generated_at", "query", "chunk_count", "edge_count",
                    "graph_density", "scores", "sources", "top_chunks",
                    "layout_algo"):
            assert key in stats, f"Missing key: {key}"

    def test_compute_stats_scores_subkeys(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=5)
        data = parse_log_file(str(log))
        G = build_graph(data)
        stats = compute_stats(G, data)
        for key in ("mean", "median", "stdev", "min", "max",
                    "high_count", "med_count", "low_count"):
            assert key in stats["scores"], f"Missing score key: {key}"

    def test_compute_stats_chunk_count_correct(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=7)
        data = parse_log_file(str(log))
        G = build_graph(data)
        stats = compute_stats(G, data)
        assert stats["chunk_count"] == 7

    def test_compute_stats_high_med_low_sum(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=8)
        data = parse_log_file(str(log))
        G = build_graph(data)
        stats = compute_stats(G, data)
        s = stats["scores"]
        assert s["high_count"] + s["med_count"] + s["low_count"] == 8

    def test_compute_stats_sources_populated(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=4)
        data = parse_log_file(str(log))
        G = build_graph(data)
        stats = compute_stats(G, data)
        assert len(stats["sources"]) > 0

    def test_compute_stats_generated_at_is_utc_iso(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=3)
        data = parse_log_file(str(log))
        G = build_graph(data)
        stats = compute_stats(G, data)
        ts = stats["generated_at"]
        assert ts.endswith("Z"), "Timestamp must end with Z (UTC)"
        assert "T" in ts

    def test_export_report_writes_json(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=4)
        data = parse_log_file(str(log))
        G = build_graph(data)
        report_path = tmp_path / "report.json"
        result = export_report(G, data, str(report_path))
        assert result == str(report_path)
        assert report_path.exists()

    def test_export_report_valid_json(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=4)
        data = parse_log_file(str(log))
        G = build_graph(data)
        report_path = tmp_path / "report.json"
        export_report(G, data, str(report_path))
        parsed = json.loads(report_path.read_text())
        assert isinstance(parsed, dict)
        assert "chunk_count" in parsed

    def test_visualize_with_export_report(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=5)
        out_svg = tmp_path / "out.svg"
        out_json = tmp_path / "out.json"
        visualize(str(log), output_path=str(out_svg),
                  export_report_path=str(out_json))
        assert out_svg.exists()
        assert out_json.exists()
        parsed = json.loads(out_json.read_text())
        assert parsed["chunk_count"] == 5

    def test_visualize_mock_with_export_report(self, tmp_path):
        out_svg = tmp_path / "mock.svg"
        out_json = tmp_path / "mock.json"
        visualize_mock(output_path=str(out_svg), n_chunks=4,
                       export_report_path=str(out_json))
        assert out_json.exists()
        parsed = json.loads(out_json.read_text())
        assert parsed["chunk_count"] == 4


# ── Test 6: Graph diff ────────────────────────────────────────────────────────

class TestGraphDiff:
    def _make_data(self, chunks: dict[str, float], query="q") -> dict:
        """Build a minimal data dict from {chunk_id: score}."""
        return {
            "query": query,
            "nodes": [{"chunk_id": k, "score": v, "content": k, "source": ""}
                      for k, v in chunks.items()],
            "edges": [],
        }

    def test_diff_added_chunk_detected(self):
        data1 = self._make_data({"c1": 0.9, "c2": 0.7})
        data2 = self._make_data({"c1": 0.9, "c2": 0.7, "c3": 0.8})
        _, diff_map, _ = diff_logs(data1, data2)
        assert diff_map["c3"] == "added"

    def test_diff_removed_chunk_detected(self):
        data1 = self._make_data({"c1": 0.9, "c2": 0.7})
        data2 = self._make_data({"c1": 0.9})
        _, diff_map, _ = diff_logs(data1, data2)
        assert diff_map["c2"] == "removed"

    def test_diff_changed_chunk_detected(self):
        data1 = self._make_data({"c1": 0.9})
        data2 = self._make_data({"c1": 0.5})  # delta = -0.4 ≥ 0.05
        _, diff_map, _ = diff_logs(data1, data2)
        assert diff_map["c1"] == "changed"

    def test_diff_unchanged_chunk(self):
        data1 = self._make_data({"c1": 0.9})
        data2 = self._make_data({"c1": 0.91})  # delta = 0.01 < 0.05
        _, diff_map, _ = diff_logs(data1, data2)
        assert diff_map["c1"] == "unchanged"

    def test_diff_score_delta_positive_for_added(self):
        data1 = self._make_data({})
        data2 = self._make_data({"c1": 0.8})
        _, _, score_delta = diff_logs(data1, data2)
        assert score_delta["c1"] > 0

    def test_diff_score_delta_negative_for_removed(self):
        data1 = self._make_data({"c1": 0.8})
        data2 = self._make_data({})
        _, _, score_delta = diff_logs(data1, data2)
        assert score_delta["c1"] < 0

    def test_diff_merged_data_contains_all_nodes(self):
        data1 = self._make_data({"c1": 0.9, "c2": 0.7})
        data2 = self._make_data({"c2": 0.8, "c3": 0.6})
        merged, _, _ = diff_logs(data1, data2)
        ids = {n["chunk_id"] for n in merged["nodes"]}
        assert ids == {"c1", "c2", "c3"}

    def test_diff_merged_uses_data2_values(self):
        data1 = self._make_data({"c1": 0.5})
        data2 = self._make_data({"c1": 0.9})
        merged, _, _ = diff_logs(data1, data2)
        c1_node = next(n for n in merged["nodes"] if n["chunk_id"] == "c1")
        assert c1_node["score"] == pytest.approx(0.9)

    def test_visualize_diff_creates_svg(self, tmp_path):
        log1 = _make_jsonl_log(tmp_path, n_chunks=4, filename="v1.log")
        log2 = _make_jsonl_log(tmp_path, n_chunks=5, filename="v2.log",
                               query="Different query")
        out = tmp_path / "diff.svg"
        result = visualize_diff(str(log1), str(log2), output_path=str(out))
        assert result == str(out)
        assert out.exists()
        assert "<svg" in out.read_text()

    def test_visualize_diff_svg_contains_diff_mode_marker(self, tmp_path):
        log1 = _make_jsonl_log(tmp_path, n_chunks=3, filename="d1.log")
        log2 = _make_jsonl_log(tmp_path, n_chunks=3, filename="d2.log")
        out = tmp_path / "diff.svg"
        visualize_diff(str(log1), str(log2), output_path=str(out))
        content = out.read_text()
        assert "DIFF" in content

    def test_diff_custom_change_threshold(self):
        data1 = self._make_data({"c1": 0.9})
        data2 = self._make_data({"c1": 0.92})  # delta = 0.02
        # With threshold=0.01, small delta is "changed"
        _, diff_map_strict, _ = diff_logs(data1, data2, change_threshold=0.01)
        assert diff_map_strict["c1"] == "changed"
        # With default threshold=0.05, same delta is "unchanged"
        _, diff_map_default, _ = diff_logs(data1, data2, change_threshold=0.05)
        assert diff_map_default["c1"] == "unchanged"


# ── Test 7: Cluster detection ─────────────────────────────────────────────────

class TestClusterDetection:
    def test_build_graph_cluster_false_no_clusters(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=5, n_connections=3)
        data = parse_log_file(str(log))
        G = build_graph(data, cluster=False)
        for _, ndata in G.nodes(data=True):
            assert ndata.get("cluster") is None

    def test_build_graph_cluster_true_assigns_ids(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=6, n_connections=4)
        data = parse_log_file(str(log))
        G = build_graph(data, cluster=True)
        chunk_clusters = [
            ndata.get("cluster")
            for _, ndata in G.nodes(data=True)
            if ndata.get("node_type") == "chunk"
        ]
        # At least some chunks should have a cluster ID (int) assigned
        assigned = [c for c in chunk_clusters if c is not None]
        assert len(assigned) >= 1

    def test_build_graph_cluster_query_node_has_none(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=5, n_connections=3)
        data = parse_log_file(str(log))
        G = build_graph(data, cluster=True)
        assert G.nodes["__query__"]["cluster"] is None

    def test_cluster_ids_are_integers(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=6, n_connections=4)
        data = parse_log_file(str(log))
        G = build_graph(data, cluster=True)
        for _, ndata in G.nodes(data=True):
            c = ndata.get("cluster")
            if c is not None:
                assert isinstance(c, int)

    def test_build_graph_cluster_no_connections_graceful(self, tmp_path):
        """Clustering with no inter-chunk edges should not crash."""
        log = _make_jsonl_log(tmp_path, n_chunks=5, n_connections=0)
        data = parse_log_file(str(log))
        G = build_graph(data, cluster=True)  # must not raise
        assert G.number_of_nodes() == 6

    def test_visualize_with_cluster_creates_svg(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=6, n_connections=4)
        out = tmp_path / "clustered.svg"
        visualize(str(log), output_path=str(out), cluster=True)
        assert out.exists()
        assert "<svg" in out.read_text()

    def test_visualize_mock_with_cluster_creates_svg(self, tmp_path):
        out = tmp_path / "mock_cluster.svg"
        visualize_mock(output_path=str(out), n_chunks=7, n_connections=5,
                       cluster=True)
        assert out.exists()
        assert "<svg" in out.read_text()


# ── Test 8: Score thresholds ──────────────────────────────────────────────────

class TestScoreThresholds:
    def test_compute_stats_respects_high_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setattr("prompt_graph_visuali.visualize.SCORE_HIGH_THRESHOLD", 0.9)
        monkeypatch.setattr("prompt_graph_visuali.visualize.SCORE_MED_THRESHOLD", 0.6)
        data = make_mock_data(n_chunks=6, seed=42)
        G = build_graph(data)
        stats = compute_stats(G, data)
        # Recount manually
        scores = [d["score"] for _, d in G.nodes(data=True)
                  if d.get("node_type") == "chunk"]
        expected_high = sum(1 for s in scores if s >= 0.9)
        assert stats["scores"]["high_count"] == expected_high

    def test_compute_stats_thresholds_in_report(self, tmp_path):
        data = make_mock_data(n_chunks=4, seed=1)
        G = build_graph(data)
        stats = compute_stats(G, data)
        assert "high_threshold" in stats["scores"]
        assert "med_threshold" in stats["scores"]

    def test_high_med_low_always_sum_to_chunk_count(self, tmp_path):
        for n in [3, 7, 10]:
            log = _make_jsonl_log(tmp_path, n_chunks=n, filename=f"t_{n}.log")
            data = parse_log_file(str(log))
            G = build_graph(data)
            stats = compute_stats(G, data)
            s = stats["scores"]
            total = s["high_count"] + s["med_count"] + s["low_count"]
            assert total == n, f"Expected {n}, got {total}"


# ── Additional: parse correctness ────────────────────────────────────────────

class TestParseCorrectness:
    def test_query_extracted(self, tmp_path):
        log = _make_jsonl_log(tmp_path, query="How does BM25 work?")
        data = parse_log_file(str(log))
        assert data["query"] == "How does BM25 work?"

    def test_rerank_events_attached_to_nodes(self, tmp_path):
        events = [
            {"event": "query", "query": "test"},
            {"event": "retrieve", "chunk_id": "c1", "score": 0.9,
             "content": "text", "source": "doc.txt"},
            {"event": "rerank", "chunk_id": "c1", "rank": 1},
        ]
        log = tmp_path / "rerank.log"
        log.write_text("\n".join(json.dumps(e) for e in events))
        data = parse_log_file(str(log))
        G = build_graph(data)
        assert G.nodes["c1"].get("rank") == 1

    def test_explicit_connections_parsed(self, tmp_path):
        events = [
            {"event": "query", "query": "q"},
            {"event": "retrieve", "chunk_id": "a", "score": 0.8,
             "content": "A", "source": ""},
            {"event": "retrieve", "chunk_id": "b", "score": 0.7,
             "content": "B", "source": ""},
            {"event": "connect", "from": "a", "to": "b", "weight": 0.6},
        ]
        log = tmp_path / "conn.log"
        log.write_text("\n".join(json.dumps(e) for e in events))
        data = parse_log_file(str(log))
        assert len(data["edges"]) == 1
        assert data["edges"][0]["from"] == "a"
        assert data["edges"][0]["to"] == "b"
        assert data["edges"][0]["weight"] == pytest.approx(0.6)

    def test_layout_positions_within_canvas(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=5)
        data = parse_log_file(str(log))
        G = build_graph(data)
        width, height = 1200, 800
        pos = compute_layout(G, width, height)
        for node_id, (x, y) in pos.items():
            assert 0 <= x <= width, f"x={x} out of canvas for {node_id}"
            assert 0 <= y <= height, f"y={y} out of canvas for {node_id}"

    def test_svg_contains_legend(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=3)
        out = tmp_path / "leg.svg"
        visualize(str(log), output_path=str(out))
        content = out.read_text()
        assert "Legend" in content

    def test_svg_contains_score_labels(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=3)
        out = tmp_path / "score.svg"
        visualize(str(log), output_path=str(out))
        content = out.read_text()
        assert "score:" in content

    def test_svg_contains_timestamp(self, tmp_path):
        log = _make_jsonl_log(tmp_path, n_chunks=3)
        out = tmp_path / "ts.svg"
        visualize(str(log), output_path=str(out))
        content = out.read_text()
        assert "Generated" in content

    def test_malformed_json_lines_skipped_gracefully(self, tmp_path):
        """Logs with some bad JSON lines should not crash — valid lines are parsed."""
        events = [
            '{"event": "query", "query": "What is RAG?"}',
            "THIS IS NOT JSON",
            '{"event": "retrieve", "chunk_id": "c1", "score": 0.9, "content": "x", "source": ""}',
            "ALSO BAD",
        ]
        log = tmp_path / "mixed.log"
        log.write_text("\n".join(events))
        data = parse_log_file(str(log))
        assert data["query"] == "What is RAG?"
        assert len(data["nodes"]) == 1

    def test_missing_log_error_message_includes_tip(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="--mock"):
            parse_log_file(str(tmp_path / "missing.log"))
