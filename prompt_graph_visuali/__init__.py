"""
prompt_graph_visuali — Prompt-Graph public API.

Visualize RAG retrieval paths as SVG graphs, 100% local.
"""

from .visualize import (
    VERSION,
    build_graph,
    compute_layout,
    compute_stats,
    diff_logs,
    export_report,
    make_mock_data,
    parse_log_file,
    render_svg,
    visualize_diff,
    visualize_mock,
)
# Import the submodule itself as `visualize` so that:
#   (a) monkeypatch can reach module-level constants via "prompt_graph_visuali.visualize.CONST"
#   (b) callers can still do `visualize(log_path, ...)` because the module is made callable
from . import visualize

__all__ = [
    "VERSION",
    "build_graph",
    "compute_layout",
    "compute_stats",
    "diff_logs",
    "export_report",
    "make_mock_data",
    "parse_log_file",
    "render_svg",
    "visualize",
    "visualize_diff",
    "visualize_mock",
]
