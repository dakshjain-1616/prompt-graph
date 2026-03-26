#!/usr/bin/env python3
"""
01_quick_start.py — Minimal Prompt-Graph example.

Generates a synthetic RAG retrieval graph and saves it to outputs/quick_start.svg.
No log file required — uses mock mode with sensible defaults.
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from prompt_graph_visuali import visualize_mock

output = visualize_mock(
    output_path="outputs/quick_start.svg",
    n_chunks=5,
    n_connections=3,
    title="Quick Start — RAG Retrieval Graph",
)

print(f"SVG written to: {output}")
