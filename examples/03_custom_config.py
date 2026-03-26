#!/usr/bin/env python3
"""
03_custom_config.py — Customising Prompt-Graph via environment variables.

All visual settings are controlled through env vars so you never need to
edit the source. This script shows the four knobs you'll reach for most:

  SVG_WIDTH / SVG_HEIGHT   — canvas size in pixels
  LAYOUT_ALGO              — spring | kamada_kawai | circular | shell
  SCORE_HIGH_THRESHOLD     — score ≥ this → green node  (default 0.75)
  SCORE_MED_THRESHOLD      — score ≥ this → amber node  (default 0.50)
  COLOR_BG                 — background hex colour
  COLOR_QUERY              — query-node hex colour

Each configuration is applied by setting os.environ *before* importing
(or reloading) prompt_graph_visuali, because the module reads env vars at
import time.  For a real project you would set these in a .env file or
your shell profile instead.
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from importlib import import_module, reload

Path("outputs").mkdir(exist_ok=True)

# ── Helper: apply env vars, reload module, render ────────────────────────────

def _render(label: str, svg_name: str, env_overrides: dict) -> None:
    """Set env vars, reload the module so constants are re-read, then render."""
    for k, v in env_overrides.items():
        os.environ[k] = v

    # Reload so module-level constants pick up the new env vars
    import prompt_graph_visuali.visualize as _vmod
    reload(_vmod)
    import prompt_graph_visuali as _pkg
    reload(_pkg)

    from prompt_graph_visuali import make_mock_data, build_graph, compute_layout, render_svg
    data = make_mock_data(n_chunks=7, n_connections=4, seed=99)
    G = build_graph(data)
    w = int(os.environ.get("SVG_WIDTH", "1200"))
    h = int(os.environ.get("SVG_HEIGHT", "800"))
    pos = compute_layout(G, w, h)
    out = f"outputs/{svg_name}"
    render_svg(G, pos, out, width=w, height=h, title=label)
    print(f"  → {out}")

    # Clean up so later runs start fresh
    for k in env_overrides:
        os.environ.pop(k, None)


# ── Config 1: default settings ───────────────────────────────────────────────
print("Config 1: defaults")
_render("Default Config", "config_default.svg", {})

# ── Config 2: wide canvas + shell layout ─────────────────────────────────────
print("Config 2: wide canvas, shell layout")
_render("Wide Canvas — Shell Layout", "config_wide_shell.svg", {
    "SVG_WIDTH": "1600",
    "SVG_HEIGHT": "700",
    "LAYOUT_ALGO": "shell",
})

# ── Config 3: strict score thresholds (only top scores turn green) ────────────
print("Config 3: strict thresholds (high=0.90, med=0.70)")
_render("Strict Thresholds", "config_strict_thresholds.svg", {
    "SCORE_HIGH_THRESHOLD": "0.90",
    "SCORE_MED_THRESHOLD": "0.70",
})

# ── Config 4: light background ───────────────────────────────────────────────
print("Config 4: light background")
_render("Light Background", "config_light_bg.svg", {
    "COLOR_BG": "#F6F8FA",
    "COLOR_TEXT": "#24292F",
    "COLOR_QUERY": "#6F42C1",
})

print("Done — check outputs/ for the four SVG variants.")
