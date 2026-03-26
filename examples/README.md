# Examples

Runnable scripts that demonstrate Prompt-Graph features from minimal to full pipeline.
Each script adds `sys.path.insert` so it works from any directory:

```bash
cd <project-root>
python examples/01_quick_start.py
```

---

## Script index

| Script | What it demonstrates |
|---|---|
| `01_quick_start.py` | Minimal working example — mock mode, 5 chunks, default layout, ~15 lines |
| `02_advanced_usage.py` | Real JSONL log, cluster detection, rank badges, JSON stats report, multiple layouts |
| `03_custom_config.py` | Four env-var configurations: wide canvas, strict thresholds, light background, shell layout |
| `04_full_pipeline.py` | End-to-end two-run experiment: visualize Run A, Run B, diff both, print stats comparison |

---

## Prerequisites

```bash
pip install -r requirements.txt   # networkx svgwrite rich
```

Outputs are written to `outputs/`. The directory is created automatically.
