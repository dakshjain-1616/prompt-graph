#!/usr/bin/env python3
"""
visualize.py — CLI entry-point for Prompt-Graph.

Core implementation lives in prompt_graph_visuali/visualize.py.
Run directly:  python visualize.py --mock
Or via package: python -m prompt_graph_visuali.visualize --mock
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from prompt_graph_visuali.visualize import main

if __name__ == "__main__":
    main()
