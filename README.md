# Prompt-Graph – Visualize RAG Retrieval Paths as SVG

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-71%20passed-brightgreen.svg)]()

> Turn opaque RAG logs into inspectable SVG graphs locally, zero API keys.

## The Problem  
RAG (Retrieval-Augmented Generation) systems often operate as black boxes, making it difficult for developers to debug or optimize retrieval paths. Existing tools lack a way to visually trace which chunks were retrieved and how they connect, leaving developers to rely on verbose logs or manual inspection of embeddings. This opacity slows down debugging, fine-tuning, and understanding of RAG workflows.

## Who it's for  
This tool is for developers building or debugging RAG systems who need a clear, visual representation of retrieval paths. For example, a developer fine-tuning a RAG pipeline for a chatbot might use this to identify why irrelevant chunks are being retrieved and adjust their embedding strategy accordingly.


## Install

```bash
git clone https://github.com/dakshjain-1616/prompt-graph
cd prompt-graph
pip install -r requirements.txt
```

## Quickstart

```python
from prompt_graph_visuali.visualize import visualize_rag_log

# Generate SVG from your JSONL retrieval logs
visualize_rag_log(
    input_path="logs/retrieval.jsonl",
    output_path="output/retrieval_graph.svg"
)
```

## Key features

- **100% Local Execution:** No data leaves your machine; no API keys or cloud accounts required.
- **RAG-Specific Semantics:** Nodes colored by similarity scores, edges encode chunk relationships.
- **Vector Output:** Generates crisp SVG files that scale infinitely for reports or debugging.
- **Single File Simplicity:** Entire pipeline runs via one script with minimal dependencies.

## Run tests

```bash
pytest tests/ -q
# 71 passed
```

## Project structure

```
prompt-graph/
├── prompt_graph_visuali/  ← main library
├── tests/                 ← test suite
├── scripts/               ← demo scripts
├── examples/              ← usage examples
└── requirements.txt
```