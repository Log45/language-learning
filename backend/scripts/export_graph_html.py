"""
Export a bounded knowledge-graph neighborhood to a standalone PyVis HTML file.

Examples (from repo root, PYTHONPATH includes project root):
  PYTHONPATH=. python backend/scripts/export_graph_html.py --word-id 1 --out /tmp/word_1.html
  PYTHONPATH=. python backend/scripts/export_graph_html.py --kanji 生 --out /tmp/kanji.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.db.neo4j_session import driver as neo4j_driver
from backend.graph.client import GraphClient
from backend.graph.subgraph import get_subgraph_for_kanji, get_subgraph_for_word


def _build_pyvis(data: dict):
    from pyvis.network import Network

    net = Network(height="720px", width="100%", directed=True, bgcolor="#222", font_color="#eee")
    net.set_edge_smooth("dynamic")

    for n in data.get("nodes", []):
        net.add_node(
            n["id"],
            label=n.get("label", n["id"]),
            color=n.get("color", "#95a5a6"),
            title=f"{n.get('group', '')}\n{n.get('label', '')}",
        )
    for link in data.get("links", []):
        net.add_edge(
            link["source"],
            link["target"],
            title=link.get("type", ""),
            label=link.get("type", "")[:12],
        )

    return net


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Neo4j subgraph to PyVis HTML")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--word-id", type=int, help="Center node: Word.id")
    g.add_argument("--kanji", type=str, help="Center node: Kanji.character (one character)")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .html path",
    )
    parser.add_argument("--edge-limit", type=int, default=100, help="Max relationships (1-300)")
    parser.add_argument("--max-nodes", type=int, default=150, help="Max nodes after trim (2-150)")
    parser.add_argument(
        "--exclude-rel",
        type=str,
        default="",
        help="Comma-separated relationship types to exclude, e.g. HAS_TAG,CO_OCCURS_WITH",
    )
    args = parser.parse_args()

    if neo4j_driver is None:
        raise SystemExit("Neo4j driver unavailable. Install neo4j package and start Neo4j.")

    exclude = [x.strip() for x in args.exclude_rel.split(",") if x.strip()]
    client = GraphClient(neo4j_driver)
    if args.word_id is not None:
        data = get_subgraph_for_word(
            client,
            word_id=args.word_id,
            edge_limit=args.edge_limit,
            max_nodes=args.max_nodes,
            exclude_rel_types=exclude or None,
        )
    else:
        char = args.kanji.strip()
        if len(char) != 1:
            raise SystemExit("--kanji must be exactly one character")
        data = get_subgraph_for_kanji(
            client,
            character=char,
            edge_limit=args.edge_limit,
            max_nodes=args.max_nodes,
            exclude_rel_types=exclude or None,
        )

    net = _build_pyvis(data)
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out))
    print(f"Wrote {out} ({len(data.get('nodes', []))} nodes, {len(data.get('links', []))} links)")


if __name__ == "__main__":
    main()
