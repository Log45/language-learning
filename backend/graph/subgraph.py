"""Build bounded {nodes, links} payloads for graph visualization."""

from __future__ import annotations

import hashlib
from typing import Any

from backend.graph import queries
from backend.graph.client import GraphClient

DEFAULT_REL_TYPES: frozenset[str] = frozenset(
    {"CONTAINS", "HAS_MEANING", "HAS_TAG", "HAS_READING", "CO_OCCURS_WITH"}
)

# vis-network / legend friendly (hex, no theme-specific styling in DB)
GROUP_COLORS: dict[str, str] = {
    "Word": "#3498db",
    "Kanji": "#e74c3c",
    "Reading": "#9b59b6",
    "Meaning": "#2ecc71",
    "Tag": "#f39c12",
    "Unknown": "#95a5a6",
}


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def allowed_rel_types(exclude_rel_types: list[str] | None) -> list[str]:
    blocked = frozenset(exclude_rel_types or ())
    return sorted(t for t in DEFAULT_REL_TYPES if t not in blocked)


def stable_node_id(labels: frozenset[str], props: dict[str, Any]) -> str:
    if "Word" in labels and props.get("id") is not None:
        return f"word:{int(props['id'])}"
    if "Kanji" in labels and props.get("character") is not None:
        return f"kanji:{props['character']}"
    if "Reading" in labels:
        v = props.get("value", "")
        t = props.get("type", "")
        return f"reading:{t}:{v}"
    if "Meaning" in labels:
        text = props.get("text", "")
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"meaning:{h}"
    if "Tag" in labels and props.get("name") is not None:
        return f"tag:{props['name']}"
    raw = "|".join(sorted(labels)) + str(sorted(props.items()))
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"other:{h}"


def _node_props(entity: Any) -> dict[str, Any]:
    if entity is None:
        return {}
    if isinstance(entity, dict):
        inner = entity.get("properties")
        if isinstance(inner, dict):
            return dict(inner)
        skip = frozenset(
            (
                "labels",
                "element_id",
                "elementId",
                "identity",
                "_element_id",
            )
        )
        return {k: v for k, v in entity.items() if k not in skip}
    items = getattr(entity, "items", None)
    if callable(items):
        return dict(items())
    return dict(entity)


def _node_labels(entity: Any) -> frozenset[str]:
    if entity is None:
        return frozenset()
    if isinstance(entity, dict):
        labs = entity.get("labels")
        if isinstance(labs, (list, tuple)):
            return frozenset(str(x) for x in labs)
        return frozenset()
    return frozenset(entity.labels)


def _rel_type(rel: Any) -> str:
    if rel is None:
        return ""
    if isinstance(rel, dict):
        return str(rel.get("type", ""))
    return str(rel.type)


def _rel_endpoints(rel: Any) -> tuple[Any, Any] | None:
    if rel is None:
        return None
    if hasattr(rel, "start_node") and hasattr(rel, "end_node"):
        return rel.start_node, rel.end_node
    if isinstance(rel, dict):
        if "start_node" in rel and "end_node" in rel:
            return rel["start_node"], rel["end_node"]
        if "start" in rel and "end" in rel:
            return rel["start"], rel["end"]
        nodes = rel.get("nodes")
        if isinstance(nodes, (list, tuple)) and len(nodes) == 2:
            return nodes[0], nodes[1]
    return None


def _display_label(labels: frozenset[str], props: dict[str, Any]) -> str:
    if "Word" in labels:
        k = props.get("kanji") or ""
        ka = props.get("kana") or ""
        return f"{k} ({ka})".strip() or f"Word#{props.get('id')}"
    if "Kanji" in labels:
        return str(props.get("character", ""))
    if "Reading" in labels:
        return str(props.get("value", ""))
    if "Meaning" in labels:
        t = str(props.get("text", ""))
        return t if len(t) <= 48 else t[:45] + "..."
    if "Tag" in labels:
        return str(props.get("name", ""))
    return ",".join(sorted(labels)) or "node"


def _primary_group(labels: frozenset[str]) -> str:
    for preferred in ("Word", "Kanji", "Reading", "Meaning", "Tag"):
        if preferred in labels:
            return preferred
    return next(iter(labels), "Unknown")


def _node_json(entity: Any) -> dict[str, Any] | None:
    if entity is None:
        return None
    labels = _node_labels(entity)
    props = _node_props(entity)
    nid = stable_node_id(labels, props)
    group = _primary_group(labels)
    return {
        "id": nid,
        "label": _display_label(labels, props),
        "group": group,
        "color": GROUP_COLORS.get(group, GROUP_COLORS["Unknown"]),
    }


def _truncate_graph(
    nodes: dict[str, dict[str, Any]],
    links: list[dict[str, Any]],
    center_key: str,
    max_nodes: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    max_nodes = _clamp_int(max_nodes, 2, 150)
    if len(nodes) <= max_nodes:
        return nodes, links
    others = sorted(k for k in nodes if k != center_key)
    keep = {center_key, *others[: max_nodes - 1]}
    new_nodes = {k: nodes[k] for k in nodes if k in keep}
    new_links = [
        L for L in links if L["source"] in keep and L["target"] in keep
    ]
    return new_nodes, new_links


def build_subgraph_from_word_rows(
    rows: list[dict[str, Any]],
    *,
    max_nodes: int = 150,
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    center_key: str | None = None

    for row in rows:
        center = row.get("center")
        if center is not None:
            cj = _node_json(center)
            if cj:
                center_key = cj["id"]
                nodes[cj["id"]] = cj

        rel = row.get("r")
        n = row.get("n")
        if rel is None or n is None:
            continue
        endpoints = _rel_endpoints(rel)
        if endpoints is None:
            continue
        start, end = endpoints
        sj = _node_json(start)
        ej = _node_json(end)
        if not sj or not ej:
            continue
        nodes[sj["id"]] = sj
        nodes[ej["id"]] = ej
        links.append(
            {
                "source": sj["id"],
                "target": ej["id"],
                "type": _rel_type(rel),
            }
        )

    if center_key is None:
        return {"nodes": [], "links": [], "center_id": None}

    nodes, links = _truncate_graph(nodes, links, center_key, max_nodes)
    return {"nodes": list(nodes.values()), "links": links, "center_id": center_key}


def build_subgraph_from_kanji_rows(
    rows: list[dict[str, Any]],
    *,
    max_nodes: int = 150,
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    center_key: str | None = None

    for row in rows:
        center = row.get("center")
        if center is not None:
            cj = _node_json(center)
            if cj:
                center_key = cj["id"]
                nodes[cj["id"]] = cj

        rel = row.get("r")
        n = row.get("n")
        if rel is None or n is None:
            continue
        endpoints = _rel_endpoints(rel)
        if endpoints is None:
            continue
        start, end = endpoints
        sj = _node_json(start)
        ej = _node_json(end)
        if not sj or not ej:
            continue
        nodes[sj["id"]] = sj
        nodes[ej["id"]] = ej
        links.append(
            {
                "source": sj["id"],
                "target": ej["id"],
                "type": _rel_type(rel),
            }
        )

    if center_key is None:
        return {"nodes": [], "links": [], "center_id": None}

    nodes, links = _truncate_graph(nodes, links, center_key, max_nodes)
    return {"nodes": list(nodes.values()), "links": links, "center_id": center_key}


def get_subgraph_for_word(
    client: GraphClient,
    *,
    word_id: int,
    edge_limit: int = 100,
    max_nodes: int = 150,
    exclude_rel_types: list[str] | None = None,
) -> dict[str, Any]:
    edge_limit = _clamp_int(edge_limit, 1, 300)
    allowed = allowed_rel_types(exclude_rel_types)
    rows = client.run(
        queries.SUBGRAPH_FROM_WORD,
        word_id=word_id,
        allowed_types=allowed,
        edge_limit=edge_limit,
    )
    out = build_subgraph_from_word_rows(rows, max_nodes=max_nodes)
    out["kind"] = "word"
    out["word_id"] = word_id
    return out


def get_subgraph_for_kanji(
    client: GraphClient,
    *,
    character: str,
    edge_limit: int = 100,
    max_nodes: int = 150,
    exclude_rel_types: list[str] | None = None,
) -> dict[str, Any]:
    edge_limit = _clamp_int(edge_limit, 1, 300)
    allowed = allowed_rel_types(exclude_rel_types)
    rows = client.run(
        queries.SUBGRAPH_FROM_KANJI,
        character=character,
        allowed_types=allowed,
        edge_limit=edge_limit,
    )
    out = build_subgraph_from_kanji_rows(rows, max_nodes=max_nodes)
    out["kind"] = "kanji"
    out["character"] = character
    return out
