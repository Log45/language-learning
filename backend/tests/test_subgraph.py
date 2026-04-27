from collections.abc import Mapping

from backend.graph.subgraph import (
    GROUP_COLORS,
    allowed_rel_types,
    build_subgraph_from_word_rows,
    stable_node_id,
    _node_json,
)


class FakeNode(Mapping):
    __slots__ = ("labels", "_props")

    def __init__(self, labels: frozenset[str], props: dict):
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "_props", dict(props))

    def __getitem__(self, key):
        return self._props[key]

    def __iter__(self):
        return iter(self._props)

    def __len__(self):
        return len(self._props)


class FakeRel:
    __slots__ = ("type", "start_node", "end_node")

    def __init__(self, rel_type: str, start_node, end_node):
        self.type = rel_type
        self.start_node = start_node
        self.end_node = end_node


def test_node_json_accepts_neo4j_style_dict():
    node = {
        "element_id": "4:x",
        "labels": ["Word"],
        "properties": {"id": 7, "kanji": "猫", "kana": "ねこ", "romaji": "", "jlptLevel": 5, "isCommon": False},
    }
    j = _node_json(node)
    assert j is not None
    assert j["id"] == "word:7"
    assert "猫" in j["label"]


def test_stable_node_id_word_kanji_reading():
    w = frozenset({"Word"})
    assert stable_node_id(w, {"id": 42}) == "word:42"
    k = frozenset({"Kanji"})
    assert stable_node_id(k, {"character": "生"}) == "kanji:生"
    r = frozenset({"Reading"})
    assert stable_node_id(r, {"value": "せい", "type": "onyomi"}) == "reading:onyomi:せい"


def test_allowed_rel_types_excludes():
    out = allowed_rel_types(["HAS_TAG", "CO_OCCURS_WITH"])
    assert "HAS_TAG" not in out
    assert "CO_OCCURS_WITH" not in out
    assert "CONTAINS" in out


def test_group_colors_cover_known_labels():
    for label in ("Word", "Kanji", "Reading", "Meaning", "Tag"):
        assert label in GROUP_COLORS


def test_build_subgraph_from_word_rows_one_hop():
    center = FakeNode(frozenset({"Word"}), {"id": 1, "kanji": "学生", "kana": "がくせい"})
    k = FakeNode(frozenset({"Kanji"}), {"character": "学"})
    r = FakeRel("CONTAINS", center, k)
    rows = [{"center": center, "r": r, "n": k}]
    data = build_subgraph_from_word_rows(rows, max_nodes=150)
    assert data["center_id"] == "word:1"
    ids = {n["id"] for n in data["nodes"]}
    assert "word:1" in ids and "kanji:学" in ids
    assert len(data["links"]) == 1
    assert data["links"][0]["type"] == "CONTAINS"


def test_build_subgraph_truncates_nodes():
    center = FakeNode(frozenset({"Word"}), {"id": 1, "kanji": "多", "kana": "た"})
    rows = [{"center": center, "r": None, "n": None}]
    for i in range(2, 30):
        k = FakeNode(frozenset({"Kanji"}), {"character": str(i)})
        rel = FakeRel("CONTAINS", center, k)
        rows.append({"center": center, "r": rel, "n": k})
    data = build_subgraph_from_word_rows(rows, max_nodes=5)
    assert len(data["nodes"]) <= 5
    assert all(L["source"] in {n["id"] for n in data["nodes"]} for L in data["links"])


def test_neo4j_subgraph_word_smoke_optional():
    """Set NEO4J_SMOKE=1 with Neo4j running to exercise real Bolt + Cypher."""
    import os

    if os.environ.get("NEO4J_SMOKE") != "1":
        import pytest

        pytest.skip("set NEO4J_SMOKE=1 to run")

    from backend.db.neo4j_session import driver
    from backend.graph.client import GraphClient
    from backend.graph.subgraph import get_subgraph_for_word

    if driver is None:
        import pytest

        pytest.skip("neo4j driver not installed")

    client = GraphClient(driver)
    try:
        data = get_subgraph_for_word(client, word_id=1, edge_limit=30, max_nodes=80)
    except Exception as exc:  # noqa: BLE001
        import pytest

        pytest.skip(f"neo4j unavailable: {exc}")
    assert "nodes" in data and "links" in data
    assert data.get("center_id") == "word:1" or len(data["nodes"]) == 0
