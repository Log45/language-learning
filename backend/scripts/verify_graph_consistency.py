from backend.db.session import SessionLocal
from backend.db.neo4j_session import driver as neo4j_driver
from backend.graph import GraphClient
from backend.models import Word, Meaning, Tag, Kanji, WordKanji


def _sql_counts(session) -> dict[str, int]:
    return {
        "words": session.query(Word).count(),
        "meanings": session.query(Meaning).count(),
        "tags": session.query(Tag).count(),
        "kanji": session.query(Kanji).count(),
        "contains": session.query(WordKanji).count(),
    }


def _graph_counts(client: GraphClient) -> dict[str, int]:
    queries = {
        "words": "MATCH (w:Word) RETURN count(w) AS count",
        "meanings": "MATCH (m:Meaning) RETURN count(m) AS count",
        "tags": "MATCH (t:Tag) RETURN count(t) AS count",
        "kanji": "MATCH (k:Kanji) RETURN count(k) AS count",
        "contains": "MATCH (:Word)-[r:CONTAINS]->(:Kanji) RETURN count(r) AS count",
    }
    counts: dict[str, int] = {}
    for key, query in queries.items():
        rows = client.run(query)
        counts[key] = int(rows[0]["count"]) if rows else 0
    return counts


def main() -> None:
    if neo4j_driver is None:
        print("Neo4j driver is unavailable. Install the neo4j package first.")
        return

    session = SessionLocal()
    client = GraphClient(neo4j_driver)
    try:
        sql = _sql_counts(session)
        graph = _graph_counts(client)
        print("SQL counts:", sql)
        print("Graph counts:", graph)

        mismatches = []
        for key in sql:
            if sql[key] != graph[key]:
                mismatches.append((key, sql[key], graph[key]))

        if mismatches:
            print("Mismatches found:")
            for key, sql_count, graph_count in mismatches:
                print(f"- {key}: sql={sql_count}, graph={graph_count}")
            raise SystemExit(1)

        print("Counts are consistent.")
    finally:
        session.close()
        client.close()


if __name__ == "__main__":
    main()

