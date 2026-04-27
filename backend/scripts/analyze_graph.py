from backend.db.neo4j_session import driver as neo4j_driver
from backend.graph import GraphClient


KANJI_AMBIGUITY_QUERY = """
MATCH (k:Kanji)
OPTIONAL MATCH (k)-[:HAS_READING]->(r:Reading)
WITH k, count(DISTINCT r) AS reading_count
MATCH (k)<-[:CONTAINS]-(w:Word)-[:HAS_MEANING]->(m:Meaning)
RETURN k.character AS kanji, reading_count, count(DISTINCT m) AS meaning_count,
       (reading_count + count(DISTINCT m)) AS ambiguity_score
ORDER BY ambiguity_score DESC
LIMIT $limit
"""

KANJI_CENTRALITY_PROXY_QUERY = """
MATCH (k:Kanji)<-[:CONTAINS]-(w:Word)
RETURN k.character AS kanji, count(DISTINCT w) AS word_degree
ORDER BY word_degree DESC
LIMIT $limit
"""

KANJI_COMMUNITY_PROXY_QUERY = """
MATCH (k1:Kanji)-[c:CO_OCCURS_WITH]->(k2:Kanji)
RETURN k1.character AS source, k2.character AS target, c.count AS weight
ORDER BY c.count DESC
LIMIT $limit
"""


def main(limit: int = 20) -> None:
    if neo4j_driver is None:
        print("Neo4j driver is unavailable. Install the neo4j package first.")
        return

    client = GraphClient(neo4j_driver)
    try:
        print("Top kanji ambiguity scores")
        for row in client.run(KANJI_AMBIGUITY_QUERY, limit=limit):
            print(row)

        print("\nCentrality proxy (word degree)")
        for row in client.run(KANJI_CENTRALITY_PROXY_QUERY, limit=limit):
            print(row)

        print("\nCo-occurrence community proxy edges")
        for row in client.run(KANJI_COMMUNITY_PROXY_QUERY, limit=limit):
            print(row)
    finally:
        client.close()


if __name__ == "__main__":
    main()

