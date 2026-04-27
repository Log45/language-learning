from backend.db.session import SessionLocal
from backend.db.neo4j_session import driver as neo4j_driver
from backend.graph import GraphClient
from backend.graph.ingest import ensure_graph_schema, upsert_word_graph
from backend.models import Word, Kanji


def main() -> None:
    if neo4j_driver is None:
        print("Neo4j driver is unavailable. Install the neo4j package first.")
        return

    session = SessionLocal()
    graph_client = GraphClient(neo4j_driver)

    try:
        ensure_graph_schema(graph_client)
        words = session.query(Word).all()
        for word in words:
            kanji_chars = [k.character for k in word.kanji_chars]
            kanji_lookup = {
                k.character: {
                    "meaning": k.meaning,
                    "onyomi": k.onyomi or [],
                    "kunyomi": k.kunyomi or [],
                }
                for k in session.query(Kanji).filter(Kanji.character.in_(kanji_chars)).all()
            } if kanji_chars else {}

            upsert_word_graph(
                graph_client,
                word=word,
                meanings=[m.meaning for m in word.meanings],
                tag_names=[t.name for t in word.tags],
                kanji_chars=kanji_chars,
                kanji_lookup=kanji_lookup,
            )
        print(f"Backfilled {len(words)} words into Neo4j.")
    finally:
        session.close()
        graph_client.close()


if __name__ == "__main__":
    main()

