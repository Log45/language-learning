import logging
from typing import Any

from backend.db.neo4j_session import driver as neo4j_driver
from backend.graph import GraphClient, repository


LOGGER = logging.getLogger(__name__)


def get_word_enrichment(word_id: int, limit: int = 5) -> dict[str, Any]:
    if neo4j_driver is None:
        return {"related_words": [], "readings": [], "kanji": []}
    client = GraphClient(neo4j_driver)
    try:
        related_words = client.execute_read(repository.get_related_words, word_id=word_id, limit=limit)
        readings = client.execute_read(repository.get_word_readings, word_id=word_id)
        kanji = client.execute_read(repository.get_word_kanji, word_id=word_id)
        return {
            "related_words": related_words,
            "readings": readings,
            "kanji": kanji,
        }
    except Exception:
        LOGGER.exception("Neo4j enrichment failed for word_id=%s", word_id)
        return {"related_words": [], "readings": [], "kanji": []}
    finally:
        client.close()

