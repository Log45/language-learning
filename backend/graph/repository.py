from typing import Any

try:
    from neo4j import ManagedTransaction
except ImportError:  # pragma: no cover
    ManagedTransaction = Any

from backend.graph import queries


def _run(tx: ManagedTransaction, query: str, **params: Any) -> list[dict[str, Any]]:
    result = tx.run(query, **params)
    return [record.data() for record in result]


def create_constraints(tx: ManagedTransaction) -> None:
    for statement in queries.CREATE_CONSTRAINTS:
        tx.run(statement)


def upsert_word(
    tx: ManagedTransaction,
    *,
    word_id: int,
    kanji: str,
    kana: str,
    romaji: str,
    jlpt_level: int,
    is_common: bool,
) -> None:
    tx.run(
        queries.UPSERT_WORD,
        id=word_id,
        kanji=kanji,
        kana=kana,
        romaji=romaji,
        jlpt_level=jlpt_level,
        is_common=is_common,
    )
    tx.run(queries.UPSERT_WORD_READING, word_id=word_id, reading=kana)


def upsert_word_meaning(tx: ManagedTransaction, *, word_id: int, meaning: str) -> None:
    tx.run(queries.UPSERT_WORD_MEANING, word_id=word_id, meaning=meaning)


def upsert_word_tag(tx: ManagedTransaction, *, word_id: int, tag_name: str) -> None:
    tx.run(queries.UPSERT_WORD_TAG, word_id=word_id, tag_name=tag_name)


def upsert_kanji(
    tx: ManagedTransaction,
    *,
    character: str,
    meaning: str,
    onyomi: list[str],
    kunyomi: list[str],
) -> None:
    tx.run(
        queries.UPSERT_KANJI,
        character=character,
        meaning=meaning,
        onyomi=onyomi,
        kunyomi=kunyomi,
    )
    for reading in onyomi:
        tx.run(
            queries.UPSERT_KANJI_READING,
            character=character,
            reading=reading,
            reading_type="onyomi",
            source="kanji",
        )
    for reading in kunyomi:
        tx.run(
            queries.UPSERT_KANJI_READING,
            character=character,
            reading=reading,
            reading_type="kunyomi",
            source="kanji",
        )


def upsert_word_contains_kanji(
    tx: ManagedTransaction,
    *,
    word_id: int,
    character: str,
    position: int,
) -> None:
    tx.run(
        queries.UPSERT_WORD_CONTAINS_KANJI,
        word_id=word_id,
        character=character,
        position=position,
    )


def refresh_cooccurs_for_word(tx: ManagedTransaction, *, word_id: int) -> None:
    tx.run(queries.UPSERT_KANJI_COOCCURS, word_id=word_id)


def get_related_words(tx: ManagedTransaction, *, word_id: int, limit: int = 5) -> list[dict[str, Any]]:
    return _run(tx, queries.RELATED_WORDS_BY_KANJI, word_id=word_id, limit=limit)


def get_word_readings(tx: ManagedTransaction, *, word_id: int) -> list[str]:
    rows = _run(tx, queries.READINGS_FOR_WORD, word_id=word_id)
    return [row["reading"] for row in rows]


def get_word_kanji(tx: ManagedTransaction, *, word_id: int) -> list[str]:
    rows = _run(tx, queries.KANJI_FOR_WORD, word_id=word_id)
    return [row["character"] for row in rows]

