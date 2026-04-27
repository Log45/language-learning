from backend.graph.client import GraphClient
from backend.graph import repository
from backend.models import Word


def ensure_graph_schema(graph_client: GraphClient) -> None:
    graph_client.execute_write(repository.create_constraints)


def upsert_word_graph(
    graph_client: GraphClient,
    *,
    word: Word,
    meanings: list[str],
    tag_names: list[str],
    kanji_chars: list[str],
    kanji_lookup: dict[str, dict],
) -> None:
    graph_client.execute_write(
        repository.upsert_word,
        word_id=word.id,
        kanji=word.kanji,
        kana=word.kana,
        romaji=word.romaji,
        jlpt_level=word.jlpt_level,
        is_common=word.is_common,
    )

    for meaning in meanings:
        graph_client.execute_write(
            repository.upsert_word_meaning,
            word_id=word.id,
            meaning=meaning,
        )

    for tag_name in tag_names:
        graph_client.execute_write(
            repository.upsert_word_tag,
            word_id=word.id,
            tag_name=tag_name,
        )

    for idx, char in enumerate(kanji_chars):
        kanji_data = kanji_lookup.get(char, {})
        graph_client.execute_write(
            repository.upsert_kanji,
            character=char,
            meaning=kanji_data.get("meaning", ""),
            onyomi=kanji_data.get("onyomi", []) or [],
            kunyomi=kanji_data.get("kunyomi", []) or [],
        )
        graph_client.execute_write(
            repository.upsert_word_contains_kanji,
            word_id=word.id,
            character=char,
            position=idx,
        )

    if len(kanji_chars) > 1:
        graph_client.execute_write(repository.refresh_cooccurs_for_word, word_id=word.id)

