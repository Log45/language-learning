from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from backend.db.session import SessionLocal
from backend.db.neo4j_session import driver as neo4j_driver
from backend.graph import GraphClient
from backend.graph.ingest import ensure_graph_schema, upsert_word_graph
from backend.models import Word, Meaning, Tag, Kanji, WordKanji


DATA_DIR = Path("./backend/data")  # folder with your JSON files
LOGGER = logging.getLogger(__name__)


# Helpers
def parse_jlpt(jlpt_str: str) -> int:
    return int(jlpt_str.replace("N", ""))


def is_katakana(text: str) -> bool:
    return text != "" and all("\u30a0" <= c <= "\u30ff" for c in text)


def is_hiragana(text: str) -> bool:
    return text != "" and all("\u3040" <= c <= "\u309f" for c in text)


def contains_kanji(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9faf" for c in text)


def extract_kanji_chars(text: str) -> list[str]:
    return [c for c in text if "\u4e00" <= c <= "\u9faf"]


def normalize_kana(entry: dict) -> str:
    # Prefer provided hiragana
    if entry.get("hiragana"):
        return entry["hiragana"]

    text = entry.get("kanji", "")

    # If it's pure kana (katakana or hiragana), use it
    if is_katakana(text) or is_hiragana(text):
        return text

    # fallback (bad data)
    return ""


# Upsert Helpers
def get_or_create_tag(session: Session, name: str) -> Tag:
    tag = session.query(Tag).filter_by(name=name).first()
    if tag:
        return tag

    tag = Tag(name=name)
    session.add(tag)
    session.flush()
    return tag


def get_or_create_kanji(session: Session, char: str) -> Kanji:
    k = session.get(Kanji, char)
    if k:
        return k

    k = Kanji(
        character=char,
        meaning="",
        onyomi=[],
        kunyomi=[],
    )
    session.add(k)
    session.flush()
    return k


def get_existing_word(session: Session, kanji: str, kana: str) -> Word | None:
    return (
        session.query(Word)
        .filter(Word.kanji == kanji, Word.kana == kana)
        .first()
    )


# Core Ingestion
def ingest_file(session: Session, filepath: Path, graph_client: GraphClient | None = None):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        kanji_text = entry.get("kanji", "")
        kana = normalize_kana(entry)

        # Skip if malformed
        if not kanji_text and not kana:
            continue

        # Avoid duplicates (uses your UNIQUE constraint)
        existing = get_existing_word(session, kanji_text, kana)
        if existing:
            continue

        # Create Word
        word = Word(
            kanji=kanji_text,
            kana=kana,
            romaji=entry.get("romaji", ""),
            jlpt_level=parse_jlpt(entry["jlpt"]),
            is_common=any(b["text"] == "popular" for b in entry.get("badges", [])),
        )
        session.add(word)
        session.flush()  # get word.id

        # Meanings
        meanings = []
        for m in entry.get("meanings", []):
            session.add(Meaning(word_id=word.id, meaning=m))
            meanings.append(m)

        # Tags
        tag_names = []
        for badge in entry.get("badges", []):
            tag_name = badge.get("text")
            if not tag_name:
                continue

            tag = get_or_create_tag(session, tag_name)
            word.tags.append(tag)
            tag_names.append(tag_name)

        # Kanji
        kanji_chars: list[str] = []
        if contains_kanji(kanji_text):
            chars = extract_kanji_chars(kanji_text)
            kanji_chars = chars

            for i, char in enumerate(chars):
                get_or_create_kanji(session, char)

                session.add(
                    WordKanji(
                        word_id=word.id,
                        kanji_character=char,
                        position=i,
                    )
                )

        if graph_client is not None:
            kanji_lookup = {
                k.character: {
                    "meaning": k.meaning,
                    "onyomi": k.onyomi or [],
                    "kunyomi": k.kunyomi or [],
                }
                for k in session.query(Kanji).filter(Kanji.character.in_(kanji_chars)).all()
            } if kanji_chars else {}
            try:
                upsert_word_graph(
                    graph_client,
                    word=word,
                    meanings=meanings,
                    tag_names=tag_names,
                    kanji_chars=kanji_chars,
                    kanji_lookup=kanji_lookup,
                )
            except Exception:
                LOGGER.exception("Graph sync failed for word id=%s", word.id)


# Entry Point
def main():
    session = SessionLocal()
    graph_client = GraphClient(neo4j_driver) if neo4j_driver is not None else None

    try:
        if graph_client is not None:
            ensure_graph_schema(graph_client)
        files = list(DATA_DIR.glob("*.json"))

        if not files:
            print("No JSON files found.")
            return

        for file in files:
            print(f"Ingesting {file.name}...")
            ingest_file(session, file, graph_client=graph_client)
            session.commit()

        print("Ingestion complete.")

    except Exception as e:
        session.rollback()
        raise e

    finally:
        session.close()
        if graph_client is not None:
            graph_client.close()


if __name__ == "__main__":
    main()