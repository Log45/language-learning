from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from backend.db.session import SessionLocal
from backend.models import Word, Meaning, Tag, Kanji, WordKanji


DATA_DIR = Path("./backend/data")  # folder with your JSON files


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
def ingest_file(session: Session, filepath: Path):
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
        for m in entry.get("meanings", []):
            session.add(Meaning(word_id=word.id, meaning=m))

        # Tags
        for badge in entry.get("badges", []):
            tag_name = badge.get("text")
            if not tag_name:
                continue

            tag = get_or_create_tag(session, tag_name)
            word.tags.append(tag)

        # Kanji
        if contains_kanji(kanji_text):
            chars = extract_kanji_chars(kanji_text)

            for i, char in enumerate(chars):
                k = get_or_create_kanji(session, char)

                session.add(
                    WordKanji(
                        word_id=word.id,
                        kanji_character=char,
                        position=i,
                    )
                )


# Entry Point
def main():
    session = SessionLocal()

    try:
        files = list(DATA_DIR.glob("*.json"))

        if not files:
            print("No JSON files found.")
            return

        for file in files:
            print(f"Ingesting {file.name}...")
            ingest_file(session, file)
            session.commit()

        print("Ingestion complete.")

    except Exception as e:
        session.rollback()
        raise e

    finally:
        session.close()


if __name__ == "__main__":
    main()