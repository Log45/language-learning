from __future__ import annotations
from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import ARRAY
from backend.db.base import Base

if TYPE_CHECKING:
    from backend.models.word import Word


class Kanji(Base):
    __tablename__ = "kanji"

    character: Mapped[str] = mapped_column(String, primary_key=True)
    meaning: Mapped[str] = mapped_column(String)
    onyomi: Mapped[list[str]] = mapped_column(ARRAY(String))
    kunyomi: Mapped[list[str]] = mapped_column(ARRAY(String))

    words: Mapped[list["Word"]] = relationship(
        secondary="word_kanji", back_populates="kanji_chars"
    )


class WordKanji(Base):
    __tablename__ = "word_kanji"

    word_id: Mapped[int] = mapped_column(ForeignKey("words.id"), primary_key=True)
    kanji_character: Mapped[str] = mapped_column(
        ForeignKey("kanji.character"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer)