from __future__ import annotations
from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, Boolean, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base

if TYPE_CHECKING:
    from backend.models.meaning import Meaning
    from backend.models.tag import Tag
    from backend.models.kanji import Kanji


class Word(Base):
    __tablename__ = "words"

    # Primary Key
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )

    # Core Fields
    kanji: Mapped[str] = mapped_column(String, nullable=False)
    kana: Mapped[str] = mapped_column(String, nullable=False)
    romaji: Mapped[str] = mapped_column(String, nullable=False)

    jlpt_level: Mapped[int] = mapped_column(Integer, nullable=False)
    is_common: Mapped[bool] = mapped_column(Boolean, default=False)

    # Constraints
    __table_args__ = (
        UniqueConstraint("kanji", "kana", name="uq_word_kanji_kana"),
    )

    # Relationships
    meanings: Mapped[list["Meaning"]] = relationship(
        back_populates="word",
        cascade="all, delete-orphan",
    )

    tags: Mapped[list["Tag"]] = relationship(
        secondary="word_tags",
        back_populates="words",
    )

    kanji_chars: Mapped[list["Kanji"]] = relationship(
        secondary="word_kanji",
        back_populates="words",
        order_by="WordKanji.position",
    )