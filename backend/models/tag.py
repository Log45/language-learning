from __future__ import annotations
from typing import TYPE_CHECKING

from sqlalchemy import String, Table, Column, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.db.base import Base

if TYPE_CHECKING:
    from backend.models.word import Word


word_tags = Table(
    "word_tags",
    Base.metadata,
    Column("word_id", ForeignKey("words.id"), primary_key=True),
    Column("tag_id", ForeignKey("tags.id"), primary_key=True),
)


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)

    words: Mapped[list["Word"]] = relationship(
        secondary=word_tags, back_populates="tags"
    )