from __future__ import annotations
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.db.base import Base

if TYPE_CHECKING:
    from backend.models.word import Word


class Meaning(Base):
    __tablename__ = "meanings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    word_id: Mapped[int] = mapped_column(ForeignKey("words.id"))
    meaning: Mapped[str] = mapped_column(Text)

    word: Mapped["Word"] = relationship(back_populates="meanings")