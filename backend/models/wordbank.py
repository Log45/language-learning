from datetime import datetime

from sqlalchemy import Float, Integer, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from backend.db.base import Base

class WordBank(Base):
    """Table holding words that a user has interacted with"""
    __tablename__ = "wordbank"
    
    user_id: Mapped[int] = mapped_column(primary_key=True)
    word_id: Mapped[int] = mapped_column(ForeignKey("words.id"), primary_key=True)

    familiarity_score: Mapped[float] = mapped_column(Float, default=0.0)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    correct_count: Mapped[int] = mapped_column(Integer, default=0)
    incorrect_count: Mapped[int] = mapped_column(Integer, default=0)