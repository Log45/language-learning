from pydantic import BaseModel
from typing import TYPE_CHECKING
from backend.models import Word, Meaning, Tag, Kanji, WordKanji, WordBank
from backend.db.session import SessionLocal
from sqlalchemy.orm import Session

from random import randint

class Flashcard(BaseModel):
    kanji: str
    kana: str
    meaning: str
    show_kana: bool
    
class Flashcards(BaseModel):
    flashcards: list[Flashcard]
    
def generate_flashcard(session: Session, word_id: int, show_kana: bool = True) -> Flashcard:
    """"""
    # show_kana will be toggle-able from the frontend, so it doesn't really matter here
    word = session.query(Word).filter(Word.id == word_id).first()
    meanings = word.meanings
    meanings_list = [meaning.meaning for meaning in meanings]
    return Flashcard(kanji=word.kanji, kana=word.kana, meaning="; ".join(meanings_list), show_kana=show_kana)

def from_wordbank(session: Session, user_id: int) -> list[dict]:
    """"""
    word_ids = session.query(WordBank).filter(WordBank.user_id == user_id).values()
    print(word_ids)
    return format_flashcards(session, word_ids)

def format_flashcards(session: Session, word_ids: list[int], show_kana: bool = True) -> list[dict]:
    """"""
    flashcards = Flashcards(flashcards=[generate_flashcard(session, word_id, show_kana) for word_id in word_ids])
    return flashcards.model_dump_json()
    
if __name__ == "__main__":
    ids = [randint(0, 5000) for _ in range(100)]
    print(format_flashcards(SessionLocal(), ids, True))