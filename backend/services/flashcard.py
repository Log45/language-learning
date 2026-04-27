from pydantic import BaseModel
from backend.models import Word, WordBank
from sqlalchemy.orm import Session

from random import randint
from backend.services.graph import get_word_enrichment

class Flashcard(BaseModel):
    word_id: int
    kanji: str
    kana: str
    meaning: str
    show_kana: bool
    enrichment: dict | None = None
    
class Flashcards(BaseModel):
    flashcards: list[Flashcard]
    
def generate_flashcard(session: Session, word_id: int, show_kana: bool = True) -> Flashcard:
    """"""
    # show_kana will be toggle-able from the frontend, so it doesn't really matter here
    word = session.query(Word).filter(Word.id == word_id).first()
    if word is None:
        raise ValueError(f"Word id {word_id} not found")
    meanings = word.meanings
    meanings_list = [meaning.meaning for meaning in meanings]
    return Flashcard(
        word_id=word.id,
        kanji=word.kanji,
        kana=word.kana,
        meaning="; ".join(meanings_list),
        show_kana=show_kana,
    )

def from_wordbank(
    session: Session,
    user_id: int,
    show_kana: bool = True,
    enrich: bool = False,
) -> list[dict]:
    """"""
    rows = session.query(WordBank.word_id).filter(WordBank.user_id == user_id).all()
    word_ids = [word_id for (word_id,) in rows]
    return format_flashcards(session, word_ids, show_kana=show_kana, enrich=enrich)

def format_flashcards(
    session: Session,
    word_ids: list[int],
    show_kana: bool = True,
    enrich: bool = False,
) -> list[dict]:
    """"""
    flashcards: list[Flashcard] = []
    for word_id in word_ids:
        try:
            flashcard = generate_flashcard(session, word_id, show_kana)
        except ValueError:
            continue
        if enrich:
            flashcard.enrichment = get_word_enrichment(word_id)
        flashcards.append(flashcard)
    wrapped = Flashcards(flashcards=flashcards)
    return wrapped.model_dump()
    
if __name__ == "__main__":
    ids = [randint(0, 5000) for _ in range(100)]
    from backend.db.session import SessionLocal

    print(format_flashcards(SessionLocal(), ids, True))