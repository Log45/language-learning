# Main file for testing out japanese chatbots
from fastapi import FastAPI, Query
from backend.services.flashcard import from_wordbank
from backend.db.session import SessionLocal

app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}   

@app.get("/chat")
def chat():
    return {"message": "This is a placeholder for chatbot functionality."}

@app.get("/flashcards")
def get_user_flashcards(
    user_id: int = Query(...),
    show_kana: bool = Query(True),
    enrich: bool = Query(False),
):
    session = SessionLocal()
    try:
        return from_wordbank(session, user_id=user_id, show_kana=show_kana, enrich=enrich)
    finally:
        session.close()