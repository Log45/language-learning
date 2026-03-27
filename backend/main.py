# Main file for testing out japanese chatbots
from fastapi import FastAPI
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
def get_user_flashcards(request):
    return from_wordbank(SessionLocal(), request.user_id)