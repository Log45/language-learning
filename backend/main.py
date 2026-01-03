# Main file for testing out japanese chatbots
from fastapi import FastAPI

app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}   

@app.get("/chat")
def chat():
    return {"message": "This is a placeholder for chatbot functionality."}
