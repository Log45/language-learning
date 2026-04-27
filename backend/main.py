# Main file for testing out japanese chatbots
from fastapi import FastAPI, HTTPException, Query
from backend.services.flashcard import from_wordbank
from backend.db.session import SessionLocal
from backend.db.neo4j_session import driver as neo4j_driver
from backend.graph.client import GraphClient
from backend.graph.subgraph import get_subgraph_for_kanji, get_subgraph_for_word

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


@app.get("/graph/subgraph")
def graph_subgraph(
    word_id: int | None = Query(None, description="Center on Word.id"),
    kanji: str | None = Query(None, description="Center on Kanji.character (one character)"),
    edge_limit: int = Query(100, ge=1, le=300),
    max_nodes: int = Query(150, ge=2, le=150),
    exclude_rel: str = Query(
        "",
        description="Comma-separated relationship types to exclude, e.g. HAS_TAG,CO_OCCURS_WITH",
    ),
):
    if neo4j_driver is None:
        raise HTTPException(status_code=503, detail="Neo4j driver not configured")
    if (word_id is None) == (kanji is None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of word_id or kanji",
        )
    exclude = [x.strip() for x in exclude_rel.split(",") if x.strip()]
    client = GraphClient(neo4j_driver)
    if word_id is not None:
        return get_subgraph_for_word(
            client,
            word_id=word_id,
            edge_limit=edge_limit,
            max_nodes=max_nodes,
            exclude_rel_types=exclude or None,
        )
    ch = (kanji or "").strip()
    if len(ch) != 1:
        raise HTTPException(status_code=400, detail="kanji must be exactly one character")
    return get_subgraph_for_kanji(
        client,
        character=ch,
        edge_limit=edge_limit,
        max_nodes=max_nodes,
        exclude_rel_types=exclude or None,
    )