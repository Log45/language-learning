from backend.db.base import Base
from backend.db.session import engine
import backend.models  # IMPORTANT: registers models

def init_db():
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    init_db()