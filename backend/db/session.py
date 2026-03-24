import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://admin:secret@localhost/language_app")

engine = create_engine(DATABASE_URL, echo=True)

SessionLocal = sessionmaker(bind=engine)