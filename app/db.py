# app/db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DB_URL, 
    pool_pre_ping=True, 
    pool_size=10,
    max_overflow=20,
    pool_recycle=1800,
    pool_timeout=30,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()