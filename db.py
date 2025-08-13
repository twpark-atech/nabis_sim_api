# app/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 필요에 맞게 환경변수/설정으로 빼도 됨
DATABASE_URL = "postgresql://postgres:postgres@172.30.1.66:5432/dashboard"

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()