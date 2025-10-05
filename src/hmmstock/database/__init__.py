from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

def get_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}")

def get_session_local(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_db_tables(engine):
    Base.metadata.create_all(engine)