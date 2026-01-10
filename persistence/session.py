from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# IMPORTANT: point to the ActionLab database, not the default postgres DB
DATABASE_URL = "postgresql://postgres@localhost/actionlab"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

