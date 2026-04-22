import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.common.logger import get_logger

DEFAULT_LOCAL_DB_URL = "postgresql+psycopg2://actionlab@localhost/actionlab"
logger = get_logger(__name__)

_EXPLICIT_DATABASE_URL = (
    os.getenv("ACTIONLAB_LOCAL_DB_URL")
    or os.getenv("ACTIONLAB_DB_URL")
)
_ENV_NAME = (os.getenv("ACTIONLAB_ENV") or "").strip().lower()

if not _EXPLICIT_DATABASE_URL and _ENV_NAME in {"prod", "production", "staging"}:
    raise RuntimeError(
        "Database URL not configured. Set ACTIONLAB_LOCAL_DB_URL or ACTIONLAB_DB_URL."
    )

DATABASE_URL = (
    _EXPLICIT_DATABASE_URL
    or DEFAULT_LOCAL_DB_URL
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        value = int(str(raw).strip())
    except Exception:
        logger.warning("[db] Invalid %s=%r; using default=%s", name, raw, default)
        return int(default)
    return max(0, value)

if not _EXPLICIT_DATABASE_URL:
    logger.warning(
        "[db] No explicit ACTIONLAB_LOCAL_DB_URL/ACTIONLAB_DB_URL set; using local fallback %s",
        DEFAULT_LOCAL_DB_URL,
    )

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=_env_int("ACTIONLAB_DB_POOL_SIZE", 10),
    max_overflow=_env_int("ACTIONLAB_DB_MAX_OVERFLOW", 20),
    pool_timeout=_env_int("ACTIONLAB_DB_POOL_TIMEOUT_SECONDS", 30),
    pool_recycle=_env_int("ACTIONLAB_DB_POOL_RECYCLE_SECONDS", 1800),
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_db():
    """
    FastAPI dependency for database sessions.
    
    Yields a database session and ensures it's closed after use.
    
    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
