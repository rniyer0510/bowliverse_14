from app.persistence.session import engine
from app.persistence.models import Base
from app.common.logger import get_logger

logger = get_logger(__name__)


def init_db():
    logger.info("Creating DB tables if missing")
    Base.metadata.create_all(bind=engine)
