from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from datetime import datetime


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=True)
    handedness = Column(String, nullable=True)
    bowler_type = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"))
    video_path = Column(String)
    result_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
