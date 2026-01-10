import uuid
from sqlalchemy import (
    Column, String, Integer, Float, Text, ForeignKey, JSON, TIMESTAMP
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Player(Base):
    __tablename__ = "players"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String)
    age = Column(Integer)
    handedness = Column(String)
    bowler_type = Column(String)
    created_at = Column(TIMESTAMP, server_default=func.now())


class Video(Base):
    __tablename__ = "videos"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(UUID(as_uuid=True), ForeignKey("players.id"))
    file_path = Column(Text, nullable=False)
    fps = Column(Float)
    duration_sec = Column(Float)
    created_at = Column(TIMESTAMP, server_default=func.now())


class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(UUID(as_uuid=True), ForeignKey("players.id"))
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"))

    schema_version = Column(String, nullable=False)
    biomechanics_json = Column(JSON, nullable=False)
    clinician_yaml = Column(JSON, nullable=False)

    verdict = Column(String)
    risk_summary = Column(JSON)
    positives = Column(JSON)
    strength_focus = Column(JSON)

    created_at = Column(TIMESTAMP, server_default=func.now())
