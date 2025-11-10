import uuid
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from core.database import Base


class TestingSession(Base):
    __tablename__ = "testing_session"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, nullable=False, index=True)
    test_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True))
    ended_at = Column(DateTime(timezone=True))
    status = Column(String(20), default="started", nullable=False, index=True)
    video_url = Column(Text)
    # incidents = Column(JSON)
    # ml_metrics = Column(JSON)
    # meta = Column(JSON)

    __table_args__ = (
        Index("idx_testing_session_user_id", "user_id"),
        Index("idx_testing_session_status", "status"),
    )
