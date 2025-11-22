from sqlalchemy import String, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Mapped, mapped_column

from datetime import datetime
import uuid

from core.database import Base


class TestingSession(Base):
    __tablename__ = "testing_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    test_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="started", nullable=False, index=True)
    # incidents = Column(JSONB)
    # ml_metrics = Column(JSONB)
    # meta = Column(JSONB)
    video = relationship("TestingVideo", back_populates="testing_session", uselist=False)

    __table_args__ = (
        Index("idx_testing_session_user_id", "user_id"),
        Index("idx_testing_session_status", "status"),
    )
