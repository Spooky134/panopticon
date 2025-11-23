from sqlalchemy import String, Integer, DateTime, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from datetime import datetime
import uuid

from core.database import Base


class TestingVideo(Base):
    __tablename__ = "testing_videos"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    s3_key: Mapped[str] = mapped_column(String(500))
    s3_bucket: Mapped[str] = mapped_column(String(255))
    duration: Mapped[int] = mapped_column(Integer, nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[str] = mapped_column(String(100), default="video/mp4", nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    testing_session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("testing_sessions.id"), unique=True)
    testing_session = relationship("TestingSession", back_populates="video")

    __table_args__ = (
        Index("idx_testing_video_s3_key", "s3_key"),
        Index("idx_testing_video_testing_session_id", "testing_session_id"),
    )