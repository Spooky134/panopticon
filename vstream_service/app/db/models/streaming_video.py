from sqlalchemy import String, Integer, DateTime, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
import uuid

from core.database import Base


class StreamingVideo(Base):
    __tablename__ = "streaming_videos"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    s3_key: Mapped[str] = mapped_column(String(500))
    s3_bucket: Mapped[str] = mapped_column(String(255))
    duration: Mapped[int] = mapped_column(Integer, nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[str] = mapped_column(String(100), default="video/mp4", nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    streaming_session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("streaming_sessions.id"), unique=True)

    streaming_session = relationship("StreamingSession", back_populates="video")

    __table_args__ = (
        Index("idx_streaming_video_s3_key", "s3_key"),
        Index("idx_streaming_video_streaming_session_id", "streaming_session_id"),
    )