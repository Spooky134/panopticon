from sqlalchemy import String, DateTime, Index, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
import uuid

from app.core.database import Base


class VideoModel(Base):
    __tablename__ = "video"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    s3_key: Mapped[str] = mapped_column(String(500))

    width: Mapped[int] = mapped_column(nullable=True)
    height: Mapped[int] = mapped_column(nullable=True)
    duration: Mapped[int] = mapped_column(nullable=True)
    fps: Mapped[float] = mapped_column(nullable=True)
    file_size: Mapped[int] = mapped_column(nullable=True)
    mime_type: Mapped[str] = mapped_column(String(100), default="video/mp4", nullable=True)
    meta: Mapped[dict] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    streaming_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("session.id")
    )

    session: Mapped["SessionModel"] = relationship(
        back_populates="videos"
    )

    __table_args__ = (
        Index("idx_video_s3_key", "s3_key"),
        Index("idx_video_session_id", "session_id"),
    )