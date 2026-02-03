from sqlalchemy import String, DateTime, Index, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Mapped, mapped_column

from datetime import datetime
import uuid

from core.database import Base



class StreamingSessionModel(Base):
    __tablename__ = "streaming_session"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="created", nullable=True, index=True)

    video: Mapped[list["StreamingVideoModel"]] = relationship(
        "StreamingVideoModel",
        back_populates="streaming_session",
    )

    __table_args__ = (
        Index("idx_streaming_session_status", "status"),
    )