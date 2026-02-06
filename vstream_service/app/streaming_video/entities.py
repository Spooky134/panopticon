from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict, field


@dataclass(frozen=True)
class VideoMetaEntity:
    width: Optional[int]
    height: Optional[int]
    duration: Optional[float]
    codec: Optional[str]
    file_size: Optional[int]
    mime_type: Optional[str]
    fps: Optional[float] = None
    bit_rate: Optional[float] = None
    frame_count: Optional[int] = None

    def get_extra(self) -> dict:
        main_columns = {"width", "height", "duration", "fps", "mime_type", "file_size"}

        all_data = asdict(self)
        return {k: v for k, v in all_data.items() if k not in main_columns}


@dataclass(frozen=True)
class StreamingVideoEntity:
    s3_key: str
    streaming_session_id: UUID

    meta: Optional[VideoMetaEntity] = None
    id: Optional[UUID] = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_db(cls, model) -> "StreamingVideoEntity":
        return cls(id=model.id,
                   s3_key=model.s3_key,
                   created_at=model.created_at,
                   streaming_session_id=model.streaming_session_id,
                   meta=VideoMetaEntity(width=model.width, height=model.height, duration=model.duration,
                                        codec=model.meta.get("codec"), file_size=model.file_size,
                                        mime_type=model.mime_type, fps=model.fps, bit_rate=model.meta.get("bit_rate"),
                                        frame_count=model.meta.get("frame_count"))
                   )



