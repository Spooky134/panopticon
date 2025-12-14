from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class VideoMeta(BaseModel):
    codec: Optional[str] = None
    frame_count: Optional[int] = None
    bit_rate: Optional[float] = None


class StreamingVideoORMCreate(BaseModel):
    streaming_session_id: UUID
    s3_key: str
    s3_bucket: str
    created_at: datetime
    duration: Optional[float] = None
    fps: Optional[float] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    meta: Optional[VideoMeta] = None



