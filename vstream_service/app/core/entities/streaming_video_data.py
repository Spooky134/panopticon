from datetime import datetime
from typing import Optional
from uuid import UUID
from dataclasses import dataclass, asdict



@dataclass(frozen=True)
class VideoMetaData:
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
class StreamingVideoData:
    s3_key: str
    s3_bucket: str
    created_at: datetime
    meta: Optional[VideoMetaData] = None


