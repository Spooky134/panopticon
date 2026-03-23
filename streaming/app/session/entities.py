from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import Optional, Union, List
from dataclasses import dataclass, field

from app.stream.engine.stream_status import StreamStatus
from app.core.types import UNSET, UnsetType
from app.video.entities import StreamingVideoEntity

@dataclass(frozen=True)
class StreamingSessionEntity:
    id: Optional[UUID]
    status: StreamStatus = field(default=StreamStatus.CREATED)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    videos: List[StreamingVideoEntity] = None


@dataclass(frozen=True)
class CreateSessionEntity:
    id: Optional[UUID]
    status: str = field(default=StreamStatus.CREATED)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class UpdateSessionEntity:
    status: Union[StreamStatus, UnsetType] = UNSET
    started_at: Union[datetime, None, UnsetType] = UNSET
    ended_at: Union[datetime, None, UnsetType] = UNSET
