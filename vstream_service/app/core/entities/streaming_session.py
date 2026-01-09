from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import Optional
from dataclasses import dataclass, field

from core.engine.live_streaming_session_status import LiveStreamingSessionStatus


@dataclass(frozen=True)
class StreamingSessionEntity:
    user_id: int
    # TODO после пересмотрах схемы бд изменить обязательность
    test_id: Optional[UUID] = None

    id: Optional[UUID] = field(default_factory=uuid4)
    status: str = field(default=LiveStreamingSessionStatus.CREATED)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    @classmethod
    def from_db(cls, model) -> "StreamingSessionEntity":
        return cls(id=model.id,
                   test_id=model.test_id,
                   user_id=model.user_id,
                   status=model.status,
                   created_at=model.created_at,
                   started_at=model.started_at,
                   ended_at=model.ended_at)