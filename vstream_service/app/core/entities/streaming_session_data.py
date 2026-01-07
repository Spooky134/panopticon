from datetime import datetime
from uuid import UUID
from typing import Optional

from dataclasses import dataclass, field

@dataclass(frozen=True)
class StreamingSessionData:
    test_id: Optional[UUID] = None
    user_id: Optional[int] = None
    status: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
