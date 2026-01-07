from datetime import datetime
from uuid import UUID

from dataclasses import dataclass, field

@dataclass
class StreamingSessionData:
    id: UUID
    test_id: UUID
    user_id: int
    status: str
    created_at: datetime
