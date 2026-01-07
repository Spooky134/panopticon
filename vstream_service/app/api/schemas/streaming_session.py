from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

class StreamingSessionResponse(BaseModel):
    streaming_session_id: UUID
    user_id: int
    test_id: UUID
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    status: str


class StreamingSessionCreateRequest(BaseModel):
    user_id: int
    test_id: UUID

class StreamingSessionRequest(BaseModel):
    streaming_session_id: UUID