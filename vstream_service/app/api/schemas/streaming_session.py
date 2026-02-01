from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

class StreamingSessionResponse(BaseModel):
    id: UUID
    user_id: int
    test_id: UUID
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    status: str

    model_config = ConfigDict(from_attributes=True)


class StreamingSessionCreateRequest(BaseModel):
    user_id: int
    test_id: UUID

class StreamingSessionRequest(BaseModel):
    id: UUID