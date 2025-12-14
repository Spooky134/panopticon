from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class StreamingSessionORMCreate(BaseModel):
    id: UUID
    test_id: UUID
    user_id: int
    status: str
    created_at: datetime


class StreamingSessionUpdate(BaseModel):
    pass
