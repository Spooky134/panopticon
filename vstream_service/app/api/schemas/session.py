import uuid

from pydantic import BaseModel

class SessionResponseInfo(BaseModel):
    session_id: uuid.UUID

class SessionResponse(BaseModel):
    session_id: uuid.UUID
    token: str

class SessionCreate(BaseModel):
    user_id: int