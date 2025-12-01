import uuid

from pydantic import BaseModel

class StreamingSessionResponseInfo(BaseModel):
    streaming_session_id: uuid.UUID

class StreamingSessionResponse(BaseModel):
    streaming_session_id: uuid.UUID
    token: str

class StreamingSessionCreate(BaseModel):
    user_id: int