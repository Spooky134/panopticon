from uuid import UUID

from pydantic import BaseModel

class StreamingSessionResponseInfo(BaseModel):
    streaming_session_id: UUID

class StreamingSessionResponse(BaseModel):
    streaming_session_id: UUID

class StreamingSessionCreate(BaseModel):
    user_id: int