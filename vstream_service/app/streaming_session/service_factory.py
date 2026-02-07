from app.streaming_session.service import StreamingSessionService
from core.database import async_session_maker

def create_streaming_session_service() -> StreamingSessionService:
    return StreamingSessionService(
        session_factory=async_session_maker
    )
