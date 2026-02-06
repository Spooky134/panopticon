from app.streaming_session.service import StreamingSessionLifecycleService
from core.database import async_session_maker

def get_streaming_session_lifecycle_service() -> StreamingSessionLifecycleService:
    return StreamingSessionLifecycleService(
        session_factory=async_session_maker
    )
