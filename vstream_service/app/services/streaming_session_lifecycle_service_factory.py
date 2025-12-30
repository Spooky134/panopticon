from services.streaming_session_lifecycle_service import StreamingSessionLifecycleService
from core.database import AsyncSessionLocal

def get_streaming_session_lifecycle_service() -> StreamingSessionLifecycleService:
    return StreamingSessionLifecycleService(
        session_factory=AsyncSessionLocal
    )
