from typing import Annotated
from fastapi import Depends

from app.streaming_session.service_factory import create_streaming_session_service
from app.streaming_session.service import StreamingSessionService


StreamingSessionServiceDep: type[StreamingSessionService] = Annotated[
    StreamingSessionService,
    Depends(create_streaming_session_service)
]