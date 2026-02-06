from typing import Annotated
from fastapi import Depends

from app.streaming_session.service import StreamingSessionLifecycleService
from app.streaming_session.service_factory import get_streaming_session_lifecycle_service


StreamingLifecycleServiceDep: type[StreamingSessionLifecycleService] = Annotated[
    StreamingSessionLifecycleService,
    Depends(get_streaming_session_lifecycle_service)
]