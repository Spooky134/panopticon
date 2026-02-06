from typing import Annotated
from fastapi import Depends, Request

from services.streaming_runtime_service import StreamingRuntimeService
from services.streaming_runtime_service_factory import create_streaming_runtime_service
from services.streaming_session_lifecycle_service import StreamingSessionLifecycleService
from services.streaming_session_lifecycle_service_factory import get_streaming_session_lifecycle_service



def get_streaming_runtime_service(
        request: Request,
        streaming_session_lifecycle_service = Depends(get_streaming_session_lifecycle_service)
) -> StreamingRuntimeService:
    return create_streaming_runtime_service(
        streaming_session_manager=request.app.state.session_manager,
        streaming_session_lifecycle_service=streaming_session_lifecycle_service,
        s3_video_storage=request.app.state.s3_video_storage
    )

StreamingRuntimeServiceDep: type[StreamingRuntimeService] = Annotated[
    StreamingRuntimeService,
    Depends(get_streaming_runtime_service)
]


StreamingLifecycleServiceDep: type[StreamingSessionLifecycleService] = Annotated[
    StreamingSessionLifecycleService,
    Depends(get_streaming_session_lifecycle_service)
]