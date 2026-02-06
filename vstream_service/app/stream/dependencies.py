from typing import Annotated
from fastapi import Depends, Request

from app.stream.service import StreamingRuntimeService
from app.stream.service_factory import create_streaming_runtime_service
from app.streaming_session.service_factory import get_streaming_session_lifecycle_service



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

