from fastapi import Depends, Request

from app.stream.service import StreamingService
from app.streaming_session.service_factory import create_streaming_session_service


def create_streaming_service(
        request: Request,
        streaming_session_service = Depends(create_streaming_session_service)
) -> StreamingService:
    return StreamingService(
        streaming_manager=request.app.state.streaming_manager,
        streaming_session_service=streaming_session_service,
        s3_video_storage=request.app.state.s3_video_storage
    )
