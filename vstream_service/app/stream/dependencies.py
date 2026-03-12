from typing import Annotated
from fastapi import Depends, Request

from app.stream.service import StreamingService
from app.streaming_session.dependencies import get_session_repo
from app.streaming_session.repository import SessionRepository


def get_streaming_service(
        request: Request,
        session_repo: SessionRepository = Depends(get_session_repo)
) -> StreamingService:
    return StreamingService(
        streaming_manager=request.app.state.streaming_manager,
        session_repo=session_repo,
        s3_storage=request.app.state.s3_video_storage
    )


StreamingServiceDep: type[StreamingService] = Annotated[
    StreamingService,
    Depends(get_streaming_service)
]

