from typing import Annotated
from fastapi import Depends, Request

from app.stream.service import StreamingService
from app.session.repository import SessionRepository
from vstream_service.app.core.unit_of_work import UnitOfWork


def get_streaming_service(request: Request) -> StreamingService:
    return StreamingService(
        streaming_manager=request.app.state.streaming_manager,
        session_repo=SessionRepository,
        uow=UnitOfWork,
        s3_storage=request.app.state.s3_video_storage
    )


StreamingServiceDep = Annotated[StreamingService, Depends(get_streaming_service)]

