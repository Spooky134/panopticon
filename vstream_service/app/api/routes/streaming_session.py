from uuid import UUID
from fastapi import APIRouter, Depends, status
from typing import List

from api.dependencies import StreamingLifecycleServiceDep
from api.schemas.streaming_session import StreamingSessionRequest, StreamingSessionResponse
from core.security.api_key import get_api_key
from core.logger import get_logger


logger = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["streaming_sessions"], dependencies=[Depends(get_api_key)])



@router.get("/{streaming_session_id}", response_model=StreamingSessionResponse)
async def get_session(streaming_session_id: UUID,
                      streaming_session_lifecycle_service: StreamingLifecycleServiceDep):
    streaming_session_entity = await streaming_session_lifecycle_service.get_one_session(
        streaming_session_id=streaming_session_id
    )
    return streaming_session_entity


@router.get("", response_model=List[StreamingSessionResponse])
async def get_sessions(streaming_session_lifecycle_service: StreamingLifecycleServiceDep):
    return await streaming_session_lifecycle_service.get_all_sessions()


@router.post("", response_model=StreamingSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(streaming_session_create: StreamingSessionRequest,
                         streaming_session_lifecycle_service: StreamingLifecycleServiceDep):
    streaming_session_entity = await streaming_session_lifecycle_service.create_session(
        streaming_session_id=streaming_session_create.streaming_session_id
    )
    return streaming_session_entity


