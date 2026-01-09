from fastapi import APIRouter, Depends, status

from api.dependencies import StreamingLifecycleServiceDep
from api.schemas.streaming_session import StreamingSessionCreateRequest, StreamingSessionRequest, StreamingSessionResponse
from core.security.api_key import get_api_key
from core.logger import get_logger


logger = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["streaming_sessions"])

@router.post("", response_model=StreamingSessionResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_api_key)])
async def create_session(request: StreamingSessionCreateRequest,
                         streaming_session_lifecycle_service: StreamingLifecycleServiceDep):
    streaming_session_entity = await streaming_session_lifecycle_service.create_session(**request.model_dump())
    return StreamingSessionResponse.model_validate(streaming_session_entity)


@router.get("/{streaming_session_id}", response_model=StreamingSessionResponse, dependencies=[Depends(get_api_key)])
async def get_session(request: StreamingSessionRequest,
                      streaming_session_lifecycle_service: StreamingLifecycleServiceDep):
    streaming_session_entity = await streaming_session_lifecycle_service.read_session(**request.model_dump())
    return StreamingSessionResponse.model_validate(streaming_session_entity)





