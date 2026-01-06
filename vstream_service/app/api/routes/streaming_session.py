from uuid import UUID

from services.streaming_session_lifecycle_service import StreamingSessionLifecycleService
from services.streaming_session_lifecycle_service_factory import get_streaming_session_lifecycle_service
from fastapi import APIRouter, Depends, status
from api.schemas.streaming_session import StreamingSessionResponse, StreamingSessionCreateRequest
from core.security.api_key import get_api_key
from core.logger import get_logger
from api.dependencies import StreamingLifecycleServiceDep

logger = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=StreamingSessionResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_api_key)])
async def create_session(streaming_session_create: StreamingSessionCreateRequest,
                         streaming_session_lifecycle_service: StreamingLifecycleServiceDep):

    return await streaming_session_lifecycle_service.create_session(streaming_session_create=streaming_session_create)


@router.get("/{streaming_session_id}", response_model=StreamingSessionResponse, dependencies=[Depends(get_api_key)])
async def get_session(streaming_session_id: UUID,
                      streaming_session_lifecycle_service: StreamingLifecycleServiceDep):

    return await streaming_session_lifecycle_service.read_session(streaming_session_id=streaming_session_id)





