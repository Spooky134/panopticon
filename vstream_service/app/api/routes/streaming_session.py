import uuid

from api.services.streaming_session import StreamingSessionService
from api.services.streaming_session_factory import get_streaming_session_service
from fastapi import APIRouter, Depends, status
from api.schemas.streaming_session import StreamingSessionResponse, StreamingSessionCreate, StreamingSessionResponseInfo
from api.exceptions.exeptions import NotFoundError, HTTPException, ValidationError
from core.security.api_key import get_api_key
from core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("/", response_model=StreamingSessionResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_api_key)])
async def create_session(streaming_session_create: StreamingSessionCreate,
                         streaming_session_service: StreamingSessionService = Depends(get_streaming_session_service)):

    return await streaming_session_service.create_session(streaming_session_create=streaming_session_create)


@router.get("/{streaming_session_id}", response_model=StreamingSessionResponseInfo, dependencies=[Depends(get_api_key)])
async def get_session(streaming_session_id: uuid.UUID,
                      streaming_session_service: StreamingSessionService = Depends(get_streaming_session_service)):

    return await streaming_session_service.read_session(streaming_session_id=streaming_session_id)





