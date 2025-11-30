import uuid

from api.services.session import SessionService
from api.services.session_factory import get_session_service
from fastapi import APIRouter, Depends, status
from api.schemas.session import SessionResponse, SessionCreate, SessionResponseInfo
from api.exceptions.exeptions import NotFoundError, HTTPException, ValidationError
from core.security.api_key import get_api_key
from core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_api_key)])
async def create_session(session_create: SessionCreate,
                         session_service: SessionService = Depends(get_session_service)):

    return await session_service.create_session(session_create)


@router.get("/{session_id}", response_model=SessionResponseInfo, dependencies=[Depends(get_api_key)])
async def get_session(session_id: uuid.UUID,
                      service: SessionService = Depends(get_session_service)):

    return await service.read_session(session_id=session_id)





