from uuid import UUID
from fastapi import APIRouter, Depends, status
from typing import List

from app.session.dependencies import SessionServiceDep
from app.session.schemas import CreateSessionRequest, SessionResponse
from app.core.security.api_key import get_api_key
from app.core.logger import get_logger


logger = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"], dependencies=[Depends(get_api_key)])


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: UUID,
                      session_service: SessionServiceDep):
    session_entity = await session_service.get_one_session(session_id)
    return session_entity


@router.get("", response_model=List[SessionResponse])
async def get_sessions(session_service: SessionServiceDep):
    return await session_service.get_all_sessions()


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(data: CreateSessionRequest, session_service: SessionServiceDep):
    session_entity = await session_service.create_session(**data.model_dump())
    return session_entity


