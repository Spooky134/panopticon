from typing import Annotated
from fastapi import Depends
from app.core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession

from app.streaming_session.service import SessionService
from app.streaming_session.repository import SessionRepository


async def get_session_repo(
    db_session: AsyncSession = Depends(get_db),
) -> SessionRepository:
    return SessionRepository(db_session)


async def get_session_service(
        session_repo: SessionRepository = Depends(get_session_repo)
) -> SessionService:
    return SessionService(
        session_repo=session_repo,
    )


StreamingSessionServiceDep: type[SessionService] = Annotated[
    SessionService,
    Depends(get_session_service)
]