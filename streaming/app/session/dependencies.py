from typing import Annotated
from fastapi import Depends
from app.core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession

from app.session.service import SessionService
from app.session.repository import SessionRepository
from app.core.dependecies import get_uow
from app.core.unit_of_work import UnitOfWork


async def get_session_repo(db_session: AsyncSession = Depends(get_db)) -> SessionRepository:
    return SessionRepository(db_session)


async def get_session_service(uow: UnitOfWork = Depends(get_uow)) -> SessionService:
    session_repo = SessionRepository(uow.session)
    return SessionService(session_repo)


SessionServiceDep = Annotated[SessionService, Depends(get_session_service)]