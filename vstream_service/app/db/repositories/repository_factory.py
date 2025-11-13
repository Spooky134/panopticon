from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from db.repositories.testing_session import TestingSessionRepository


def get_testing_session_repository(db: AsyncSession = Depends(get_db)) -> TestingSessionRepository:
    return TestingSessionRepository(db=db)
