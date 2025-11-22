from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import exists, select, update, delete
from sqlalchemy.orm import selectinload
from db.models.testing_sessions import TestingSession
from core.logger import get_logger

#TODO сделать репозиторий

logger = get_logger(__name__)

class TestingVideoRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(self, session_id: str) -> Optional[TestingSession]:
        pass

    async def update(self, session_id: str, data: dict) -> Optional[TestingSession]:
        pass

    async def delete(self, session_id: str) -> Optional[TestingSession]:
        pass

    async def create(self, session_id: str, data: dict) -> Optional[TestingSession]:
        pass
