from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import exists, select, update, delete
from sqlalchemy.orm import selectinload, joinedload
from db.models import TestingVideo, TestingSession
from core.logger import get_logger
import uuid


logger = get_logger(__name__)

class TestingVideoRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(self, testing_video_id) -> Optional[TestingVideo]:
        result = await self.db.execute(
            select(TestingVideo).
            options(joinedload(TestingVideo.testing_session)).
            where(TestingVideo.id == testing_video_id))
        return result.scalar_one_or_none()

    async def create(self, data: dict) -> TestingVideo:
        new_video = TestingVideo(**data)
        self.db.add(new_video)
        await self.db.commit()
        # await self.db.refresh(new_video)

        return await self.get(new_video.id)