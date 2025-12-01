from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import exists, select, update, delete
from sqlalchemy.orm import selectinload, joinedload
from db.models import StreamingVideo, StreamingSession
from core.logger import get_logger
import uuid


logger = get_logger(__name__)

class StreamingVideoRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(self, streaming_video_id: uuid.UUID) -> Optional[StreamingVideo]:
        streaming_video = await self.db.execute(
            select(StreamingVideo).
            options(joinedload(StreamingVideo.streaming_session)).
            where(StreamingVideo.id == streaming_video_id))
        return streaming_video.scalar_one_or_none()

    async def create(self, data: dict) -> StreamingVideo:
        new_streaming_video = StreamingVideo(**data)
        self.db.add(new_streaming_video)
        await self.db.commit()
        # await self.db.refresh(new_streaming_video)

        return await self.get(new_streaming_video.id)