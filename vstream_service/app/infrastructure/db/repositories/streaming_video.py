from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from infrastructure.db.models import StreamingVideo
from core.logger import get_logger
from schemas.streaming_video import StreamingVideoORMCreate
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

    async def create(self, new_streaming_video: StreamingVideoORMCreate) -> StreamingVideo:
        data_dict = new_streaming_video.model_dump()
        streaming_video = StreamingVideo(**data_dict)

        self.db.add(streaming_video)
        await self.db.commit()
        # await self.db.refresh(new_streaming_video)

        return await self.get(streaming_video.id)