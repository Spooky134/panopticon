from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from infrastructure.db.models import StreamingVideo
from core.logger import get_logger
from core.entities.streaming_video_data import StreamingVideoData
from dataclasses import asdict
from uuid import UUID


logger = get_logger(__name__)

#TODO оптимизировать
class StreamingVideoRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(self, streaming_video_id: UUID) -> Optional[StreamingVideo]:
        streaming_video = await self.db.execute(
            select(StreamingVideo).
            options(joinedload(StreamingVideo.streaming_session)).
            where(StreamingVideo.id == streaming_video_id))
        return streaming_video.scalar_one_or_none()

    async def create(self, streaming_session_id: UUID, streaming_video_data: StreamingVideoData) -> StreamingVideo:
        streaming_video = StreamingVideo(
            streaming_session_id=streaming_session_id,
            s3_key=streaming_video_data.s3_key,
            s3_bucket=streaming_video_data.s3_bucket,
            created_at=streaming_video_data.created_at,
            duration=streaming_video_data.meta.duration,
            fps=streaming_video_data.meta.fps,
            width=streaming_video_data.meta.width,
            height=streaming_video_data.meta.height,
            file_size=streaming_video_data.meta.file_size,
            mime_type=streaming_video_data.meta.mime_type,
            meta=streaming_video_data.meta.get_extra()
        )

        self.db.add(streaming_video)
        await self.db.commit()
        # await self.db.refresh(new_streaming_video)

        return await self.get(streaming_video.id)