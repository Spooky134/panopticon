from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from uuid import UUID

from core.entities.streaming_video import StreamingVideoEntity
from infrastructure.db.models import StreamingVideoModel


#TODO оптимизировать
class StreamingVideoRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(self, streaming_video_id: UUID) -> Optional[StreamingVideoEntity]:
        result = await self.db.execute(
            select(StreamingVideoModel).
            options(joinedload(StreamingVideoModel.streaming_session)).
            where(StreamingVideoModel.id == streaming_video_id))
        model = result.scalar_one_or_none()
        return StreamingVideoEntity.from_db(model)

    async def create(self, streaming_video_entity: StreamingVideoEntity) -> StreamingVideoEntity:

        streaming_video = StreamingVideoModel(
            streaming_session_id=streaming_video_entity.streaming_session_id,
            s3_key=streaming_video_entity.s3_key,
            created_at=streaming_video_entity.created_at,
            duration=streaming_video_entity.meta.duration,
            fps=streaming_video_entity.meta.fps,
            width=streaming_video_entity.meta.width,
            height=streaming_video_entity.meta.height,
            file_size=streaming_video_entity.meta.file_size,
            mime_type=streaming_video_entity.meta.mime_type,
            meta=streaming_video_entity.meta.get_extra()
        )

        self.db.add(streaming_video)
        await self.db.commit()
        return await self.get(streaming_video_id=streaming_video.id)