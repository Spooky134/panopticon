from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import joinedload
from app.streaming_video.entities import StreamingVideoEntity
from app.streaming_video.models import StreamingVideoModel
from app.core.logger import get_logger
from uuid import UUID
from dataclasses import asdict

from app.streaming_session.entities import StreamingSessionEntity
from app.streaming_session.models import StreamingSessionModel



logger = get_logger(__name__)

#TODO оптимизировать
class SessionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(self, streaming_session_id: UUID) -> Optional[StreamingSessionEntity]:
        result = await self.db.execute(
            select(StreamingSessionModel).
            options(joinedload(StreamingSessionModel.video)).
            where(StreamingSessionModel.id == streaming_session_id))
        model = result.unique().scalar_one_or_none()
        if model:
            return StreamingSessionEntity.from_db(model)
        return None

    async def create(self, streaming_session_entity: StreamingSessionEntity) -> Optional[StreamingSessionEntity]:
        data_dict = asdict(streaming_session_entity)
        new_streaming_session = StreamingSessionModel(**data_dict)

        self.db.add(new_streaming_session)
        await self.db.commit()


        return await self.get(streaming_session_id=new_streaming_session.id)

    async def update(self, streaming_session_entity: StreamingSessionEntity) -> Optional[StreamingSessionEntity]:
        entity_data = asdict(streaming_session_entity)
        # TODO id тоже перезаписывается?
        await self.db.execute(
            update(StreamingSessionModel)
            .where(StreamingSessionModel.id == streaming_session_entity.id)
            .values(**entity_data)
        )

        await self.db.commit()

        logger.info(f"session: {streaming_session_entity.id} - updated")
        return await self.get(streaming_session_id=streaming_session_entity.id)


    async def get_all(self) -> List[StreamingSessionEntity]:
        result = await self.db.execute(
            select(StreamingSessionModel).
            options(joinedload(StreamingSessionModel.video))
        )
        models = result.unique().scalars().all()
        entities = [StreamingSessionEntity.from_db(model) for model in models if model]

        return entities


    async def delete(self, streaming_session_id: UUID) -> Optional[StreamingSessionEntity]:
        ...

    async def attach_video(self, session_id: UUID, video_entity: StreamingVideoEntity) -> None:
        streaming_video = StreamingVideoModel(
            streaming_session_id=session_id,
            s3_key=video_entity.s3_key,
            created_at=video_entity.created_at,
            duration=video_entity.meta.duration,
            fps=video_entity.meta.fps,
            width=video_entity.meta.width,
            height=video_entity.meta.height,
            file_size=video_entity.meta.file_size,
            mime_type=video_entity.meta.mime_type,
            meta=video_entity.meta.get_extra()
        )

        self.db.add(streaming_video)
        await self.db.commit()

        return video_entity