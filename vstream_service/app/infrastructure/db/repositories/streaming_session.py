from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import joinedload
from core.logger import get_logger
from uuid import UUID
from dataclasses import asdict

from core.entities.streaming_session import StreamingSessionEntity
from infrastructure.db.models import StreamingSessionModel


logger = get_logger(__name__)

#TODO оптимизировать
class StreamingSessionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(self, streaming_session_id: UUID) -> Optional[StreamingSessionEntity]:
        result = await self.db.execute(
            select(StreamingSessionModel).
            options(joinedload(StreamingSessionModel.video)).
            where(StreamingSessionModel.id == streaming_session_id))
        model = result.scalar_one_or_none()
        return StreamingSessionEntity.from_db(model)

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

    async def delete(self, streaming_session_id: UUID) -> Optional[StreamingSessionEntity]:
        ...