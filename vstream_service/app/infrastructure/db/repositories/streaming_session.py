from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from infrastructure.db.models import StreamingSession
from core.logger import get_logger
from uuid import UUID
from core.entities.streaming_session_data import StreamingSessionData
from dataclasses import asdict


logger = get_logger(__name__)

#TODO оптимизировать
class StreamingSessionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(self, streaming_session_id: UUID) -> Optional[StreamingSession]:
        result = await self.db.execute(
            select(StreamingSession).
            options(joinedload(StreamingSession.video)).
            where(StreamingSession.id == streaming_session_id))
        return result.scalar_one_or_none()


    async def create(self, streaming_session_data: StreamingSessionData) -> Optional[StreamingSession]:
        data_dict = asdict(streaming_session_data)
        new_streaming_session = StreamingSession(**data_dict)

        self.db.add(new_streaming_session)
        await self.db.commit()


        return await self.get(streaming_session_id=new_streaming_session.id)

    async def update(self, streaming_session_id: UUID, streaming_session_data: StreamingSessionData) -> Optional[StreamingSession]:
        streaming_session = await self.get(streaming_session_id=streaming_session_id)
        if not streaming_session:
            logger.warning(f"testing_session: {streaming_session_id} -  Not found in DB.")
            return None

        streaming_session_data_dict = asdict(streaming_session_data)
        for field, value in streaming_session_data_dict.items():
            if value is not None:
                setattr(streaming_session, field, value)

        # session.ended_at = data.get("ended_at", session.ended_at)
        # session.started_at = data.get("started_at", session.started_at)
        # session.status = data.get("status", session.status)
        # session.meta = data.get("meta", session.meta)
        # session.ml_metrics = data.get("ml_metrics", session.ml_metrics)

        # streaming_session.time_update = datetime.now(timezone.utc)
        await self.db.commit()

        logger.info(f"testing_session: {streaming_session_id} - Updated")
        return await self.get(streaming_session_id=streaming_session_id)
