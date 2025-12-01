from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import exists, select, update, delete
from sqlalchemy.orm import selectinload, joinedload
from db.models import StreamingSession
from core.logger import get_logger
import uuid


logger = get_logger(__name__)

class StreamingSessionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(self, streaming_session_id: uuid.UUID) -> Optional[StreamingSession]:
        result = await self.db.execute(
            select(StreamingSession).
            options(joinedload(StreamingSession.video)).
            where(StreamingSession.id == streaming_session_id))
        return result.scalar_one_or_none()

    # TODO поменять на схему session_data
    async def create(self, streaming_session_data: dict) -> Optional[StreamingSession]:
        new_streaming_session = StreamingSession(**streaming_session_data)
        self.db.add(new_streaming_session)
        await self.db.commit()


        return await self.get(streaming_session_id=new_streaming_session.id)

    async def update(self, streaming_session_id: uuid.UUID, data: dict) -> Optional[StreamingSession]:
        streaming_session = await self.get(streaming_session_id=streaming_session_id)
        if not streaming_session:
            logger.warning(f"testing_session: {streaming_session_id} -  Not found in DB.")
            return None

        for field, value in data.items():
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
