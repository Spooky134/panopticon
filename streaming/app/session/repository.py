from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.video.entities import StreamingVideoEntity, CreateVideoEntity
from app.video.mappers import VideoMapper
from app.video.models import VideoModel
from app.core.logging import get_logger
from uuid import UUID
from dataclasses import asdict

from app.session.entities import StreamingSessionEntity, CreateSessionEntity, UpdateSessionEntity
from app.session.models import SessionModel
from app.core.repositories import BaseRepository
from app.core.types import UNSET
from app.session.mappers import SessionMapper


logger = get_logger(__name__)


class SessionRepository(BaseRepository):
    def __init__(self, db_session: AsyncSession):
        super().__init__(SessionModel, db_session)


    async def get(self, session_id: UUID) -> Optional[StreamingSessionEntity]:
        query = (
            select(self._model)
            .where(self._model.id == session_id)
            .options(selectinload(self._model.videos))
        )
        res = await self._db_session.execute(query)

        session_orm = res.scalar_one_or_none()

        if session_orm is None:
            logger.debug(f"Streaming session id={session_id} not found in DB")
            return None
        return SessionMapper.to_entity(session_orm)


    async def get_all(self, offset: int = 0, limit: int = 100) -> List[StreamingSessionEntity]:
        session_orms = await self._list_by_filters(
            offset=offset, limit=limit
        )

        return SessionMapper.to_entities(session_orms)


    async def create(self, session_entity: CreateSessionEntity) -> Optional[StreamingSessionEntity]:
        payload = asdict(session_entity)
        session_orm = self._model(**payload)

        self._db_session.add(session_orm)
        await self._db_session.flush()
        await self._db_session.refresh(session_orm)

        return SessionMapper.to_entity(session_orm)
    

    async def update(self, session_id: UUID, session_entity: UpdateSessionEntity) -> Optional[StreamingSessionEntity]:
        session_orm = await self._get_by_filters(
            self._model.id == session_id,
        )
        if not session_orm:
            return None
        
        update_data = asdict(session_entity)
        update_data = {
            key: value for key, value in update_data.items()
            if value is not UNSET
        }

        await self._update(session_orm, update_data)
        logger.info(f"session: {session_id} - updated")
        return SessionMapper.to_entity(session_orm)
    

    async def delete(self, session_id: UUID) -> bool:
        return await self._delete_by_filters(
            self._model.id == session_id
        )


    async def attach_video(self, session_id: UUID, video_entity: CreateVideoEntity) -> StreamingVideoEntity:
        streaming_video = VideoModel(
            session_id=session_id,
            duration=video_entity.meta.duration,
            fps=video_entity.meta.fps,
            width=video_entity.meta.width,
            height=video_entity.meta.height,
            file_size=video_entity.meta.file_size,
            mime_type=video_entity.meta.mime_type,
            meta=video_entity.meta.get_extra()
        )

        self._db_session.add(streaming_video)
        await self._db_session.flush()
        await self._db_session.refresh(streaming_video)

        return VideoMapper.to_entity(streaming_video)