from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.video.entities import StreamingVideoEntity
from app.video.models import VideoModel
from app.core.logger import get_logger
from uuid import UUID
from dataclasses import asdict

from app.session.entities import StreamingSessionEntity, CreateSessionEntity, UpdateSessionEntity
from app.session.models import SessionModel
from vstream_service.app.core.repositories import BaseRepository
from vstream_service.app.core.types import UNSET
from vstream_service.app.session.mappers import StreamingSessionMapper


logger = get_logger(__name__)


class SessionRepository(BaseRepository):
    def __init__(self, db_session: AsyncSession):
        super().__init__(SessionModel, db_session)


    async def get(self, session_id: UUID) -> Optional[StreamingSessionEntity]:
        query = (
            select(self._model)
            .where(self._model.id == session_id)
            .options(selectinload(self._model.video))
        )
        res = await self._db_session.execute(query)

        session_orm = res.scalar_one_or_none()

        if session_orm is None:
            logger.debug(f"Streaming session id={session_id} not found in DB")
            return None
        return StreamingSessionMapper.to_entity(session_orm)


    async def get_all(self, offset: int = 0, limit: int = 100) -> List[StreamingSessionEntity]:
        session_orms = await self._list_by_filters(
            offset=offset, limit=limit
        )

        return StreamingSessionMapper.to_entities(session_orms)


    async def create(self, session_entity: CreateSessionEntity) -> Optional[StreamingSessionEntity]:
        payload = asdict(session_entity)
        session_orm = self._model(**payload)

        self.db.add(session_orm)
        await self._async_session.flush()
        await self._async_session.refresh(session_orm)

        return StreamingSessionMapper.to_entity(session_orm)
    

    async def update(self, session_id: int, session_entity: UpdateSessionEntity) -> Optional[StreamingSessionEntity]:
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
        return StreamingSessionEntity.to_entity(session_orm)
    

    async def delete(self, session_id: UUID) -> bool:
        return await self._delete_by_filters(
            self._model.id == session_id
        )


    async def attach_video(self, session_id: UUID, video_entity: StreamingVideoEntity) -> None:
        streaming_video = VideoModel(
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
        await self._db_session.flush()
        await self._db_session.refresh(streaming_video)

        return video_entity