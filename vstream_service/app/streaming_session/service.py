from dataclasses import replace
from typing import List
from uuid import UUID
from datetime import datetime

from app.exceptions import NotFoundError
from app.streaming_video.entities import StreamingVideoEntity, VideoMetaEntity
from app.streaming_session.entities import StreamingSessionEntity
from app.streaming_session.repository import SessionRepository
from app.core.logger import get_logger

logger = get_logger(__name__)

class SessionService:
    def __init__(
            self,
            session_repo: SessionRepository,
    ):
        self._session_repo = session_repo

    async def create_session(self, session_id: UUID) -> StreamingSessionEntity:
        session_entity = await self._session_repo.get(session_id)
        if session_entity:
            logger.info(f"session: {session_entity.id} - existed.")
            return session_entity

        session_entity = StreamingSessionEntity(
            id=session_id
        )

        session_entity = await self._session_repo.create(
            session_entity
        )
        if session_entity:
            logger.info(f"session: {session_entity.id} - created.")

        await self._session_repo.db.commit()
        return session_entity


    async def get_one_session(self, session_id: UUID) -> StreamingSessionEntity:
        session_entity = await self._session_repo.get(session_id)

        if not session_entity:
            raise NotFoundError

        return session_entity


    async def get_all_sessions(self) -> List[StreamingSessionEntity]:
        all_session = await self._session_repo.get_all()

        return all_session


    async def update_session(self, session_id: UUID, status: str=None, started_at: datetime=None, ended_at: datetime=None) -> StreamingSessionEntity:
        session_entity = await self._session_repo.get(session_id)

        updated_data = {
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at
        }
        updated_data = {k: v for k, v in updated_data.items() if v is not None}

        session_entity_updated = replace(
            session_entity,
            **updated_data
        )

        session_entity = await self._session_repo.update(
            session_entity_updated
        )

        await self._session_repo.db.commit()

        return session_entity


    async def attach_video_to_session(self, session_id: UUID, s3_key: str, video_meta: VideoMetaEntity=None) -> None:
        video_entity = StreamingVideoEntity(
            s3_key=s3_key,
            streaming_session_id=session_id,
            meta=video_meta
        )

        await self._session_repo.attach_video(
            session_id,
            video_entity
        )

        await self._session_repo.db.commit()



