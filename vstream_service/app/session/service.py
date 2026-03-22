from typing import List
from uuid import UUID

from app.exceptions import NotFoundError
from app.video.entities import StreamingVideoEntity, VideoMetaEntity
from app.session.entities import StreamingSessionEntity, UpdateSessionEntity, CreateSessionEntity
from app.session.repository import SessionRepository
from app.core.logger import get_logger

logger = get_logger(__name__)

class SessionService:
    def __init__(self, session_repo: SessionRepository):
        self._session_repo = session_repo

    async def create_session(self, session_id: UUID) -> StreamingSessionEntity:
        session_entity = await self._session_repo.get(session_id)
        if session_entity:
            logger.info(f"session: {session_entity.id} - existed.")
            return session_entity

        session_entity = CreateSessionEntity(session_id)

        created_session = await self._session_repo.create(session_entity)
        if created_session:
            logger.info(f"session: {session_id} - created.")

        return session_entity


    async def get_one_session(self, session_id: UUID) -> StreamingSessionEntity:
        session_entity = await self._session_repo.get(session_id)

        if session_entity is None:
            raise NotFoundError

        return session_entity


    async def get_all_sessions(self, skip: int = 0, limit: int = 10) -> List[StreamingSessionEntity]:
        all_session = await self._session_repo.get_all(skip, limit)

        return all_session


    async def update_session(self, session_id: UUID, entity: UpdateSessionEntity) -> StreamingSessionEntity:
        session_entity = await self._session_repo.get(session_id)

        if session_entity is None:
            raise NotFoundError
        
        session_entity = await self._session_repo.update(session_id, entity)

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



