from dataclasses import replace
from typing import List
from uuid import UUID
from datetime import datetime

from api.exceptions.exeptions import NotFoundError
from core.entities.streaming_video import StreamingVideoEntity, VideoMetaEntity
from core.entities.streaming_session import StreamingSessionEntity
from core.logger import get_logger
from infrastructure.db.repositories import StreamingSessionRepository, StreamingVideoRepository


logger = get_logger(__name__)

class StreamingSessionLifecycleService:
    def __init__(self,
                 session_factory):
        self._db_session_factory = session_factory

    async def create_session(self, streaming_session_id: UUID) -> StreamingSessionEntity:
        async with self._db_session_factory() as db_session:
            streaming_session_repository = StreamingSessionRepository(db=db_session)

            streaming_session_entity = await streaming_session_repository.get(streaming_session_id=streaming_session_id)
            if streaming_session_entity:
                logger.info(f"session: {streaming_session_entity.id} - existed.")
                return streaming_session_entity

            streaming_session_entity = StreamingSessionEntity(
                id=streaming_session_id
            )

            streaming_session_entity = await streaming_session_repository.create(
                streaming_session_entity=streaming_session_entity
            )
            if streaming_session_entity:
                logger.info(f"session: {streaming_session_entity.id} - created.")

            return streaming_session_entity


    async def get_one_session(self, streaming_session_id: UUID) -> StreamingSessionEntity:
        async with self._db_session_factory() as db_session:
            streaming_session_repository = StreamingSessionRepository(db=db_session)

            streaming_session_entity = await streaming_session_repository.get(streaming_session_id=streaming_session_id)

            if not streaming_session_entity:
                raise NotFoundError

            return streaming_session_entity


    async def get_all_sessions(self) -> List[StreamingSessionEntity]:
        async with self._db_session_factory() as db_session:
            streaming_session_repository = StreamingSessionRepository(db=db_session)

            all_streaming_session = await streaming_session_repository.get_all()

            return all_streaming_session


    async def update_session(self, streaming_session_id: UUID, status: str=None, started_at: datetime=None, ended_at: datetime=None) -> StreamingSessionEntity:
        async with self._db_session_factory() as db_session:
            streaming_session_repository = StreamingSessionRepository(db=db_session)

            streaming_session_entity = await streaming_session_repository.get(streaming_session_id=streaming_session_id)

            updated_data = {"status": status,
                            "started_at": started_at,
                            "ended_at": ended_at}
            updated_data = {k: v for k, v in updated_data.items() if v is not None}

            streaming_session_entity_updated = replace(
                streaming_session_entity,
                **updated_data
            )

            return await streaming_session_repository.update(
                streaming_session_entity=streaming_session_entity_updated
            )


    async def attach_video_to_session(self, streaming_session_id: UUID, s3_key: str, video_meta: VideoMetaEntity=None) -> StreamingVideoEntity:
        async with self._db_session_factory() as db_session:
            streaming_video_repository = StreamingVideoRepository(db=db_session)

            streaming_video_entity = StreamingVideoEntity(
                s3_key=s3_key,
                streaming_session_id=streaming_session_id,
                meta=video_meta
            )

            streaming_video_entity = await streaming_video_repository.create(
                streaming_video_entity=streaming_video_entity
            )

            return streaming_video_entity


