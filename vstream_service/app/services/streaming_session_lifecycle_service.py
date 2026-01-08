from typing import Tuple
from uuid import UUID
from datetime import datetime, timezone

from infrastructure.db.repositories import StreamingSessionRepository, StreamingVideoRepository
from core.logger import get_logger
from api.exceptions.exeptions import NotFoundError
from core.entities.streaming_session_data import StreamingSessionData
from core.engine.live_streaming_session_status import LiveStreamingSessionStatus


logger = get_logger(__name__)


class StreamingSessionLifecycleService:
    def __init__(self,
                 session_factory):
        self._session_factory = session_factory


    async def create_session(self, user_id: int, test_id: UUID) -> Tuple[UUID, StreamingSessionData]:
        #TODO проверить есть ли сесиия чтобы не создавать еще одну при повторном запросе

        async with self._session_factory() as session:
            streaming_session_repository = StreamingSessionRepository(db=session)
            new_streaming_session_data = StreamingSessionData(test_id=test_id,
                                                              user_id=user_id,
                                                              status=LiveStreamingSessionStatus.CREATED,
                                                              created_at=datetime.now(timezone.utc))

            streaming_session = await streaming_session_repository.create(streaming_session_data=new_streaming_session_data)
            if streaming_session:
                logger.info(f"session: {streaming_session.id} - created.")

            streaming_session_data = StreamingSessionData(user_id=streaming_session.user_id,
                                                          test_id=streaming_session.test_id,
                                                          created_at=streaming_session.created_at,
                                                          started_at=streaming_session.started_at,
                                                          ended_at=streaming_session.ended_at,
                                                          status=streaming_session.status)

            return streaming_session.id, streaming_session_data


    async def read_session(self, streaming_session_id: UUID) -> Tuple[UUID, StreamingSessionData]:
        async with self._session_factory() as session:
            streaming_session_repository = StreamingSessionRepository(db=session)

            streaming_session = await streaming_session_repository.get(streaming_session_id=streaming_session_id)

            if not streaming_session:
                raise NotFoundError

            streaming_session_data = StreamingSessionData(user_id=streaming_session.user_id,
                                                          test_id=streaming_session.test_id,
                                                          created_at=streaming_session.created_at,
                                                          started_at=streaming_session.started_at,
                                                          ended_at=streaming_session.ended_at,
                                                          status=streaming_session.status)

            return streaming_session.id, streaming_session_data


    async def update_session(self, streaming_session_id: UUID, streaming_session_data: StreamingSessionData):
        async with self._session_factory() as session:
            streaming_session_repository = StreamingSessionRepository(db=session)
            await streaming_session_repository.update(streaming_session_id=streaming_session_id, streaming_session_data=streaming_session_data)

            # await session.commit()

    async def attached_session_video(self, streaming_session_id: UUID, streaming_video_data):
        async with self._session_factory() as session:
            streaming_video_repository = StreamingVideoRepository(db=session)
            await streaming_video_repository.create(streaming_session_id=streaming_session_id, streaming_video_data=streaming_video_data)

