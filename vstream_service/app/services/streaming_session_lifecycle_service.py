import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import async_sessionmaker

from api.schemas.streaming_session import StreamingSessionCreateRequest, StreamingSessionResponse
from infrastructure.db.repositories import StreamingSessionRepository, StreamingVideoRepository
from core.logger import get_logger
from api.exceptions.exeptions import NotFoundError
from schemas.streaming_session import StreamingSessionORMCreate
from core.engine.live_streaming_session_status import LiveStreamingSessionStatus
from infrastructure.s3.s3_service import S3Service


logger = get_logger(__name__)

class StreamingSessionLifecycleService:
    def __init__(self,
                 session_factory,
                 s3_storage: S3Service = None,
                 ):
        self._session_factory = session_factory
        self.s3_storage = s3_storage


    async def create_session(self, streaming_session_create: StreamingSessionCreateRequest) -> StreamingSessionResponse:
        #TODO проверить есть ли сесиия чтобы не создавать еще одну при повторном запросе
        #TODO где давать id для сессии

        async with self._session_factory() as session:
            streaming_session_repository = StreamingSessionRepository(db=session)
            new_streaming_session = StreamingSessionORMCreate(id=uuid.uuid4(),
                                                              test_id=uuid.uuid4(),
                                                              user_id=streaming_session_create.user_id,
                                                              status=LiveStreamingSessionStatus.CREATED,
                                                              created_at=datetime.now(timezone.utc))

            streaming_session_created = await streaming_session_repository.create(streaming_session_data=new_streaming_session)
            if streaming_session_created:
                logger.info(f"session: {streaming_session_created.id} - created.")


            return StreamingSessionResponse(streaming_session_id=streaming_session_created.id)


    async def read_session(self, streaming_session_id: uuid.UUID):
        async with self._session_factory() as session:
            streaming_session_repository = StreamingSessionRepository(db=session)

            session = await streaming_session_repository.get(streaming_session_id=streaming_session_id)

            if not session:
                raise NotFoundError

            return session


    async def update_session(self, streaming_session_id: uuid.UUID, data: dict, new_streaming_video=None):
        async with self._session_factory() as session:
            streaming_session_repository = StreamingSessionRepository(db=session)
            streaming_video_repository = StreamingVideoRepository(db=session)

            if new_streaming_video:
                await streaming_video_repository.create(new_streaming_video=new_streaming_video)

            await streaming_session_repository.update(streaming_session_id=streaming_session_id, data=data)

            # await session.commit()
