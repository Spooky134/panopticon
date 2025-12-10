import os
import uuid
from datetime import datetime, timedelta, timezone
import jwt

from api.schemas.streaming_session import StreamingSessionCreate, StreamingSessionResponse
from db.repositories import StreamingSessionRepository, StreamingVideoRepository
from core.logger import get_logger
from api.exceptions.exeptions import NotFoundError
from config.settings import settings



logger = get_logger(__name__)

class StreamingSessionService:
    def __init__(self,
                 streaming_session_repository: StreamingSessionRepository = None,
                 streaming_video_repository: StreamingVideoRepository = None,
                 ):
        self.streaming_session_repository = streaming_session_repository
        self.streaming_video_repository = streaming_video_repository


    async def create_session(self, streaming_session_create: StreamingSessionCreate) -> StreamingSessionResponse:
        #TODO проверить есть ли сесиия чтобы не создавать еще одну при повторном запросе
        #TODO где давать id для сессии

        new_streaming_session = {
            "id": uuid.uuid4(),
            "test_id": uuid.uuid4(),
            "user_id": streaming_session_create.user_id,
            "status": "created",
            "created_at": datetime.now(timezone.utc),
        }

        streaming_session_created = await self.streaming_session_repository.create(streaming_session_data=new_streaming_session)
        if streaming_session_created:
            logger.info(f"session: {streaming_session_created.id} - created")


        return StreamingSessionResponse(streaming_session_id=streaming_session_created.id)


    async def read_session(self, streaming_session_id: uuid.UUID):
        session = await self.streaming_session_repository.get(streaming_session_id=streaming_session_id)

        if not session:
            raise NotFoundError

        return session