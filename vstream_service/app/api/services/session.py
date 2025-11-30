import os
import uuid
from datetime import datetime, timedelta
import jwt

from api.schemas.session import SessionCreate, SessionResponse
from utils.session_manager import SessionManager
from api.schemas.sdp import SDPData
from storage.s3_storage import S3Storage
from db.repositories import TestingSessionRepository, TestingVideoRepository
from core.logger import get_logger
from utils.session import Session
from api.exceptions.exeptions import NotFoundError
from config.settings import settings



logger = get_logger(__name__)

class SessionService:
    def __init__(self,
                 testing_session_repository: TestingSessionRepository = None,
                 testing_video_repository: TestingVideoRepository = None,
                 ):
        self.testing_session_repository = testing_session_repository
        self.testing_video_repository = testing_video_repository


    async def create_session(self, session_create: SessionCreate) -> SessionResponse:
        #TODO проверить есть ли сесиия чтобы не создавать еще одну при повторном запросе
        #TODO где давать id для сессии

        new_session = {
            "id": uuid.uuid4(),
            "test_id": uuid.uuid4(),
            "user_id": session_create.user_id,
            "status": "created",
        }

        session = await self.testing_session_repository.create(session_data=new_session)
        if session:
            logger.info(f"session: {session.id} - created")

        #TODO разобраться со временем для токенов
        payload = {
            "user_id": session_create.user_id,
            "session_id": str(session.id),
            "exp": datetime.now() + timedelta(minutes=1000)
        }

        token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

        return SessionResponse(session_id=session.id, token=token)


    async def read_session(self, session_id: uuid.UUID):
        session = await self.testing_session_repository.get(session_id=session_id)

        if not session:
            raise NotFoundError

        return session