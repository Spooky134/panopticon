import os
from datetime import datetime

from core.security.token import verify_token
from utils.session_manager import SessionManager
from api.schemas.sdp import SDPData
from storage.s3_storage import S3Storage
from db.repositories import TestingSessionRepository, TestingVideoRepository
from core.logger import get_logger
from utils.session import Session


logger = get_logger(__name__)

# TODO добавить в таблицу с сессиями продолжительность сессии
class StreamService:
    def __init__(self,
                 session_manager: SessionManager,
                 s3_storage: S3Storage = None,
                 testing_session_repository: TestingSessionRepository = None,
                 testing_video_repository: TestingVideoRepository = None,
                 ):
        self.session_manager = session_manager
        self.s3_storage = s3_storage
        self.testing_session_repository = testing_session_repository
        self.testing_video_repository = testing_video_repository

    async def offer(self, token, sdp_data: SDPData) -> dict:
        payload = verify_token(token)
        user_id = payload["user_id"]
        session_id = payload["session_id"]
        logger.info(f"session: {session_id} - Authorized user {user_id} starting stream")

        answer = await self.session_manager.initiate_session(
            session_id=session_id,
            user_id=user_id,
            sdp_data=sdp_data,
            on_session_started=self._started_update,
            on_session_finished=self._finished_update,
        )

        return answer

    async def _started_update(self, session: Session):
        testing_session = await self.testing_session_repository.update(session_id=session.session_id,
                                                                       data={"status": "running",
                                                                             "started_at": session.started_at})

    async def _finished_update(self, session: Session):
        logger.info(f"StreamService: Saving session: id - {session.session_id} :)))))")

        s3_key, meta = self._save_data_to_s3(session)

        data = {
            "testing_session_id": session.session_id,
            "s3_key": s3_key,
            "s3_bucket": self.s3_storage.bucket_name,
            "duration": meta.get("duration"),
            "file_size": meta.get("file_size"),
            "mime_type": meta.get("mime_type"),
            "created_at": datetime.now()
        }

        await self.testing_video_repository.create(data=data)

        await self.testing_session_repository.update(session.session_id, {
            "status": "finished",
            "ended_at": session.finished_at,
        })

    async def _save_data_to_s3(self, session: Session):
        meta = None
        s3_key = None

        if session.collector:
            if session.collector.output_file:
                local_file = session.collector.output_file
                meta = await session.collector.get_metadata()

                object_name = f"{session.session_id}.mp4"
                try:
                    logger.info(f"session: {session.session_id} - Loading {local_file} → {object_name}")
                    s3_key = await self.s3_storage.upload_file(file_path=local_file, object_name=object_name)
                    # TODO удаление видео
                    os.remove(local_file)
                    logger.info(f"session: {session.session_id} - The video has been successfully uploaded to S3: {object_name}")
                except Exception as e:
                    logger.error(f"session: {session.session_id} - Error loading in: {e}")


        return s3_key, meta