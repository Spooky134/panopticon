import os
from datetime import datetime
from datetime import datetime, timedelta, timezone

from core.security.token import verify_token
from utils.streaming_session_manager import StreamingSessionManager
from api.schemas.sdp import SDPData
from storage.s3_storage import S3Storage
from db.repositories import StreamingSessionRepository, StreamingVideoRepository
from core.logger import get_logger
from utils.streaming_session import StreamingSession


logger = get_logger(__name__)


class StreamService:
    def __init__(self,
                 streaming_session_manager: StreamingSessionManager,
                 s3_storage: S3Storage = None,
                 streaming_session_repository: StreamingSessionRepository = None,
                 streaming_video_repository: StreamingVideoRepository = None,
                 ):
        self.streaming_session_manager = streaming_session_manager
        self.s3_storage = s3_storage
        self.streaming_session_repository = streaming_session_repository
        self.streaming_video_repository = streaming_video_repository

    async def offer(self, token, sdp_data: SDPData) -> dict:
        payload = verify_token(token)
        user_id = int(payload["user_id"])
        streaming_session_id = payload["streaming_session_id"]

        logger.info(f"session: {streaming_session_id} - Authorized user {user_id} starting stream")

        answer = await self.streaming_session_manager.initiate_session(
            streaming_session_id=streaming_session_id,
            user_id=int(user_id),
            sdp_data=sdp_data,
            on_streaming_session_started=self._started_update,
            on_streaming_session_finished=self._finished_update,
        )

        return answer


    async def _started_update(self, streaming_session: StreamingSession):
        streaming_session_updated = await self.streaming_session_repository.update(streaming_session_id=streaming_session.id,
                                                                                   data={
                                                                                       "status": "running",
                                                                                       "started_at": streaming_session.started_at
                                                                                   })

    async def _finished_update(self, streaming_session: StreamingSession):
        logger.info(f"StreamService: Saving session: id - {streaming_session.id}")

        s3_key, meta = await self._save_data_to_s3(streaming_session)

        data = {
            "streaming_session_id": streaming_session.id,
            "s3_key": s3_key,
            "s3_bucket": self.s3_storage.bucket_name,
            "duration": meta.get("duration"),
            "file_size": meta.get("file_size"),
            "mime_type": meta.get("mime_type"),
            "created_at": datetime.now(timezone.utc)
        }

        await self.streaming_video_repository.create(data=data)

        await self.streaming_session_repository.update(streaming_session_id=streaming_session.id,
                                                       data={"status": "finished",
                                                             "ended_at": streaming_session.finished_at})

    async def _save_data_to_s3(self, streaming_session: StreamingSession):
        meta = None
        s3_key = None

        if streaming_session.collector and streaming_session.collector.file_exists():
            file_path = streaming_session.collector.output_file_path
            object_name = streaming_session.collector.file_name
            meta = await streaming_session.collector.get_metadata()
            try:
                logger.info(f"session: {streaming_session.id} - Loading {file_path} → {object_name}")
                s3_key = await self.s3_storage.upload_multipart(file_path=file_path, object_name=object_name)
                logger.info(f"session: {streaming_session.id} - The video has been successfully uploaded to S3: {object_name}")
            except Exception as e:
                logger.error(f"session: {streaming_session.id} - Error loading in: {e}")


        return s3_key, meta