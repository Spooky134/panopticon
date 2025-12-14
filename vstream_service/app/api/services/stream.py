import os
import uuid
from uuid import UUID
from datetime import datetime
from datetime import datetime, timedelta, timezone

from attr.filters import exclude

from core.security.token import verify_token
from utils.streaming_session_manager import StreamingSessionManager
from api.schemas.sdp import SDPData
from storage.s3_storage import S3Storage
from db.repositories import StreamingSessionRepository, StreamingVideoRepository
from core.logger import get_logger
from utils.streaming_session import StreamingSession
from schemas.streaming_video import StreamingVideoORMCreate, VideoMeta

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

    async def offer(self, streaming_session_id: UUID, sdp_data: SDPData) -> dict:
        user_id = 1
        logger.info(f"session: {streaming_session_id} - Authorized user {user_id} starting stream")

        await self.streaming_session_manager.create_streaming_session(user_id=user_id,
                                                                      streaming_session_id=streaming_session_id,
                                                                      on_streaming_session_started=self._started_update,
                                                                      on_streaming_session_finished=self._finished_update)


        answer = await self.streaming_session_manager.start_streaming_session(streaming_session_id=streaming_session_id,
                                                                              sdp_data=sdp_data)

        return answer

    async def stop(self, streaming_session_id: UUID) -> dict:
        logger.info(f"session: {streaming_session_id} - type{type(streaming_session_id)}")
        try:
            await self.streaming_session_manager.dispose_streaming_session(streaming_session_id=streaming_session_id)
        except Exception as e:
            logger.error(f"streaming_session: {streaming_session_id} - stop error:{e}")
            return {
                "status": "error",
                "message": str(e)
            }
        return {
            "status": "success",
            "message": f"Stream session {streaming_session_id} stopped"
        }

    async def _started_update(self, streaming_session_id: UUID, started_at: datetime):
        await self.streaming_session_repository.update(streaming_session_id=streaming_session_id,
                                                       data={"status": "running",
                                                             "started_at": started_at})

    async def _finished_update(self,
                               streaming_session_id: UUID,
                               finished_at: datetime,
                               file_path: str,
                               file_name: str,
                               video_meta: dict):
        logger.info(f"session: {streaming_session_id} - saving results...")

        s3_key=None

        try:
            s3_key = await self._save_data_to_s3(streaming_session_id=streaming_session_id,
                                                 file_path=file_path,
                                                 object_name=file_name)
        except Exception as e:
            logger.error(f"session: {streaming_session_id} - error to save video to s3: {e}")

        meta = {key: video_meta[key] for key in video_meta.keys() if key not in ["duration",
                                                                                 "file_size",
                                                                                 "mime_type",
                                                                                 "width",
                                                                                 "height",
                                                                                 "fps",]}
        new_streaming_video = StreamingVideoORMCreate(
            streaming_session_id=streaming_session_id,
            s3_key=s3_key,
            s3_bucket=self.s3_storage.bucket_name,
            created_at=datetime.now(timezone.utc),
            duration=video_meta.get("duration", None),
            fps=video_meta.get("fps", None),
            file_size=video_meta.get("file_size", None),
            mime_type=video_meta.get("mime_type", None),
            width=video_meta.get("width", None),
            height=video_meta.get("height", None),
            meta=VideoMeta(**meta)
        )

        await self.streaming_video_repository.create(new_streaming_video=new_streaming_video)

        await self.streaming_session_repository.update(streaming_session_id=streaming_session_id,
                                                       data={"status": "finished",
                                                             "ended_at": finished_at})

    async def _save_data_to_s3(self, streaming_session_id: UUID, file_path:str, object_name:str):
        s3_key = None
        try:
            logger.info(f"session: {streaming_session_id} - loading {file_path} → {object_name}")
            s3_key = await self.s3_storage.upload_multipart(file_path=file_path, object_name=object_name)
            #TODO говорит все успешно хотя загрузки небыло
            logger.info(f"session: {streaming_session_id} - the video has been successfully uploaded to S3: {object_name}")
        except Exception as e:
            logger.error(f"session: {streaming_session_id} - error loading in: {e}")


        return s3_key