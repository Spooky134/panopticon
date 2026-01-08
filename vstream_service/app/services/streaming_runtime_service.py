from typing import Optional
from uuid import UUID
from datetime import datetime, timezone

from core.engine.live_streaming_session_manager import LiveStreamingSessionManager
from core.entities.sdp_data import SDPData
from core.entities.streaming_session_data import StreamingSessionData
from infrastructure.s3.s3_video_storage import S3VideoStorage
from core.logger import get_logger
from core.entities.streaming_video_data import StreamingVideoData, VideoMetaData
from core.engine.live_streaming_session_status import LiveStreamingSessionStatus
from services.streaming_session_lifecycle_service import StreamingSessionLifecycleService

logger = get_logger(__name__)

# TODO выгрузка в s3 отдельный задачу
# TODO Транзакции
class StreamingRuntimeService:
    def __init__(self,
                 streaming_session_manager: LiveStreamingSessionManager,
                 streaming_session_lifecycle_service: StreamingSessionLifecycleService,
                 s3_video_storage: S3VideoStorage = None,
                 ):
        self._streaming_session_lifecycle_service = streaming_session_lifecycle_service
        self._streaming_session_manager = streaming_session_manager
        self._s3_video_storage = s3_video_storage

    async def offer(self, streaming_session_id: UUID, sdp_data: SDPData) -> SDPData:
        _, streaming_session_data  = await self._streaming_session_lifecycle_service.read_session(streaming_session_id)
        user_id = streaming_session_data.user_id
        logger.info(f"session: {streaming_session_id} - authorized user {user_id} starting stream")

        await self._streaming_session_manager.create_streaming_session(streaming_session_id=streaming_session_id,
                                                                      on_streaming_session_started=self._started_update,
                                                                      on_streaming_session_finished=self._finished_update)

        sdp_data_answer = await (self._streaming_session_manager
                                 .start_streaming_session(streaming_session_id=streaming_session_id, sdp_data=sdp_data))

        return sdp_data_answer

    async def stop(self, streaming_session_id: UUID) -> dict:
        logger.info(f"session: {streaming_session_id} - type{type(streaming_session_id)}")
        try:
            await self._streaming_session_manager.dispose_streaming_session(streaming_session_id=streaming_session_id)
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
        streaming_session_data = StreamingSessionData(status=LiveStreamingSessionStatus.RUNNING, started_at=started_at)
        await self._streaming_session_lifecycle_service.update_session(streaming_session_id=streaming_session_id,
                                                                       streaming_session_data=streaming_session_data)

    async def _finished_update(self,
                               streaming_session_id: UUID,
                               finished_at: datetime,
                               file_path: str,
                               file_name: str,
                               video_meta: VideoMetaData):
        logger.info(f"session: {streaming_session_id} - saving results...")
        s3_key=None


        try:
            logger.info(f"session: {streaming_session_id} - loading {file_path} → {file_name}")
            s3_key = await self._s3_video_storage.upload_multipart(file_path=file_path, object_name=file_name)
            #TODO говорит все успешно хотя загрузки небыло
            logger.info(f"session: {streaming_session_id} - the video has been successfully uploaded to S3: {file_name}")
        except Exception as e:
            logger.error(f"session: {streaming_session_id} - error to save video to s3: {e}")



        streaming_video_data = StreamingVideoData(
            s3_key=s3_key,
            s3_bucket=self._s3_video_storage.bucket_name,
            created_at=datetime.now(timezone.utc),
            meta=video_meta
        )

        streaming_session_data = StreamingSessionData(status=LiveStreamingSessionStatus.FINISHED,
                                                      ended_at=finished_at)

        await self._streaming_session_lifecycle_service.update_session(streaming_session_id=streaming_session_id,
                                                                       streaming_session_data=streaming_session_data)
        await self._streaming_session_lifecycle_service.attached_session_video(streaming_session_id=streaming_session_id,
                                                                               streaming_video_data=streaming_video_data)
