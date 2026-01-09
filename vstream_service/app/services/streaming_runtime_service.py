from uuid import UUID
from datetime import datetime

from core.engine.live_streaming_session_manager import LiveStreamingSessionManager
from core.entities.sdp_data import SDPEntity
from infrastructure.s3.s3_video_storage import S3VideoStorage
from core.logger import get_logger
from core.entities.streaming_video import VideoMetaEntity
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

    async def offer(self, streaming_session_id: UUID, sdp_data: SDPEntity) -> SDPEntity:
        streaming_session_entity = await self._streaming_session_lifecycle_service.read_session(streaming_session_id)
        user_id = streaming_session_entity.user_id
        logger.info(f"session: {streaming_session_id} - authorized user {user_id} starting stream")

        await self._streaming_session_manager.create_streaming_session(
            streaming_session_id=streaming_session_id,
            on_streaming_session_started=self._started_update,
            on_streaming_session_finished=self._finished_update
        )

        sdp_data_answer = await self._streaming_session_manager.start_streaming_session(
            streaming_session_id=streaming_session_id,
            sdp_data=sdp_data
        )

        return sdp_data_answer

    async def stop(self, streaming_session_id: UUID) -> dict:
        logger.info(f"session: {streaming_session_id} - type {type(streaming_session_id)}")
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

    async def _started_update(self, streaming_session_id: UUID, started_at: datetime) -> None:
        await self._streaming_session_lifecycle_service.update_session(
            streaming_session_id=streaming_session_id,
            status=LiveStreamingSessionStatus.RUNNING,
            started_at=started_at
        )
    #TODO типизация в колбэках
    async def _finished_update(self,
                               streaming_session_id: UUID,
                               finished_at: datetime,
                               file_path: str,
                               video_meta: VideoMetaEntity):
        logger.info(f"session: {streaming_session_id} - saving results...")

        s3_key = await self._s3_video_storage.upload_multipart(
            file_path=file_path,
            streaming_session_id=streaming_session_id)


        await self._streaming_session_lifecycle_service.update_session(
            streaming_session_id=streaming_session_id,
            status=LiveStreamingSessionStatus.FINISHED,
            ended_at=finished_at,
        )

        await self._streaming_session_lifecycle_service.attach_video_to_session(
            streaming_session_id=streaming_session_id,
            s3_key=s3_key,
            video_meta=video_meta
        )
