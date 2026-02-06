import asyncio
import os
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

# TODO типизация в колбэках
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
        streaming_session_entity = await self._streaming_session_lifecycle_service.get_one_session(
            streaming_session_id=streaming_session_id
        )
        # if streaming_session_entity.status == LiveStreamingSessionStatus.FINISHED:
        #     pass
        logger.info(f"session: {streaming_session_id} - starting stream")

        #TODO переименовать
        await self._streaming_session_manager.create_streaming_session(
            streaming_session_id=streaming_session_id,
            on_finished=self._finished_update
        )

        sdp_data_answer = await self._streaming_session_manager.start_streaming_session(
            streaming_session_id=streaming_session_id,
            sdp_data=sdp_data,
            on_started=self._started_update
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

    async def _finished_update(self,
                               streaming_session_id: UUID,
                               finished_at: datetime,
                               file_path: str,
                               video_meta: VideoMetaEntity):
        logger.info(f"session: {streaming_session_id} - saving results...")

        await self._streaming_session_lifecycle_service.update_session(
            streaming_session_id=streaming_session_id,
            status=LiveStreamingSessionStatus.FINISHED,
            ended_at=finished_at,
        )

        asyncio.create_task(
            self._update_video_background(
                streaming_session_id=streaming_session_id,
                file_path=file_path,
                video_meta=video_meta
            )
        )
        logger.info(f"session: {streaming_session_id} - session updated, video upload started in background")


    async def _update_video_background(self, streaming_session_id: UUID, file_path: str, video_meta: VideoMetaEntity):
        try:
            s3_key = await self._s3_video_storage.upload_multipart(
                streaming_session_id=streaming_session_id,
                file_path=file_path,
                object_name=str(streaming_session_id),
                mime_type=video_meta.mime_type
            )
            if s3_key is not None:
                await self._streaming_session_lifecycle_service.attach_video_to_session(
                    streaming_session_id=streaming_session_id,
                    s3_key=s3_key,
                    video_meta=video_meta
                )
                logger.info(f"session: {streaming_session_id} - background video attach success")
        except Exception as e:
            logger.error(f"session: {streaming_session_id} - background upload critical error: {e}")
        finally:
            # TODO нужно добавить умную очистку чтобы при ошибки не удалять файл а пробовать загрузить повторно
            # TODO также добавить очистку накопившихся файлов
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"session: {streaming_session_id} - temporary file removed")
            except Exception as e:
                logger.error(f"session: {streaming_session_id} - error removing temporary file: {e}")
