import asyncio
from dataclasses import replace
import os
from uuid import UUID
from datetime import datetime

from app.stream.engine.streaming_manager import StreamingManager
from app.stream.engine.stream_status import StreamStatus
from app.stream.entities import SDPEntity
from app.aws.s3_video_storage import S3VideoStorage
from app.core.logger import get_logger
from app.video.entities import StreamingVideoEntity, VideoMetaEntity
from app.session.repository import SessionRepository
from app.exceptions import NotFoundError
from vstream_service.app.core.unit_of_work import UnitOfWork
from vstream_service.app.session.entities import UpdateSessionEntity

logger = get_logger(__name__)

# TODO типизация в колбэках
# TODO Транзакции
class StreamingService:
    def __init__(
        self,
        streaming_manager: StreamingManager,
        session_repo_factory: type[SessionRepository],
        uow_factory: type[UnitOfWork],
        s3_storage: S3VideoStorage = None,
    ):
        self._streaming_manager = streaming_manager
        self._session_repo_factory = session_repo_factory
        self._uow_factory = uow_factory
        self._s3_storage = s3_storage
        

    async def offer(self, session_id: UUID, sdp_data: SDPEntity) -> SDPEntity:
        async with self._uow_factory as uow:
            repo = self._session_repo_factory(uow.session)
            session_entity = await repo.get(session_id)

            if not session_entity:
                raise NotFoundError

        # if streaming_session_entity.status == StreamStatus.FINISHED:
        #     pass
        logger.info(f"session: {session_id} - starting stream")

        #TODO переименовать
        await self._streaming_manager.create_session(
            session_id=session_id,
            on_finished=self._finished_update
        )

        sdp_data_answer = await self._streaming_manager.start_session(
            session_id=session_id,
            sdp_data=sdp_data,
            on_started=self._started_update
        )

        return sdp_data_answer

    async def stop(self, session_id: UUID) -> dict:
        logger.info(f"session: {session_id} - type {type(session_id)}")
        try:
            await self._streaming_manager.dispose_session(session_id=session_id)
        except Exception as e:
            logger.error(f"streaming_session: {session_id} - stop error:{e}")
            return {
                "status": "error",
                "message": str(e)
            }
        return {
            "status": "success",
            "message": f"Stream session {session_id} stopped"
        }

    async def _started_update(self, session_id: UUID, started_at: datetime) -> None:
        async with self._uow_factory as uow:
            repo = self._session_repo_factory(uow.session)
            session_entity = await repo.get(session_id)

            if session_entity is None:
                raise NotFoundError

            updated = UpdateSessionEntity(
                status=StreamStatus.RUNNING,
                started_at=started_at
            )

            session_entity = await self.repo.update(session_id, updated)

            await uow.commit()

    async def _finished_update(
        self,
        session_id: UUID,
        finished_at: datetime,
        file_path: str,
        video_meta: VideoMetaEntity
    ) -> None:
        logger.info(f"session: {session_id} - saving results...")
        async with self._uow_factory as uow:
            repo = self._session_repo_factory(uow.session)
            session_entity = await repo.get(session_id)
            session_entity_updated = replace(
                session_entity,
                status=StreamStatus.FINISHED,
                ended_at=finished_at
            )

            session_entity = await repo.update(
                session_entity_updated
            )

            await self.uow.commit()

            # TODO TASKIQ
        asyncio.create_task(
            self._update_video_background(
                session_id=session_id,
                file_path=file_path,
                video_meta=video_meta
            )
        )
        logger.info(f"session: {session_id} - session updated, video upload started in background")

    async def _update_video_background(self, session_id: UUID, file_path: str, video_meta: VideoMetaEntity):
        try:
            async with self._uow_factory as uow:
                repo = self._session_repo_factory(uow.session)
                s3_key = await self._s3_storage.upload_multipart(
                    streaming_session_id=session_id,
                    file_path=file_path,
                    object_name=str(session_id),
                    mime_type=video_meta.mime_type
                )
                if s3_key is not None:
                    video_entity = StreamingVideoEntity(
                        s3_key=s3_key,
                        streaming_session_id=session_id,
                        meta=video_meta
                    )

                    await repo.attach_video(
                        session_id,
                        video_entity
                    )
                    await uow.commit()
                    logger.info(f"session: {session_id} - background video attach success")
        except Exception as e:
            logger.error(f"session: {session_id} - background upload critical error: {e}")
        finally:
            # TODO нужно добавить умную очистку чтобы при ошибки не удалять файл а пробовать загрузить повторно
            # TODO также добавить очистку накопившихся файлов
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"session: {session_id} - temporary file removed")
            except Exception as e:
                logger.error(f"session: {session_id} - error removing temporary file: {e}")
