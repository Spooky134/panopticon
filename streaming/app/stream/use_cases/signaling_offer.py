from uuid import UUID
from datetime import datetime
from app.stream.engine.streaming_manager import StreamingManager
from app.stream.engine.stream_status import StreamStatus
from app.stream.entities import SDPEntity
from app.core.logging import get_logger
from app.video.entities import VideoMetaEntity
from app.session.repository import SessionRepository
from app.exceptions import NotFoundError
from app.session.entities import UpdateSessionEntity
from app.stream.tasks import uploading_video_to_s3
from app.core.unit_of_work import UnitOfWork

logger = get_logger(__name__)


# TODO типизация в колбэках
# TODO Транзакции
class SignalingOfferUseCase:
    def __init__(
            self,
            streaming_manager: StreamingManager,
            session_repo_factory: type[SessionRepository],
            uow_factory: type[UnitOfWork],
    ):
        self._streaming_manager = streaming_manager
        self._session_repo_factory = session_repo_factory
        self._uow_factory = uow_factory

    async def execute(self, session_id: UUID, sdp_data: SDPEntity) -> SDPEntity:
        async with self._uow_factory() as uow:
            repo = self._session_repo_factory(uow.session)
            session_entity = await repo.get(session_id)

            if not session_entity:
                raise NotFoundError

        # if streaming_session_entity.status == StreamStatus.FINISHED:
        #     pass
        logger.info(f"session: {session_id} - starting stream")

        # TODO переименовать
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

    async def _started_update(self, session_id: UUID, started_at: datetime) -> None:
        async with self._uow_factory() as uow:
            repo = self._session_repo_factory(uow.session)
            session_entity = await repo.get(session_id)

            if session_entity is None:
                raise NotFoundError

            updated = UpdateSessionEntity(
                status=StreamStatus.RUNNING,
                started_at=started_at
            )

            session_entity = await repo.update(session_id, updated)

            await uow.commit()

    async def _finished_update(
            self,
            session_id: UUID,
            finished_at: datetime,
            file_path: str,
            video_meta: VideoMetaEntity
    ) -> None:
        logger.info(f"session: {session_id} - saving results...")
        async with self._uow_factory() as uow:
            repo = self._session_repo_factory(uow.session)

            session_entity = await repo.get(session_id)
            if not session_entity:
                raise NotFoundError


            session_entity_updated = UpdateSessionEntity(
                status=StreamStatus.FINISHED,
                ended_at=finished_at
            )

            session_entity = await repo.update(
                session_id,
                session_entity_updated
            )

            await uow.commit()

        await uploading_video_to_s3.kiq(
            session_id=session_id,
            file_path=file_path,
            width=video_meta.width,
            height=video_meta.height,
            duration=video_meta.duration,
            codec=video_meta.codec,
            file_size=video_meta.file_size,
            mime_type=video_meta.mime_type,
            fps=video_meta.fps,
            bit_rate=video_meta.bit_rate,
            frame_count=video_meta.frame_count,
        )
        logger.info(f"session: {session_id} - session updated, video upload started in background")

