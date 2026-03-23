from uuid import UUID
from app.stream.engine.streaming_manager import StreamingManager
from app.core.logging import get_logger
from app.session.repository import SessionRepository
from app.core.unit_of_work import UnitOfWork

logger = get_logger(__name__)

# TODO типизация в колбэках
class SignalingStopUseCase:
    def __init__(
            self,
            streaming_manager: StreamingManager,
            session_repo_factory: type[SessionRepository],
            uow_factory: type[UnitOfWork],
    ):
        self._streaming_manager = streaming_manager
        self._session_repo_factory = session_repo_factory
        self._uow_factory = uow_factory

    async def execute(self, session_id: UUID) -> dict:
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
