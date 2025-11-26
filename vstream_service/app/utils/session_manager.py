import asyncio

from aiortc import RTCConfiguration

from api.schemas.sdp import SDPData
from utils.session import Session
from utils.frame_collector import FrameCollector
from webrtc.connection_manager import ConnectionManager
from grpc_client.processor_manager import ProcessorManager
from core.logger import get_logger
from datetime import datetime
from uuid import UUID


logger = get_logger(__name__)


#TODO попробовать перенести сохранение в сервис api
#TODO обьедиенение сервисов сохранения в один????
class SessionManager:
    def __init__(self,
                 connection_manager: ConnectionManager,
                 processor_manager: ProcessorManager,
                 ice_servers,
                 ):
        self.connection_manager = connection_manager
        self.processor_manager = processor_manager
        self.ice_servers = ice_servers
        self.sessions: dict[UUID, Session] = {}
        self.on_session_finished = None


    async def initiate_session(self, user_id:str, session_id: UUID, sdp_data: SDPData, on_session_started=None, on_session_finished=None) -> dict:
        self.on_session_finished = on_session_finished
        if session_id in self.sessions:
            await self._dispose_session(session_id)

        peer_connection = await self.connection_manager.create_connection(
            session_id=session_id,
            rtc_config=RTCConfiguration(iceServers=self.ice_servers)
        )
        grpc_processor = await self.processor_manager.create_processor(
            session_id=session_id
        )
        collector = FrameCollector(
            session_id=session_id)

        session = Session(
            session_id=session_id,
            user_id=user_id,
            peer_connection=peer_connection,
            video_processor=grpc_processor,
            collector=collector
        )

        session.register_event_handlers(on_disconnect=self._dispose_session)
        answer = await session.start(sdp_data=sdp_data)

        self.sessions[session_id] = session

        logger.info(f"session: {session_id} - Created for user {user_id}")

        await on_session_started(session=session)


        return answer

    async def _dispose_session(self, session_id: UUID):
        session = self.sessions.get(session_id)
        if not session:
            return

        logger.info(f"session: {session_id} - Cleaning up")

        try:
            await session.finalize()
        except Exception as e:
            logger.error(f"session: {session_id} - Finalize error: {e}")


        await self.on_session_finished(session=session)
        # await self.save_session_result(session_id=session_id)

        await asyncio.gather(
            self.processor_manager.close_processor(session_id),
            self.connection_manager.close_connection(session_id),
            return_exceptions=True,
        )

        self.sessions.pop(session_id, None)
        logger.info(f"session: {session_id} - Cleaned up")


    async def get_session(self, session_id: UUID) -> Session:
        return self.sessions.get(session_id)

    async def dispose_all_sessions(self):
        for session_id in self.sessions.keys():
            await self._dispose_session(session_id=session_id)