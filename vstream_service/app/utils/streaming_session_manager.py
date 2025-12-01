import asyncio

from aiortc import RTCConfiguration

from api.schemas.sdp import SDPData
from utils.streaming_session import StreamingSession
from utils.frame_collector import FrameCollector
from webrtc.connection_manager import ConnectionManager
from grpc_client.processor_manager import ProcessorManager
from core.logger import get_logger
from datetime import datetime
from uuid import UUID


logger = get_logger(__name__)


class StreamingSessionManager:
    def __init__(self,
                 connection_manager: ConnectionManager,
                 processor_manager: ProcessorManager,
                 ice_servers,
                 ):
        self.connection_manager = connection_manager
        self.processor_manager = processor_manager
        self.ice_servers = ice_servers
        self.streaming_sessions: dict[UUID, StreamingSession] = {}
        self.on_streaming_session_finished = None
        self.on_streaming_session_started = None


    async def initiate_session(self, user_id:int, streaming_session_id: UUID, sdp_data: SDPData, on_streaming_session_started=None, on_streaming_session_finished=None) -> dict:
        self.on_streaming_session_finished = on_streaming_session_finished
        self.on_streaming_session_started = on_streaming_session_started

        if streaming_session_id in self.streaming_sessions:
            await self._dispose_session(streaming_session_id)

        peer_connection = await self.connection_manager.create_connection(
            session_id=streaming_session_id,
            rtc_config=RTCConfiguration(iceServers=self.ice_servers)
        )
        grpc_processor = await self.processor_manager.create_processor(
            session_id=streaming_session_id
        )
        collector = FrameCollector(
            session_id=streaming_session_id)

        streaming_session = StreamingSession(
            session_id=streaming_session_id,
            user_id=user_id,
            peer_connection=peer_connection,
            video_processor=grpc_processor,
            collector=collector
        )

        streaming_session.register_event_handlers(on_disconnect=self._dispose_session)
        answer = await streaming_session.start(sdp_data=sdp_data)

        self.streaming_sessions[streaming_session_id] = streaming_session

        logger.info(f"session: {streaming_session_id} - Created for user {user_id}")

        await on_streaming_session_started(streaming_session=streaming_session)


        return answer

    async def _dispose_session(self, streaming_session_id: UUID):
        streaming_session = self.streaming_sessions.get(streaming_session_id)
        if not streaming_session:
            return

        logger.info(f"session: {streaming_session_id} - Cleaning up")

        try:
            await streaming_session.finalize()
        except Exception as e:
            logger.error(f"session: {streaming_session_id} - Finalize error: {e}")

        await self.on_streaming_session_finished(streaming_session=streaming_session)

        await asyncio.gather(
            self.processor_manager.close_processor(streaming_session_id),
            self.connection_manager.close_connection(streaming_session_id),
            return_exceptions=True,
        )

        self.streaming_sessions.pop(streaming_session_id, None)
        logger.info(f"session: {streaming_session_id} - Cleaned up")

    async def get_session(self, streaming_session_id: UUID) -> StreamingSession:
        return self.streaming_sessions.get(streaming_session_id)

    async def dispose_all_sessions(self):
        for streaming_session_id in self.streaming_sessions.keys():
            await self._dispose_session(streaming_session_id=streaming_session_id)