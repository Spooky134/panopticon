import asyncio
import uuid

from aiortc import RTCConfiguration

from api.schemas.sdp import SDPData
from utils.frame_collector_factory import FrameCollectorFactory
from utils.streaming_session import StreamingSession
from utils.frame_collector import FrameCollector

from webrtc.connection_manager import ConnectionManager
from grpc_client.processor_manager import ProcessorManager
from core.logger import get_logger
from datetime import datetime
from config import settings
from uuid import UUID
from utils.frame_collector import FrameCollector


logger = get_logger(__name__)


class StreamingSessionManager:
    def __init__(self,
                 connection_manager: ConnectionManager,
                 processor_manager: ProcessorManager,
                 collector_factory: FrameCollectorFactory,
                 ice_servers):

        self.connection_manager = connection_manager
        self.processor_manager = processor_manager
        self.collector_factory = collector_factory
        self.ice_config = ice_servers

        self.streaming_sessions: dict[UUID, StreamingSession] = {}

        self._on_streaming_session_started = None
        self._on_streaming_session_finished = None

    async def create_streaming_session(self, user_id:int,
                                       streaming_session_id: UUID,
                                       on_streaming_session_started=None,
                                       on_streaming_session_finished=None) -> uuid.UUID:

        self._on_streaming_session_started = on_streaming_session_started
        self._on_streaming_session_finished = on_streaming_session_finished

        peer_connection = await self.connection_manager.create_connection(session_id=streaming_session_id,
                                                                          rtc_config=RTCConfiguration(
                                                                              iceServers=self.ice_config))
        grpc_processor = await self.processor_manager.create_processor(session_id=streaming_session_id)

        collector = self.collector_factory.create(session_id=streaming_session_id)

        session = StreamingSession(
            user_id=user_id,
            session_id=streaming_session_id,
            ice_config=self.ice_config,

            peer_connection = peer_connection,
            grpc_processor = grpc_processor,
            collector = collector,

            on_disconnect = self.dispose_streaming_session
        )

        self.streaming_sessions[streaming_session_id] = session
        logger.info(f"session: {streaming_session_id} - Created for user {user_id}")

        return session.id

    async def start_streaming_session(self, streaming_session_id: UUID, sdp_data: SDPData) -> dict:
        session = self.streaming_sessions.get(streaming_session_id)
        if session is None:
            raise ValueError("session not found")


        answer = await session.start(sdp_data=sdp_data)

        await self._on_streaming_session_started(streaming_session_id=session.id,
                                                 started_at=session.started_at)
        return answer


    async def dispose_streaming_session(self, streaming_session_id: UUID):
        logger.info(f"session: {streaming_session_id} - Cleaning up")
        session = self.streaming_sessions.pop(streaming_session_id, None)

        video_file_path, video_file_name, video_meta = await session.finalize()
        try:
            await self._on_streaming_session_finished(streaming_session_id=session.id,
                                                      finished_at=session.finished_at,
                                                      file_path=video_file_path,
                                                      file_name=video_file_name,
                                                      video_meta=video_meta)
        except Exception as e:
            logger.error(f"session: {streaming_session_id} - on_finished callback error: {e}")


        if session:
            await session.shutdown()


        await asyncio.gather(
            self.connection_manager.close_connection(session_id=streaming_session_id),
            self.processor_manager.close_processor(session_id=streaming_session_id),
            return_exceptions=True,
        )

    async def get_streaming_session(self, streaming_session_id: UUID) -> StreamingSession:
        return self.streaming_sessions.get(streaming_session_id)

    async def dispose_all_sessions(self):
        for streaming_session_id in self.streaming_sessions.keys():
            await self.dispose_streaming_session(streaming_session_id=streaming_session_id)