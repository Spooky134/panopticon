from uuid import UUID

from api.schemas.sdp import SDPData
from utils.frame_collector_factory import FrameCollectorFactory
from core.engine.live_streaming_session import LiveStreamingSession
from infrastructure.grpc_client.processor_factory import VideoProcessorFactory

from core.logger import get_logger

from infrastructure.webrtc.get_connection_factory import ConnectionFactory

logger = get_logger(__name__)



class LiveStreamingSessionManager:
    _instance = None

    @classmethod
    def get_instance(cls, **kwargs) -> "LiveStreamingSessionManager":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance


    def __init__(self,
                 connection_factory: ConnectionFactory,
                 processor_factory: VideoProcessorFactory,
                 collector_factory: FrameCollectorFactory,
                 max_sessions):

        self._max_sessions = max_sessions

        self._collector_factory = collector_factory
        self._processor_factory = processor_factory
        self._connection_factory = connection_factory

        self._streaming_sessions: dict[UUID, LiveStreamingSession] = {}

        self._on_streaming_session_started = None
        self._on_streaming_session_finished = None


    async def create_streaming_session(self, user_id:int,
                                       streaming_session_id: UUID,
                                       on_streaming_session_started=None,
                                       on_streaming_session_finished=None) -> UUID:
        if len(self._streaming_sessions) >= self._max_sessions:
            raise Exception("Server busy: too many active sessions")

        self._on_streaming_session_started = on_streaming_session_started
        self._on_streaming_session_finished = on_streaming_session_finished

        peer_connection = self._connection_factory.create()

        grpc_processor = self._processor_factory.create(streaming_session_id=streaming_session_id)

        # collector = self.collector_factory.create(session_id=streaming_session_id)
        collector = None

        session = LiveStreamingSession(
            user_id=user_id,
            session_id=streaming_session_id,
            peer_connection = peer_connection,
            grpc_processor = grpc_processor,
            collector = collector,
            on_disconnect = self.dispose_streaming_session
        )

        self._streaming_sessions[streaming_session_id] = session

        return streaming_session_id

    async def start_streaming_session(self, streaming_session_id: UUID, sdp_data: SDPData) -> dict:
        session: LiveStreamingSession = await self.get_streaming_session(streaming_session_id)
        if session is None:
            raise ValueError("session not found")

        answer = await session.start(sdp_data=sdp_data)

        await self._on_streaming_session_started(streaming_session_id=session.id,
                                                 started_at=session.started_at)
        return answer


    async def dispose_streaming_session(self, streaming_session_id: UUID):
        logger.info(f"session: {streaming_session_id} - cleaning up")
        session = self._streaming_sessions.pop(streaming_session_id, None)
        video_file_path, video_file_name, video_meta = await session.finalize()
        try:
            await self._on_streaming_session_finished(streaming_session_id=streaming_session_id,
                                                      finished_at=session.finished_at,
                                                      file_path=video_file_path,
                                                      file_name=video_file_name,
                                                      video_meta=video_meta)
        except Exception as e:
            logger.error(f"session: {streaming_session_id} - on_finished callback error: {e}")


        if session:
            await session.shutdown()



    async def get_streaming_session(self, streaming_session_id: UUID) -> LiveStreamingSession:
        return self._streaming_sessions.get(streaming_session_id, None)

    async def dispose_all_sessions(self):
        for streaming_session_id in self._streaming_sessions.keys():
            await self.dispose_streaming_session(streaming_session_id=streaming_session_id)