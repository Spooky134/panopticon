from typing import Callable
from uuid import UUID

from core.entities.sdp_data import SDPEntity
from infrastructure.video.frame_collector_factory import FrameCollectorFactory
from core.engine.live_streaming_session import LiveStreamingSession
from infrastructure.triton_proccessor.video_processor_factory import VideoProcessorFactory
from functools import partial

from core.logger import get_logger
from infrastructure.webrtc.connection_factory import ConnectionFactory


logger = get_logger(__name__)

class LiveStreamingSessionManager:
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


    async def create_streaming_session(self,
                                       streaming_session_id: UUID,
                                       on_finished=None) -> UUID:
        if len(self._streaming_sessions) >= self._max_sessions:
            raise Exception("server busy: too many active sessions")

        peer_connection = self._connection_factory.create()

        grpc_processor = self._processor_factory.create(streaming_session_id=streaming_session_id)

        collector = self._collector_factory.create(session_id=streaming_session_id)

        on_disconnect_callback = None
        if on_finished is not None:
            on_disconnect_callback = partial(self.dispose_streaming_session,
                                             streaming_session_id=streaming_session_id,
                                             on_finished=on_finished)

        session = LiveStreamingSession(
            session_id=streaming_session_id,
            peer_connection = peer_connection,
            grpc_processor = grpc_processor,
            collector = collector,
            on_disconnect = on_disconnect_callback
        )

        self._streaming_sessions[streaming_session_id] = session

        return streaming_session_id

    async def start_streaming_session(self, streaming_session_id: UUID,
                                      sdp_data: SDPEntity, on_started: Callable=None) -> SDPEntity:
        session = await self.get_streaming_session(streaming_session_id)
        if session is None:
            raise Exception(f"session: {streaming_session_id} - not found")

        sdp_data_answer = await session.start(sdp_data=sdp_data)
        if on_started is not None:
            await on_started(streaming_session_id=session.id,started_at=session.started_at)

        return sdp_data_answer

    async def dispose_streaming_session(self, streaming_session_id: UUID, on_finished: Callable=None):
        logger.info(f"session: {streaming_session_id} - cleaning up")
        session = self._streaming_sessions.pop(streaming_session_id, None)
        if session is not None:
            video_file_path, video_meta = await session.shutdown()
            if on_finished is not None:
                try:
                    await on_finished(streaming_session_id=streaming_session_id,
                                      finished_at=session.finished_at,
                                      file_path=video_file_path,
                                      video_meta=video_meta)
                except Exception as e:
                    logger.error(
                        f"session: {streaming_session_id} - _on_finished callback error: {e}"
                    )
        else:
            raise Exception(f"session: {streaming_session_id} - not found error")

    async def get_streaming_session(self, streaming_session_id: UUID) -> LiveStreamingSession:
        return self._streaming_sessions.get(streaming_session_id, None)

    async def dispose_all_sessions(self):
        for streaming_session_id in self._streaming_sessions.keys():
            await self.dispose_streaming_session(streaming_session_id=streaming_session_id)