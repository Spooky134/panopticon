from typing import Callable
from uuid import UUID
from functools import partial

from app.stream.entities import SDPEntity
from app.stream.engine.stream import Stream
from app.ml_processor.video_processor_type import ProcessorType
from app.ml_processor.video_processor_factory import VideoProcessorFactory
from app.video.recorder.frame_collector_factory import FrameCollectorFactory
from app.stream.webrtc.connection_factory import ConnectionFactory
from app.core.logging import get_logger

logger = get_logger(__name__)

class StreamingManager:
    def __init__(
            self,
            connection_factory: ConnectionFactory,
            processor_factory: VideoProcessorFactory,
            collector_factory: FrameCollectorFactory,
            max_sessions
        ):

        self._max_sessions = max_sessions

        self._collector_factory = collector_factory
        self._processor_factory = processor_factory
        self._connection_factory = connection_factory

        self._sessions: dict[UUID, Stream] = {}


    async def create_session(
            self,
            session_id: UUID,
            on_finished=None
        ) -> UUID:
        if len(self._sessions) >= self._max_sessions:
            raise Exception("server busy: too many active sessions")

        peer_connection = self._connection_factory.create(session_id)

        grpc_processor = self._processor_factory.create(
            streaming_session_id=session_id,
            processor_type=ProcessorType.MONITORING
        )

        collector = self._collector_factory.create(session_id)

        on_disconnect_callback = None
        if on_finished is not None:
            on_disconnect_callback = partial(
                self.dispose_session,
                session_id=session_id,
                on_finished=on_finished
            )

        session = Stream(
            session_id=session_id,
            peer_connection = peer_connection,
            grpc_processor = grpc_processor,
            collector = collector,
            on_disconnect = on_disconnect_callback
        )

        self._sessions[session_id] = session

        return session_id

    async def start_session(
            self, session_id: UUID,
            sdp_data: SDPEntity,
            on_started: Callable=None
        ) -> SDPEntity:

        session = await self.get_session(session_id)
        if session is None:
            raise Exception(f"session: {session_id} - not found")

        sdp_data_answer = await session.start(sdp_data)
        if on_started is not None:
            await on_started(
                session_id=session.id,
                started_at=session.started_at
            )

        return sdp_data_answer

    async def dispose_session(self, session_id: UUID, on_finished: Callable=None):
        logger.info(f"session: {session_id} - cleaning up")
        session = self._sessions.pop(session_id, None)
        if session is not None:
            video_file_path, video_meta = await session.shutdown()
            if on_finished is not None:
                try:
                    await on_finished(
                        session_id=session_id,
                        finished_at=session.finished_at,
                        file_path=video_file_path,
                        video_meta=video_meta
                    )
                except Exception as e:
                    logger.error(
                        f"session: {session_id} - _on_finished callback error: {e}"
                    )
        else:
            raise Exception(f"session: {session_id} - not found error")

    async def get_session(self, session_id: UUID) -> Stream:
        return self._sessions.get(session_id, None)

    async def dispose_all_sessions(self):
        for session_id in self._sessions.keys():
            await self.dispose_session(session_id=session_id)