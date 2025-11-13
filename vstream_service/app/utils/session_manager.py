import asyncio
from aiortc import RTCConfiguration

from api.schemas.sdp import SDPData
from utils.session import Session
from utils.frame_collector import FrameCollector
from webrtc.video_transform_track import VideoTransformTrack
from webrtc.connection_manager import ConnectionManager
from grpc_client.processor_manager import ProcessorManager
from storage.s3_storage import S3Storage
from db.repositories.testing_session import TestingSessionRepository
from core.logger import get_logger


logger = get_logger(__name__)


class SessionManager:
    def __init__(self,
                 connection_manager: ConnectionManager,
                 processor_manager: ProcessorManager,
                 ice_servers,
                 s3_storage: S3Storage,
                 testing_session_repository: TestingSessionRepository = None,
                 ):
        self.connection_manager = connection_manager
        self.processor_manager = processor_manager
        self.s3_storage = s3_storage
        self.ice_servers = ice_servers
        self.testing_session_repository = testing_session_repository
        self.sessions: dict[str, Session] = {}

    def _register_event_handlers(self, session: Session):
        peer_connection = session.peer_connection
        session_id = session.session_id

        @peer_connection.on("track")
        async def on_track(track):
            logger.info(f"session: {session_id} - Track received: {track.kind}")
            if track.kind == "video":
                transformed = VideoTransformTrack(track, session.grpc_processor, session.collector)
                peer_connection.addTrack(transformed)

        @peer_connection.on("iceconnectionstatechange")
        async def on_ice_state_change():
            state = peer_connection.iceConnectionState
            logger.info(f"session: {session_id} - ICE state → {state}")

            if state in ["failed", "closed", "disconnected"]:
                logger.info(f"session: {session_id} - ICE state → {state}")
                await self._dispose_session(session_id)

    async def initiate_session(self, user_id:str, session_id: str, sdp_data: SDPData) -> dict:
        if session_id in self.sessions:
            await self._dispose_session(session_id)

        await self.s3_storage.ensure_bucket()

        peer_connection = await self.connection_manager.create_connection(
            session_id=session_id,
            rtc_config=RTCConfiguration(iceServers=self.ice_servers)
        )
        grpc_processor = await self.processor_manager.create_processor(
            session_id=session_id
        )
        collector = FrameCollector(
            session_id=session_id,
            s3_storage=self.s3_storage)

        session = Session(
            session_id=session_id,
            user_id=user_id,
            peer_connection=peer_connection,
            video_processor=grpc_processor,
            collector=collector
        )

        self._register_event_handlers(session)

        answer = await session.start(sdp_data=sdp_data)

        self.sessions[session_id] = session
        logger.info(f"session: {session_id} - Created for user {user_id}")

        testing_session = await self.testing_session_repository.update(session_id=session_id,
                                                                       data={"status": "running",
                                                                             "started_at": self.sessions.get(session_id).started_at})
        return answer

    async def _dispose_session(self, session_id: str):
        session = self.sessions.get(session_id)
        if not session:
            return

        logger.info(f"session: {session_id} - Cleaning up")

        try:
            await session.finalize()
        except Exception as e:
            logger.error(f"session: {session_id} - Finalize error: {e}")

        await self.testing_session_repository.update(session_id, {
            "status": "finished",
            "ended_at": self.sessions.get(session_id).finished_at,
            "video_url": f"s3://bucket/{session_id}.mp4",
        })

        await asyncio.gather(
            self.processor_manager.close_processor(session_id),
            self.connection_manager.close_connection(session_id),
            return_exceptions=True,
        )

        self.sessions.pop(session_id, None)
        logger.info(f"session: {session_id} - Cleaned up")

    async def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    async def dispose_all_sessions(self):
        for session_id in self.sessions.keys():
            await self._dispose_session(session_id=session_id)