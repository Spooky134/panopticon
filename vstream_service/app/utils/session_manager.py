import asyncio
from aiortc import RTCConfiguration

from schemas.sdp import SDPData
from utils.session import Session
from utils.frame_collector import FrameCollector
from webrtc.video_transform_track import VideoTransformTrack


class SessionManager:
    def __init__(self, connection_manager, processor_manager, s3_storage, ice_servers):
        self.connection_manager = connection_manager
        self.processor_manager = processor_manager
        self.s3_storage = s3_storage
        self.ice_servers = ice_servers
        self.sessions = {}

    async def create_session(self, user_id:str, session_id: str, sdp_data: SDPData):
        if session_id in self.sessions:
            await self._cleanup_session(session_id)

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
            grpc_processor=grpc_processor,
            collector=collector
        )

        self._register_event_handlers(session)

        answer = await session.start(sdp_data=sdp_data)

        self.sessions[session_id] = session
        print(f"[SessionManager] Created session {session_id} for user {user_id}")

        return answer

    def _register_event_handlers(self, session: Session):
        peer_connection = session.peer_connection
        session_id = session.session_id

        @peer_connection.on("track")
        async def on_track(track):
            print(f"[Session] Track received: {track.kind} for session {session_id}")
            if track.kind == "video":
                transformed = VideoTransformTrack(track, session.grpc_processor, session.collector)
                peer_connection.addTrack(transformed)

        @peer_connection.on("iceconnectionstatechange")
        async def on_ice_state_change():
            state = peer_connection.iceConnectionState
            print(f"[Session {session_id}] ICE state → {state} for session {session_id}")

            if state in ["failed", "closed", "disconnected"]:
                print(f"[Session] ICE inactive — cleaning up {session_id}")
                await self._cleanup_session(session_id)


    async def _cleanup_session(self, session_id: str):
        session = self.sessions.get(session_id)
        if not session:
            return

        print(f"[SessionManager] Cleaning up session {session_id}")

        try:
            await session.finalize()
        except Exception as e:
            print(f"[SessionManager] Session finalize error: {e}")

        await asyncio.gather(
            self.processor_manager.close_processor(session_id),
            self.connection_manager.close_connection(session_id),
            return_exceptions=True,
        )

        self.sessions.pop(session_id, None)
        print(f"[SessionManager] Session {session_id} cleaned up")


    async def get_session(self, session_id):
        return self.sessions.get(session_id)


    async def close_all(self):
        for session_id in self.sessions.keys():
            await self._cleanup_session(session_id=session_id)