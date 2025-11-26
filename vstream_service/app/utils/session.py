import asyncio
from datetime import datetime
from aiortc import RTCSessionDescription, RTCPeerConnection
from uuid import UUID

from grpc_client.base_processor import BaseProcessor
from utils.base_frame_collector import BaseFrameCollector
from api.schemas.sdp import SDPData
from core.logger import get_logger
from webrtc.video_transform_track import VideoTransformTrack


logger = get_logger(__name__)


class Session:
    def __init__(self, session_id:UUID, user_id:str, peer_connection: RTCPeerConnection,
                 video_processor: BaseProcessor, collector: BaseFrameCollector=None):
        self.session_id = session_id
        self.user_id = user_id
        self.peer_connection = peer_connection
        self.grpc_processor = video_processor
        self.collector = collector
        self.started_at = None
        self.finished_at = None
        self._is_finalized = False


    async def start(self, sdp_data: SDPData) -> dict:
        offer = RTCSessionDescription(sdp_data.sdp, sdp_data.type)
        await self.peer_connection.setRemoteDescription(offer)
        await asyncio.sleep(1)
        answer = await self.peer_connection.createAnswer()
        await self.peer_connection.setLocalDescription(answer)

        self.started_at = datetime.now()
        logger.info(f"session: {self.session_id} - Started for user {self.user_id}")

        #TODO заменить на схему
        return {
            "sdp": self.peer_connection.localDescription.sdp,
            "type": self.peer_connection.localDescription.type,
        }

    def register_event_handlers(self, on_disconnect):
        @self.peer_connection.on("track")
        async def on_track(track):
            logger.info(f"session: {self.session_id} - Track received: {track.kind}")
            if track.kind == "video":
                transformed = VideoTransformTrack(track, self.grpc_processor, self.collector)
                self.peer_connection.addTrack(transformed)

        @self.peer_connection.on("iceconnectionstatechange")
        async def on_ice_state_change():
            state = self.peer_connection.iceConnectionState
            logger.info(f"session: {self.session_id} - ICE state → {state}")

            if state in ["failed", "closed", "disconnected"]:
                logger.info(f"session: {self.session_id} - ICE state → {state}")
                await on_disconnect(self.session_id)


    async def finalize(self):
        if self._is_finalized:
            return

        logger.info(f"session: {self.session_id} - Finalizing session resources...")
        self._is_finalized = True

        try:
            await self.collector.finalize()
        except Exception as e:
            logger.error(f"session: {self.session_id} - Collector finalize error: {e}")

        self.finished_at = datetime.now()