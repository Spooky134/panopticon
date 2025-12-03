import asyncio
from datetime import datetime, timezone
from aiortc import RTCSessionDescription, RTCPeerConnection
from uuid import UUID
from aiortc import RTCConfiguration

from grpc_client.base_processor import BaseProcessor
from utils.base_frame_collector import BaseFrameCollector
from api.schemas.sdp import SDPData
from core.logger import get_logger
from utils.frame_collector import FrameCollector
from webrtc.video_transform_track import VideoTransformTrack


logger = get_logger(__name__)


class StreamingSession:
    def __init__(self,
                 session_id: UUID,
                 user_id: int,
                 ice_config,
                 on_disconnect,
                 peer_connection=None,
                 grpc_processor=None,
                 collector=None,
                 ):
        self._id = session_id
        self._user_id = user_id
        self._ice_config = ice_config

        self._on_disconnect = on_disconnect

        self._peer_connection = peer_connection
        self._peer_connection.on("track", self._on_track)
        self._peer_connection.on("iceconnectionstatechange", self._on_ice_state_change)

        self._grpc_processor = grpc_processor
        self._collector = collector

        self._started_at = None
        self._finished_at = None


    @property
    def id(self):
        return self._id


    @property
    def started_at(self):
        return self._started_at


    @property
    def finished_at(self):
        return self._finished_at


    async def _on_ice_state_change(self):
        state = self._peer_connection.iceConnectionState
        logger.info(f"session: {self._id} - ICE state → {state}")

        if state in ["failed", "closed", "disconnected"]:
            logger.info(f"session: {self._id} - ICE state → {state}")
            await self._on_disconnect(self._id)


    async def _on_track(self, track):
        logger.info(f"session: {self._id} - Track received: {track.kind}")
        if track.kind == "video":
            transformed = VideoTransformTrack(track, self._grpc_processor, self._collector)
            self._peer_connection.addTrack(transformed)


    async def start(self, sdp_data: SDPData):
        offer = RTCSessionDescription(sdp_data.sdp, sdp_data.type)
        await self._peer_connection.setRemoteDescription(offer)

        await asyncio.sleep(1)
        answer = await self._peer_connection.createAnswer()
        await self._peer_connection.setLocalDescription(answer)

        self._started_at = datetime.now(timezone.utc)
        logger.info(f"session: {self._id} - Started for user {self._user_id}")

        #TODO заменить на схему
        return {
            "sdp": self._peer_connection.localDescription.sdp,
            "type": self._peer_connection.localDescription.type,
        }


    async def finalize(self):
        self._finished_at = datetime.now(timezone.utc)

        video_file_path, video_file_name, video_meta = None, None, None

        try:
            logger.info(f"session: {self._id} - finalizing session resources...")
            await self._collector.finalize()
        except Exception as e:
            logger.error(f"session: {self._id} - finalize error: {e}")

        try:
            video_file_path = self._collector.output_file_path
            video_file_name = self._collector.file_name
            video_meta = await self._collector.get_metadata()
        except Exception as e:
            logger.error(f"session: {self._id} - metadata read error: {e}")

        return video_file_path, video_file_name, video_meta


    async def shutdown(self):
        logger.info(f"session: {self._id} - cleaning up")
        try:
            await self._collector.cleanup()
            logger.info(f"session: {self._id} - collector cleaned up")
        except Exception as e:
            logger.error(f"session: {self._id} - cleanup error: {e}")

        logger.info(f"session: {self._id} - shutdown complete")