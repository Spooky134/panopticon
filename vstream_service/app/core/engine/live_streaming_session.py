import asyncio
from datetime import datetime, timezone
from typing import Callable, Awaitable, Optional, Tuple

from aiortc import RTCSessionDescription, RTCPeerConnection
from uuid import UUID

from core.entities.sdp_data import SDPData
from core.entities.streaming_video_data import VideoMetaData
from core.logger import get_logger
from infrastructure.video.frame_collector import FrameCollector
from infrastructure.grpc_client.video_processor import VideoProcessor
from infrastructure.webrtc.video_transform_track import VideoTransformTrack


logger = get_logger(__name__)


class LiveStreamingSession:
    def __init__(self,
                 session_id: UUID,
                 peer_connection: RTCPeerConnection,
                 grpc_processor: VideoProcessor,
                 collector: FrameCollector,
                 on_disconnect: Callable[[UUID], Awaitable[None]],
                 ):

        self._id = session_id
        self._peer_connection = peer_connection
        self._grpc_processor = grpc_processor
        self._collector = collector
        self._on_disconnect = on_disconnect

        self._started_at: datetime | None = None
        self._finished_at: datetime | None = None

        self._peer_connection.on("track", self._on_track)
        self._peer_connection.on("iceconnectionstatechange", self._on_ice_state_change)


    @property
    def id(self) -> UUID:
        return self._id


    @property
    def started_at(self) -> datetime:
        return self._started_at


    @property
    def finished_at(self) -> datetime:
        return self._finished_at


    async def _on_ice_state_change(self):
        state = self._peer_connection.iceConnectionState
        logger.info(f"session: {self._id} - ICE state → {state}")

        if state in ["failed", "closed", "disconnected"]:
            # logger.info(f"session: {self._id} - ICE state → {state}")
            await self._on_disconnect(self._id)


    async def _on_track(self, track):
        logger.info(f"session: {self._id} - Track received: {track.kind}")
        if track.kind == "video":
            transformed = VideoTransformTrack(track, self._grpc_processor, self._collector)
            self._peer_connection.addTrack(transformed)


    async def start(self, sdp_data: SDPData) -> SDPData:
        logger.info(f"session: {self._id} - starting...")
        await self._grpc_processor.start()

        offer = RTCSessionDescription(sdp_data.sdp, sdp_data.type)
        await self._peer_connection.setRemoteDescription(offer)

        #TODO ограничить скорость сбора кандидатов
        await asyncio.sleep(1)
        answer = await self._peer_connection.createAnswer()
        await self._peer_connection.setLocalDescription(answer)

        self._started_at = datetime.now(timezone.utc)
        logger.info(f"session: {self._id} - started")

        return SDPData(sdp=self._peer_connection.localDescription.sdp,
                       type=self._peer_connection.localDescription.type)


    async def finalize(self) -> Tuple[Optional[str], Optional[str], Optional[VideoMetaData]]:
        self._finished_at = datetime.now(timezone.utc)

        video_file_path, video_file_name, video_meta = None, None, None

        try:
            logger.info(f"session: {self._id} - finalizing session resources...")
            video_file_path, video_file_name, video_meta = await self._collector.finalize()
        except Exception as e:
            logger.error(f"session: {self._id} - finalize error: {e}")


        return video_file_path, video_file_name, video_meta


    async def shutdown(self):
        logger.info(f"session: {self._id} - cleaning up")

        if self._peer_connection:
            await self._peer_connection.close()
            logger.info(f"peer_connection: {self._id} - is closed")

        if self._grpc_processor:
            await self._grpc_processor.stop()
            logger.info(f"session: {self._id} - processor stopped")

        if self._collector:
            await self._collector.cleanup()
            logger.info(f"session: {self._id} - collector cleaned up")


        logger.info(f"session: {self._id} - shutdown complete")