import time

import av
from aiortc import VideoStreamTrack
from grpc_client.base_processor import BaseProcessor
from utils.frame_collector import FrameCollector
from core.logger import get_logger


logger = get_logger(__name__)


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track: VideoStreamTrack, processor: BaseProcessor, collector:FrameCollector=None):
        super().__init__()
        self.track = track
        self.processor = processor
        self.collector = collector



    async def recv(self) -> av.VideoFrame:
        frame = await self.track.recv()
        first_ts = time.time_ns()
        processed, first_ts = await self.processor.process_frame(frame, first_ts)
        second_ts = time.time_ns()
        latency = (second_ts - first_ts) / 1_000_000
        logger.info(f"session: {self.processor.session_id} - grpc video processor latency= { latency } ms")
        if self.collector:
            try:
                await self.collector.add_frame(processed)
            except Exception as e:
                logger.error(f"session: {self.collector._session_id} - error adding frame to collector: {e}")
        return frame

