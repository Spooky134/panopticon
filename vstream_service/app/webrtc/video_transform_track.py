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
        processed = await self.processor.process_frame(frame)
        if self.collector:
            try:
                await self.collector.add_frame(processed)
            except Exception as e:
                logger.error(f"processor: {self.processor.session_id} - Error adding frame to collector: {e}")
        return processed
