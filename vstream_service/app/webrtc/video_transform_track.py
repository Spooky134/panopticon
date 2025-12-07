import time

import av
from aiortc import VideoStreamTrack
from grpc_client.base_processor import BaseProcessor
from utils.frame_collector import FrameCollector
from core.logger import get_logger


logger = get_logger(__name__)


async def on_frame(frame):
    img = frame.to_ndarray(format="rgba")
    row = img[0]
    ts = 0

    for i in range(8):
        ts |= int(row[i][3]) << (i * 8)  # читаем АЛЬФУ

    now = int(time.time() * 1000)
    latency = now - ts
    logger.info(f"FRAME LATENCY = {ts} ms")


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track: VideoStreamTrack, processor: BaseProcessor, collector:FrameCollector=None):
        super().__init__()
        self.track = track
        self.processor = processor
        self.collector = collector

    async def recv(self) -> av.VideoFrame:
        frame = await self.track.recv()
        # await on_frame(frame)
        # first_ts = time.time_ns()
        # logger.info(f"RECV: frame ts = {first_ts}")
        # processed, ts = await self.processor.process_frame(frame, first_ts)
        processed = frame
        # second_ts = time.time_ns()
        # logger.info(f"session: {self.processor.session_id} - grpc video processor latency= { (second_ts - ts) / 1_000_000 } ms")
        if self.collector:
            try:
                await self.collector.add_frame(processed)
            except Exception as e:
                logger.error(f"session: {self.collector._session_id} - error adding frame to collector: {e}")
        return processed

