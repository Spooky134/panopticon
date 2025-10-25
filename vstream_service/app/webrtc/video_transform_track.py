import av
from aiortc import VideoStreamTrack
from grpc_client.base_processor import BaseProcessor
from utils.frame_collector import FrameCollector

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
                print(f"[VideoTransformTrack] Ошибка при добавлении кадра в collector: {e}")
        return processed
