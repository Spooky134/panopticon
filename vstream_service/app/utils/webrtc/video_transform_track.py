import av
from aiortc import VideoStreamTrack
from utils.grpc.base_processor import BaseProcessor


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track: VideoStreamTrack, processor: BaseProcessor):
        super().__init__()
        self.track = track
        self.processor = processor

    async def recv(self) -> av.VideoFrame:
        frame = await self.track.recv()
        return await self.processor.process_frame(frame)