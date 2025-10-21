from aiortc import VideoStreamTrack
from utils.grpc_video_processor import GrpcVideoProcessor



class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, grpc_processor: GrpcVideoProcessor):
        super().__init__()
        self.track = track
        self.grpc_processor = grpc_processor

    async def recv(self):
        frame = await self.track.recv()
        # Используем gRPC для обработки вместо локальной обработки
        return await self.grpc_processor.process_frame(frame)