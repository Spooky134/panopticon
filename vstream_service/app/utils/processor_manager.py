from platform import processor

from aiortc import RTCPeerConnection, RTCConfiguration
from typing import Dict

from utils.grpc_video_processor import GrpcVideoProcessor


class ProcessorManager:
    def __init__(self, max_connections: int=1000):
        self.max_connections = max_connections
        self.processors: Dict[str, GrpcVideoProcessor] = {}


    async def create_processor(self, session_id: str) -> GrpcVideoProcessor:
        if len(self.processors) >= self.max_connections:
            raise Exception("Server busy")#TODO придумать ошибку

        processor =  GrpcVideoProcessor(session_id)
        await processor.start()
        self.processors[session_id] = processor

        return processor

    async def get_processor(self, session_id: str) -> GrpcVideoProcessor:
        return self.processors.get(session_id)

    async def close_processor(self, session_id: str):
        processor = self.processors.pop(session_id, None)
        if processor:
            await processor.stop()
