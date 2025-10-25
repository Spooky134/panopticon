from typing import Dict

from grpc_client.video_processor import VideoProcessor


class GrpcProcessorManager:
    def __init__(self, max_connections: int=1000):
        self.max_connections = max_connections
        self.processors: Dict[str, VideoProcessor] = {}


    async def create_processor(self, session_id: str) -> VideoProcessor:
        if len(self.processors) >= self.max_connections:
            raise Exception("Server busy")#TODO придумать ошибку

        processor =  VideoProcessor(session_id)
        await processor.start()
        self.processors[session_id] = processor

        return processor

    async def get_processor(self, session_id: str) -> VideoProcessor:
        return self.processors.get(session_id)

    async def close_processor(self, session_id: str):
        processor = self.processors.pop(session_id, None)
        if processor:
            await processor.stop()
