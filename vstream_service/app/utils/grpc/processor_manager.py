from typing import Dict

from utils.grpc.grpc_video_processor import GrpcProcessor


class ProcessorManager:
    def __init__(self, max_connections: int=1000):
        self.max_connections = max_connections
        self.processors: Dict[str, GrpcProcessor] = {}


    async def create_processor(self, session_id: str) -> GrpcProcessor:
        if len(self.processors) >= self.max_connections:
            raise Exception("Server busy")#TODO придумать ошибку

        processor =  GrpcProcessor(session_id)
        await processor.start()
        self.processors[session_id] = processor

        return processor

    async def get_processor(self, session_id: str) -> GrpcProcessor:
        return self.processors.get(session_id)

    async def close_processor(self, session_id: str):
        processor = self.processors.pop(session_id, None)
        if processor:
            await processor.stop()
