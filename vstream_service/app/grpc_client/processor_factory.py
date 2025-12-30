from uuid import UUID
from grpc_client.video_processor import VideoProcessor


class VideoProcessorFactory:
    def __init__(self, service_url:str):
        self._service_url = service_url

    def create(self, streaming_session_id: UUID) -> VideoProcessor:
        return VideoProcessor(service_url=self._service_url,
                              session_id=streaming_session_id)