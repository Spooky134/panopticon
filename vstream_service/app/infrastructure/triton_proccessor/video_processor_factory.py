from uuid import UUID
from infrastructure.triton_proccessor.video_processor import VideoProcessor
import tritonclient.grpc.aio as grpcclient


class VideoProcessorFactory:
    def __init__(self, triton_client: grpcclient.InferenceServerClient):
        self._triton_client = triton_client

    def create(self, streaming_session_id: UUID) -> VideoProcessor:
        return VideoProcessor(triton_client=self._triton_client,
                              session_id=streaming_session_id)