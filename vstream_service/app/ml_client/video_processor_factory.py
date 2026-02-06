from uuid import UUID
import tritonclient.grpc.aio as grpcclient

from app.ml_client.video_processor import VideoProcessor
from app.ml_client.video_processor_type import ProcessorType


class VideoProcessorFactory:
    def __init__(self, triton_client: grpcclient.InferenceServerClient):
        self._triton_client = triton_client

    def create(self, streaming_session_id: UUID, processor_type: ProcessorType) -> VideoProcessor:
        return VideoProcessor(triton_client=self._triton_client,
                              session_id=streaming_session_id,
                              processor_type=processor_type)