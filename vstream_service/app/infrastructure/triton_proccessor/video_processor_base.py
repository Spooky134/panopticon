import av
from numpy import ndarray
import asyncio
import tritonclient.grpc.aio as triton_grpc
from infrastructure.triton_proccessor.video_processor_type import ProcessorType


class VideoProcessorBase:
    def __init__(
        self,
        triton_client: triton_grpc.InferenceServerClient,
        processor_type: ProcessorType,
    ):
        self._processor_type = processor_type
        self._input_w, self._input_h=None, None

        self._client = triton_client
        self._queue = asyncio.Queue(maxsize=1)

        self._latest_processed_frame: ndarray | None = None

        self._infer_task: asyncio.Task | None = None

    async def start(self):
        self._infer_task = asyncio.create_task(self._infer_loop())

    async def stop(self):
        if self._infer_task:
            self._infer_task.cancel()

    async def process_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        raise NotImplementedError()

    async def _infer_loop(self):
        raise NotImplementedError()
