import sys

import av
import cv2
import numpy as np
import asyncio
import time
from uuid import UUID
import sys

import tritonclient.grpc.aio as triton_grpc
from tritonclient.grpc import InferInput, InferRequestedOutput

from core.logger import get_logger

logger = get_logger(__name__)

sys.path.append('/app/generated')

import ml_worker_pb2

class VideoProcessor:
    def __init__(
        self,
        triton_client: triton_grpc.InferenceServerClient,
        session_id: UUID,
        model_name: str = "monitoring",
        input_size=(640, 480),
    ):
        self._session_id = str(session_id)
        self._model_name = model_name
        self._input_w, self._input_h = input_size

        self._client = triton_client
        self._queue = asyncio.Queue(maxsize=1)

        # shared inference result
        self._latest_boxes: np.ndarray | None = None

        self._infer_task: asyncio.Task | None = None

    async def start(self):
        self._infer_task = asyncio.create_task(self._infer_loop())

    async def stop(self):
        if self._infer_task:
            self._infer_task.cancel()

    async def process_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        orig_h, orig_w, _ = img.shape

        resized = cv2.resize(img, (self._input_w, self._input_h))

        # enqueue (drop old if needed)
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        try:
            self._queue.put_nowait(
                (
                    resized,
                    int(time.time() * 1000),
                )
            )
        except asyncio.QueueFull:
            pass

        if self._latest_boxes is not None:
            for box in self._latest_boxes:
                x = int(box.x * orig_w / self._input_w)
                y = int(box.y * orig_h / self._input_h)
                w = int(box.width * orig_w / self._input_w)
                h = int(box.height * orig_h / self._input_h)

                cv2.rectangle(
                    img,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

        out = av.VideoFrame.from_ndarray(img, format="bgr24")
        out.pts = frame.pts
        out.time_base = frame.time_base
        return out

    async def _infer_loop(self):
        try:
            while True:
                img, ts = await self._queue.get()

                frame_request = ml_worker_pb2.FrameRequest()
                frame_request.session_id = str(self._session_id)
                frame_request.frame_data = img.tobytes()
                frame_request.width = self._input_w
                frame_request.height = self._input_h
                frame_request.channels = 3
                frame_request.ts=ts

                request_bytes = frame_request.SerializeToString()

                tensor_data = np.array(list(request_bytes), dtype=np.uint8)

                tensor = InferInput("raw_input", tensor_data.shape, "UINT8")
                tensor.set_data_from_numpy(tensor_data)


                response = await self._client.infer(
                    model_name=self._model_name,
                    inputs=[tensor]
                )
                output_data = response.as_numpy("raw_output")
                if output_data is not None:
                    frame_response = ml_worker_pb2.FrameResponse()
                    frame_response.ParseFromString(output_data.tobytes())

                    res_boxes = frame_response.boxes
                    if res_boxes is not None:
                        self._latest_boxes = res_boxes

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                f"session:{self._session_id} - Triton infer error: {e}"
            )
