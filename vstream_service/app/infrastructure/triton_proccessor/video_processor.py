import av
import cv2
import numpy as np
import asyncio
import time
from uuid import UUID

import tritonclient.grpc.aio as triton_grpc
from tritonclient.grpc import InferInput, InferRequestedOutput

from core.logger import get_logger

logger = get_logger(__name__)


class VideoProcessor:
    def __init__(
        self,
        triton_client: triton_grpc.InferenceServerClient,
        session_id: UUID,
        model_name: str = "proctoring_ml_model",
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
            for x, y, w, h in self._latest_boxes:
                x = int(x * orig_w / self._input_w)
                y = int(y * orig_h / self._input_h)
                w = int(w * orig_w / self._input_w)
                h = int(h * orig_h / self._input_h)

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

                inputs = []

                inp = InferInput("session_id", [1, 1], "BYTES")
                inp.set_data_from_numpy(
                    np.array([[self._session_id.encode("utf-8")]], dtype=object)
                )
                inputs.append(inp)

                inp = InferInput("frame_data", [1, img.size], "UINT8")
                inp.set_data_from_numpy(img.flatten().reshape(1, -1))
                inputs.append(inp)

                inp = InferInput("width", [1, 1], "INT32")
                inp.set_data_from_numpy(
                    np.array([[self._input_w]], dtype=np.int32)
                )
                inputs.append(inp)

                inp = InferInput("height", [1, 1], "INT32")
                inp.set_data_from_numpy(
                    np.array([[self._input_h]], dtype=np.int32)
                )
                inputs.append(inp)

                inp = InferInput("channels", [1, 1], "INT32")
                inp.set_data_from_numpy(
                    np.array([[3]], dtype=np.int32)
                )
                inputs.append(inp)

                inp = InferInput("ts", [1, 1], "INT64")
                inp.set_data_from_numpy(
                    np.array([[ts]], dtype=np.int64)
                )
                inputs.append(inp)

                outputs = [
                    InferRequestedOutput("boxes"),
                ]

                result = await self._client.infer(
                    model_name=self._model_name,
                    inputs=inputs,
                    outputs=outputs,
                )
                res_boxes = result.as_numpy("boxes")
                if res_boxes is not None:
                    self._latest_boxes = res_boxes[0]

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                f"session:{self._session_id} - Triton infer error: {e}"
            )
