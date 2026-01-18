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
        input_size=(720, 1280),
    ):
        self._session_id = str(session_id)
        self._model_name = model_name
        self._input_w, self._input_h=None, None

        self._client = triton_client
        self._queue = asyncio.Queue(maxsize=1)

        # shared inference result
        self._latest_boxes: np.ndarray | None = None

        self._latest_processed_frame: np.ndarray | None = None

        self._infer_task: asyncio.Task | None = None

    async def start(self):
        self._infer_task = asyncio.create_task(self._infer_loop())

    async def stop(self):
        if self._infer_task:
            self._infer_task.cancel()

    async def process_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        # frame = cv2.flip(frame, 1)
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        self._input_h, self._input_w = img.shape[:2]
        if not self._queue.full():
            self._queue.put_nowait(
                (
                    img,
                    int(time.time() * 1000),
                )
            )
        else:
            pass
            # дроп

        if self._latest_processed_frame is not None:
            self._latest_processed_frame.pts = frame.pts
            self._latest_processed_frame.time_base = frame.time_base
            return self._latest_processed_frame

        return frame


    async def _infer_loop(self):
        try:
            while True:
                img, ts = await self._queue.get()

                if img.shape[1] != self._input_w or img.shape[0] != self._input_h:
                    img = cv2.resize(img, (self._input_w, self._input_h))

                img = np.ascontiguousarray(img)

                frame_request = ml_worker_pb2.FrameRequest()
                frame_request.session_id = str(self._session_id)
                frame_request.frame_data = img.tobytes()
                frame_request.width = self._input_w
                frame_request.height = self._input_h
                frame_request.channels = 3
                frame_request.ts=ts

                request_bytes = frame_request.SerializeToString()

                # tensor_data = np.array(list(request_bytes), dtype=np.uint8)
                tensor_data = np.frombuffer(request_bytes, dtype=np.uint8)
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

                    height = frame_response.height
                    width = frame_response.width
                    channels = frame_response.channels

                    # session_id = frame_response.session_id

                    frame_data_bytes = frame_response.frame_data
                    if frame_data_bytes is not None:
                        frame_arr = np.frombuffer(frame_data_bytes, dtype=np.uint8)
                        processed_image = frame_arr.reshape((height, width, channels))

                        new_frame = av.VideoFrame.from_ndarray(processed_image, format="bgr24")
                        self._latest_processed_frame = new_frame
                        # logger.info(f"frame for session: {session_id}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                f"session:{self._session_id} - Triton infer error: {e}"
            )
