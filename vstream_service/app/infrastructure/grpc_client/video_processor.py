import av
import cv2
import numpy as np
import asyncio
import grpc
import time
from uuid import UUID

import ml_worker_pb2
import ml_worker_pb2_grpc
from core.logger import get_logger


logger = get_logger(__name__)

#TODO если возвращать UUID в ответе то нужно привести к UUID
class VideoProcessor:
    def __init__(self, service_url: str, session_id: UUID):
        self._session_id = str(session_id)
        self._channel = grpc.aio.insecure_channel(service_url)
        self._stub = ml_worker_pb2_grpc.MLServiceStub(self._channel)

        self._request_queue = asyncio.Queue(maxsize=1)
        self._latest_processed_frame: av.VideoFrame | None = None
        self._latest_boxes: list | None = None

        self._processing_task = None
        self._receiving_task = None

    async def start(self):
        self._processing_task = asyncio.create_task(self._stream_loop())

    async def stop(self):
        if self._processing_task:
            self._processing_task.cancel()
        await self._channel.close()

    async def process_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        orig_h, orig_w, _ = img.shape
        scale_width, scale_height = 640, 480
        small_img = cv2.resize(img, (scale_width, scale_height))

        if not self._request_queue.full():
            frame_bytes = small_img.tobytes()
            req = ml_worker_pb2.FrameRequest(
                session_id=self._session_id,
                frame_data=frame_bytes,
                width=scale_width,
                height=scale_height,
                channels=3,
                ts=int(time.time() * 1000),
            )
            self._request_queue.put_nowait(req)
        else:
            # дроп кадров
            # TODO логирование
            pass

        img_copy = img.copy()
        if self._latest_boxes is not None:
            for box in self._latest_boxes:
                x = int(box.x * orig_w / scale_width)
                y = int(box.y * orig_h / scale_height)
                w = int(box.width * orig_w / scale_width)
                h = int(box.height * orig_h / scale_height)
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            new_frame = av.VideoFrame.from_ndarray(img_copy, format="bgr24")
            self._latest_processed_frame = new_frame

        if self._latest_processed_frame is not None:
            self._latest_processed_frame.pts = frame.pts
            self._latest_processed_frame.time_base = frame.time_base
            return self._latest_processed_frame

        return frame

    async def _stream_loop(self):
        async def request_generator():
            while True:
                req = await self._request_queue.get()
                yield req


        try:
            async for response in self._stub.StreamFrames(request_generator()):
                self._latest_boxes = response.boxes
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"session:{self._session_id} - error in gRPC stream: {e}")
