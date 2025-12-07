import time

import av
import cv2
import numpy as np
import asyncio
import grpc
from uuid import UUID
import zlib

import ml_worker_pb2
import ml_worker_pb2_grpc
from config.settings import settings
from grpc_client.base_processor import BaseProcessor
from core.logger import get_logger


logger = get_logger(__name__)

class VideoProcessor(BaseProcessor):
    def __init__(self, session_id: UUID):
        super().__init__(session_id)
        self.channel = grpc.aio.insecure_channel(settings.ML_SERVICE_URL)
        self.stub = ml_worker_pb2_grpc.MLServiceStub(self.channel)
        self.request_queue = asyncio.Queue(150)
        self.response_queue = asyncio.Queue(150)
        self.processing_task = None

    async def start(self):
        self.processing_task = asyncio.create_task(self._process_stream())

    async def stop(self):
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        await self.channel.close()

# TODO может не преобразовывать кадры при передаче в jpg??
    async def process_frame(self, frame, ts) -> tuple[av.VideoFrame, int]:
        img = frame.to_ndarray(format="bgr24")
        _, jpeg_bytes = cv2.imencode(".jpg", img)

        logger.info(f"session: {self.session_id} - grpc request queue= {self.request_queue.qsize()}")
        logger.info(f"session: {self.session_id} - grpc response queue= {self.response_queue.qsize()}")


        await self.request_queue.put({"jpeg": jpeg_bytes.tobytes(),
                                      "ts":ts})
        response = await self.response_queue.get()

        nparr = np.frombuffer(response.processed_image, np.uint8)
        processed_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        new_frame = av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame, response.ts

    async def _process_stream(self):
        try:
            async def request_generator():
                while True:
                    frame_data = await self.request_queue.get()
                    yield ml_worker_pb2.FrameRequest(
                        session_id=self.session_id,
                        image=frame_data["jpeg"],
                        ts=frame_data["ts"]
                    )

            async for response in self.stub.StreamFrames(request_generator()):
                await self.response_queue.put(response)

        except asyncio.CancelledError:
            logger.info(f"session: {self.session_id} - gRPC stream is stopped")
        except Exception as e:
            logger.error(f"session:{self.session_id} - error in gRPC stream: {e}")


