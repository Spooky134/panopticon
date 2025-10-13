from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
import av
import cv2
import numpy as np
import asyncio
import uuid
import grpc
import ml_worker_pb2
import ml_worker_pb2_grpc
from fastapi import BackgroundTasks
from config import settings


class GrpcVideoProcessor:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.channel = grpc.aio.insecure_channel(f'{settings.ML_SERVICE_HOST}:{settings.ML_SERVICE_PORT}')
        self.stub = ml_worker_pb2_grpc.MLServiceStub(self.channel)
        self.request_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.processing_task = None
        
    async def start(self):
        """Запускает обработку кадров через gRPC"""
        self.processing_task = asyncio.create_task(self._process_stream())
        
    async def stop(self):
        """Останавливает обработку"""
        if self.processing_task:
            self.processing_task.cancel()
        await self.channel.close()
    
    async def process_frame(self, frame) -> av.VideoFrame:
        """Добавляет кадр в очередь на обработку и возвращает результат"""
        # Конвертируем кадр в JPEG
        img = frame.to_ndarray(format="bgr24")
        _, jpeg_bytes = cv2.imencode(".jpg", img)
        
        # Отправляем в gRPC сервер
        await self.request_queue.put(jpeg_bytes.tobytes())
        
        # Ждем результат
        processed_data = await self.response_queue.get()
        
        # Создаем новый кадр из результата
        nparr = np.frombuffer(processed_data, np.uint8)
        processed_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        new_frame = av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
    
    async def _process_stream(self):
        """Фоновая задача для потоковой передачи кадров"""
        try:
            async def request_generator():
                while True:
                    frame_data = await self.request_queue.get()
                    yield ml_worker_pb2.FrameRequest(
                        session_id=self.session_id,
                        image=frame_data
                    )
            
            async for response in self.stub.StreamFrames(request_generator()):
                await self.response_queue.put(response.processed_image)


        except asyncio.CancelledError:
            print(f"gRPC поток остановлен для {self.session_id}")
        except Exception as e:
            print(f"Ошибка в gRPC потоке для сессии {self.session_id}: {e}")




class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, grpc_processor: GrpcVideoProcessor):
        super().__init__()
        self.track = track
        self.grpc_processor = grpc_processor
    
    async def recv(self):
        frame = await self.track.recv()
        # Используем gRPC для обработки вместо локальной обработки
        return await self.grpc_processor.process_frame(frame)