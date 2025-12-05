import cv2
import numpy as np
import grpc
from concurrent import futures
import ml_worker_pb2_grpc, ml_worker_pb2
import asyncio
from open_pose import OpenPoseProcessor

class MLServiceServicer(ml_worker_pb2_grpc.MLServiceServicer):
    def __init__(self):
        # Можно добавить словарь для хранения состояния сессий если нужно
        self.sessions = {}
        # self.pose_processor = OpenPoseProcessor("graph_opt.pb")
    
    async def StreamFrames(self, request_iterator, context):
        try:
            async for request in request_iterator:
                # Логируем получение кадра для сессии
                print(f"Обрабатываем кадр для сессии: {request.session_id}")
                print(f'time stamp: {request.ts}')
                
                # Декодируем JPEG → numpy
                nparr = np.frombuffer(request.image, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print(f"Ошибка декодирования для сессии {request.session_id}")
                    continue
                
                # Ваша обработка кадра (пример - серый фильтр)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                processed_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # processed_frame = self.pose_processor.process_frame(frame)


                # Кодируем обратно в JPEG
                _, jpeg_bytes = cv2.imencode(".jpg", processed_frame)
                
                # Возвращаем результат
                yield ml_worker_pb2.FrameResponse(
                    processed_image=jpeg_bytes.tobytes(),
                    comment=f"Обработан кадр для сессии {request.session_id}",
                    ts=request.ts,
                )
                
        except Exception as e:
            print(f"Ошибка в потоке: {e}")

async def serve():
    # server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    server = grpc.aio.server()
    ml_worker_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    await server.start()
    print("ML Worker gRPC server running on port 50051", flush=True)
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())