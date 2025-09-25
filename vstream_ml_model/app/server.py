import cv2
import numpy as np
import grpc
from concurrent import futures
from generated import mlworker_pb2_grpc, mlworker_pb2


class MLServiceServicer(mlworker_pb2_grpc.MLServiceServicer):
    async def StreamFrames(self, request_iterator, context):
        async for request in request_iterator:
            # Декодируем JPEG → numpy
            nparr = np.frombuffer(request.image, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Обработка: ч/б фильтр
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Кодируем обратно в JPEG
            _, jpeg_bytes = cv2.imencode(".jpg", processed)

            # Формируем ответ
            yield mlworker_pb2.FrameResponse(
                processed_image=jpeg_bytes.tobytes(),
                comment="Applied grayscale filter"
            )

async def serve():
    server = grpc.aio.server()
    mlworker_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    print("ML Worker gRPC server (streaming) running on port 50051")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    import asyncio
    asyncio.run(serve())
