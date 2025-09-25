import cv2
import numpy as np
import grpc
import mlworker_pb2
import mlworker_pb2_grpc
import asyncio


async def stream_frames():
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = mlworker_pb2_grpc.MLServiceStub(channel)

        async def frame_generator():
            cap = cv2.VideoCapture(0)  # камера
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _, jpeg_bytes = cv2.imencode(".jpg", frame)
                yield mlworker_pb2.FrameRequest(
                    session_id="test123",
                    image=jpeg_bytes.tobytes()
                )

        async for response in stub.StreamFrames(frame_generator()):
            # Декодируем обратно
            nparr = np.frombuffer(response.processed_image, np.uint8)
            processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("Processed", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

if __name__ == "__main__":
    asyncio.run(stream_frames())
