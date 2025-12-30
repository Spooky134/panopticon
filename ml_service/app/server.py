import cv2
import numpy as np
import grpc
import ml_worker_pb2_grpc, ml_worker_pb2
import asyncio


class MLServiceServicer(ml_worker_pb2_grpc.MLServiceServicer):
    def __init__(self):
        pass
    
    async def StreamFrames(self, request_iterator, context):
        async for request in request_iterator:
            try:
                w, h, c = request.width, request.height, request.channels
                frame = np.frombuffer(request.frame_data, dtype=np.uint8)
                frame = frame.reshape((h, w, c)).copy()

                # faces = detect_faces_dlib_cnn(frame, upsample=1)
                #
                # # Рисуем рамки
                # for (x, y, w, h) in faces:
                #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                processed_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # processed_frame = frame

                # processed_frame = detect_faces_simple(frame)

                session_id = request.session_id

                yield ml_worker_pb2.FrameResponse(
                    session_id=session_id,
                    processed_frame_data=processed_frame.tobytes(),
                    width=w,
                    height=h,
                    channels=c,
                    comment=f"session: {session_id} - frame is processed",
                    ts=request.ts,
                )
            except Exception as e:
                print(f"error processing frame: {e}")



async def serve():
    # server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    server = grpc.aio.server()
    ml_worker_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    await server.start()
    print("ML Service running on port 50051", flush=True)
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())