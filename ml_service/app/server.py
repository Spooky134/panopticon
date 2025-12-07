import cv2
import numpy as np
import grpc
import ml_worker_pb2_grpc, ml_worker_pb2
import asyncio


class MLServiceServicer(ml_worker_pb2_grpc.MLServiceServicer):
    def __init__(self):
        pass
    
    async def StreamFrames(self, request_iterator, context):
        try:
            async for request in request_iterator:
                nparr = np.frombuffer(request.image, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print(f"session: {request.session_id} - error of decoding frame")
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                processed_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                _, jpeg_bytes = cv2.imencode(".jpg", processed_frame)

                yield ml_worker_pb2.FrameResponse(
                    processed_image=jpeg_bytes.tobytes(),
                    comment=f"session: {request.session_id} - frame is processed",
                    ts=request.ts,
                )
                
        except Exception as e:
            print(f"error in stream: {e}")

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