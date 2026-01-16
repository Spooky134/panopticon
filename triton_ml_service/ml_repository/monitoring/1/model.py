import triton_python_backend_utils as pb_utils
import numpy as np
import json
import sys


sys.path.append('/app/generated')

import ml_worker_pb2

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        print("Proctoring Model Initialized")

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, 'raw_input')
                session_id, frame, width, height, channels, ts = self.get_input_data(
                    tensor=input_tensor
                )






                boxes = np.array([[50, 50, 150, 150]], dtype=np.int32)
                comment = f"session: {session_id} - frame is processed"







                response_bytes = self.pack_response_to_bytes(
                    session_id=session_id,
                    comment=comment,
                    ts=ts,
                    boxes=boxes,
                )

                output_tensor = pb_utils.Tensor(
                    "raw_output",
                    np.frombuffer(response_bytes, dtype=np.uint8)
                )

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
                responses.append(inference_response)

            except Exception as e:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Error processing frame: {str(e)}")
                ))

        return responses

    def get_input_data(self, tensor):
        raw_data = tensor.as_numpy().tobytes()
        frame_request = ml_worker_pb2.FrameRequest()
        frame_request.ParseFromString(raw_data)

        session_id = frame_request.session_id
        ts = frame_request.ts
        height = frame_request.height
        width = frame_request.width
        channels = frame_request.channels

        frame_data = np.frombuffer(frame_request.frame_data, dtype=np.uint8)
        frame = frame_data.reshape((height, width, channels)).copy()

        return session_id, frame, width, height, channels, ts

    def pack_response_to_bytes(self, session_id, comment, ts, boxes):
        frame_response = ml_worker_pb2.FrameResponse()
        frame_response.session_id = session_id if isinstance(session_id, str) else session_id.decode()
        frame_response.comment = comment
        frame_response.ts = int(ts)

        for box in boxes:
            pb_box = frame_response.boxes.add()
            pb_box.x = int(box[0])
            pb_box.y = int(box[1])
            pb_box.width = int(box[2])
            pb_box.height = int(box[3])

        response_bytes = frame_response.SerializeToString()

        return response_bytes


    def finalize(self):
        print("Proctoring Model Unloaded")