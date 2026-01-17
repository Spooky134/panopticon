import triton_python_backend_utils as pb_utils
import numpy as np
import json
import sys
# from utils.open_pose_model import RecognitionModel
from utils.mm import RecognitionModel
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# PROTO_FILE = os.path.join(current_dir, "utils", "pose_deploy_linevec.prototxt")
# WIGHT_FILE = os.path.join(current_dir, "utils", "pose_iter_440000.caffemodel")
#
sys.path.append('/app/generated')


import ml_worker_pb2

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.annotator = RecognitionModel()
        print("Proctoring Model Initialized")

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, 'raw_input')
                session_id, frame, width, height, channels, ts = self.get_input_data(
                    tensor=input_tensor
                )


                annotated_frame = self.annotator.process_frame(frame=frame)


                # cv2.rectangle(frame, (50,50), (200, 200), (0, 255, 0), 2)
                #
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # processed_image = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)



                # boxes = np.array([[50, 50, 150, 150]], dtype=np.int32)
                comment = f"session: {session_id} - frame is processed"







                response_bytes = self.pack_response_to_bytes(
                    session_id=session_id,
                    frame=annotated_frame,
                    width=width,
                    height=height,
                    channels=channels,
                    ts=ts,
                    comment=comment,
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
        height = frame_request.height
        width = frame_request.width
        channels = frame_request.channels
        ts = frame_request.ts

        frame_bytes = frame_request.frame_data
        expected_size = width * height * channels
        if len(frame_bytes) != expected_size:
            raise ValueError(f"Frame size mismatch! Expected {expected_size}, got {len(frame_bytes)}. Check client resize/stride.")
        frame_data = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = frame_data.reshape((height, width, channels)).copy()

        return session_id, frame, width, height, channels, ts

    def pack_response_to_bytes(self, session_id, frame, width, height, channels, ts, comment):
        frame_response = ml_worker_pb2.FrameResponse()
        frame_response.session_id = session_id if isinstance(session_id, str) else session_id.decode()
        frame_response.frame_data = frame.tobytes()
        frame_response.width = int(width)
        frame_response.height = int(height)
        frame_response.channels = int(channels)
        frame_response.ts = int(ts)
        frame_response.comment = comment


        # for box in boxes:
        #     pb_box = frame_response.boxes.add()
        #     pb_box.x = int(box[0])
        #     pb_box.y = int(box[1])
        #     pb_box.width = int(box[2])
        #     pb_box.height = int(box[3])

        response_bytes = frame_response.SerializeToString()

        return response_bytes


    def finalize(self):
        self.annotator.release()
        print("Proctoring Model Unloaded")