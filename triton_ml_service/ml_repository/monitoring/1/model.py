import triton_python_backend_utils as pb_utils
import numpy as np
import json
import sys
from utils.media_p import RecognitionModel
import os
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append('/app/generated')


import ml_worker_pb2

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.session_states = {}
        print("Proctoring Model Initialized")

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, 'raw_input')
                session_id, frame, width, height, channels, ts = self.get_input_data(
                    tensor=input_tensor
                )

                if session_id not in self.session_states:
                    self.session_states[session_id] = {
                        'frame_count': 0,
                        'calibration_done': False,
                        "recognition_model": RecognitionModel()
                    }

                state = self.session_states[session_id]
                state['frame_count'] += 1

                if not state['calibration_done'] and state['frame_count'] <= 900:
                    annotated_frame = state["recognition_model"].calibration_session(frame)
                    cv2.putText(
                        frame,
                        "CALIBRATION...",
                        (frame.shape[1] - 300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3
                    )

                    if state['frame_count'] == 900:
                        state["recognition_model"].calibration_profile()
                        state['calibration_done'] = True

                else:
                    annotated_frame = state["recognition_model"].monitoring_session(frame,
                                                                                tolerance=0.05,
                                                                                angle_tolerance=20)
                    cv2.putText(
                        frame,
                        "MONITORING...",
                        (frame.shape[1] - 300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        3)



                # annotated_frame = self.recognition_model.process_frame(frame=frame)




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