import triton_python_backend_utils as pb_utils
import numpy as np
import json


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        print("Multimodal Model Initialized")

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                in_frame_data = pb_utils.get_input_tensor_by_name(request, "frame_data").as_numpy()[0]
                in_width = pb_utils.get_input_tensor_by_name(request, "width").as_numpy()[0]
                in_height = pb_utils.get_input_tensor_by_name(request, "height").as_numpy()[0]
                in_channels = pb_utils.get_input_tensor_by_name(request, "channels").as_numpy()[0]

                session_id_raw = pb_utils.get_input_tensor_by_name(request, "session_id").as_numpy()[0]

                session_id = session_id_raw.decode('utf-8') if isinstance(session_id_raw, bytes) else session_id_raw

                ts = pb_utils.get_input_tensor_by_name(request, "ts").as_numpy()[0]

                frame = in_frame_data.reshape((in_height, in_width, in_channels)).copy()

                # Имитация работы:
                boxes = np.array([[50, 50, 150, 150]], dtype=np.int32)
                comment = f"session: {session_id} - frame is processed"

                # Выходные тензоры
                out_session_id = pb_utils.Tensor("session_id", np.array([session_id], dtype=object))

                out_ts = pb_utils.Tensor("ts", np.array([ts], dtype=np.int64))
                out_comment = pb_utils.Tensor("comment", np.array([comment], dtype=object))
                
                out_boxes = pb_utils.Tensor("boxes", boxes)

                # Ответ для конкретного запроса
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_session_id, out_comment, out_ts, out_boxes]
                )
                responses.append(inference_response)

            except Exception as e:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Error processing frame: {str(e)}")
                ))

        return responses

    def finalize(self):
        """Выгрузка модели"""
        print("Multimodal Model Unloaded")