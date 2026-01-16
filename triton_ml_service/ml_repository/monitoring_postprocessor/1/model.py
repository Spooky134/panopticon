# model.py для monitoring_postprocessor
import triton_python_backend_utils as pb_utils
import numpy as np
import json


class TritonPythonModel:
    def initialize(self, args):
        """Инициализация модели"""
        print("Monitoring Postprocessing Model Initialized")
        
    def execute(self, requests):
        """Обработка запросов - просто пропускаем данные через себя"""
        responses = []
        
        for request in requests:
            try:
                # Получаем все входные тензоры
                session_id_tensor = pb_utils.get_input_tensor_by_name(request, "session_id")
                comment_tensor = pb_utils.get_input_tensor_by_name(request, "comment")
                ts_tensor = pb_utils.get_input_tensor_by_name(request, "ts")
                boxes_tensor = pb_utils.get_input_tensor_by_name(request, "boxes")
                
                # Проверяем, что все тензоры присутствуют
                if None in [session_id_tensor, comment_tensor, ts_tensor, boxes_tensor]:
                    error_response = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError("Missing required input tensors")
                    )
                    responses.append(error_response)
                    continue
                
                # Просто переносим данные на выход без изменений
                out_session_id = pb_utils.Tensor(
                    "session_id", 
                    session_id_tensor.as_numpy()
                )
                out_comment = pb_utils.Tensor(
                    "comment", 
                    comment_tensor.as_numpy()
                )
                out_ts = pb_utils.Tensor(
                    "ts", 
                    ts_tensor.as_numpy()
                )
                out_boxes = pb_utils.Tensor(
                    "boxes", 
                    boxes_tensor.as_numpy()
                )
                
                # Собираем ответ
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_session_id, out_comment, out_ts, out_boxes]
                )
                responses.append(inference_response)
                
            except Exception as e:
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"Error in postprocessor: {str(e)}")
                )
                responses.append(error_response)
        
        return responses
    
    def finalize(self):
        """Выгрузка модели"""
        print("Monitoring Postprocessor Model Unloaded")