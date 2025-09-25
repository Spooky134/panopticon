python3 -m grpc_tools.protoc -I. --python_out=../vstream_ml_model/app/generated --grpc_python_out=../vstream_ml_model/app/generated ml_worker.proto
python3 -m grpc_tools.protoc -I. --python_out=../vstream_gateway/app/generated --grpc_python_out=../vstream_gateway/app/generated ml_worker.proto
