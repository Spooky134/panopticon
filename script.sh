python3 -m grpc_tools.protoc -I. --python_out=./ml_service/app --grpc_python_out=./ml_service/app ./proto/ml_worker.proto
python3 -m grpc_tools.protoc -I. --python_out=./vstream_service/app --grpc_python_out=./vstream_service/app ./proto/ml_worker.proto
