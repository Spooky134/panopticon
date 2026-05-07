python3 -m grpc_tools.protoc -I. --python_out=./ml_service/app --grpc_python_out=./ml_service/app ./proto/ml_worker.proto
python3 -m grpc_tools.protoc -I./proto --python_out=./streaming/app/generated --grpc_python_out=./streaming/app/generated ml_worker.proto
