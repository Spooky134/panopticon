from grpc_client.processor_manager import ProcessorManager


def get_processor_manager():
    return ProcessorManager(max_connections=1000)
