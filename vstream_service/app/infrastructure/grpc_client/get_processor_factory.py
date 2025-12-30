from config.settings import settings
from infrastructure.grpc_client.processor_factory import VideoProcessorFactory

def get_processor_factory() -> VideoProcessorFactory:
    return VideoProcessorFactory(service_url=settings.ML_SERVICE_URL)