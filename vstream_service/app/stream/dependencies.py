from typing import Annotated
from fastapi import Depends

from app.stream.service import StreamingService
from app.stream.service_factory import create_streaming_service


StreamingServiceDep: type[StreamingService] = Annotated[
    StreamingService,
    Depends(create_streaming_service)
]

