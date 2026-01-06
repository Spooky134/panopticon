from typing import Annotated
from fastapi import Depends

from services.streaming_runtime_service import StreamingRuntimeService
from services.streaming_runtime_service_factory import get_streaming_runtime_service
from core.security.api_key import get_api_key
from services.streaming_session_lifecycle_service import StreamingSessionLifecycleService
from services.streaming_session_lifecycle_service_factory import get_streaming_session_lifecycle_service



StreamingRuntimeServiceDep: type[StreamingRuntimeService] = Annotated[StreamingRuntimeService,
                                                                            Depends(get_streaming_runtime_service)]
StreamingLifecycleServiceDep: type[StreamingSessionLifecycleService] = Annotated[StreamingSessionLifecycleService,
                                                                                    Depends(get_streaming_session_lifecycle_service)]