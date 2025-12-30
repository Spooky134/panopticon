from uuid import UUID
from services.streaming_runtime_service import StreamingRuntimeService
from services.streaming_runtime_service_factory import get_streaming_runtime_service
from fastapi import APIRouter, Depends
from api.schemas.sdp import SDPData
from core.security.api_key import get_api_key

router = APIRouter(prefix="/stream", tags=["stream"])


@router.post("/{streaming_session_id}/offer", response_model=SDPData, dependencies=[Depends(get_api_key)])
async def offer(streaming_session_id: UUID,
                sdp_data: SDPData,
                streaming_runtime_service: StreamingRuntimeService = Depends(get_streaming_runtime_service)):
    data = await streaming_runtime_service.offer(streaming_session_id=streaming_session_id,
                                      sdp_data=sdp_data)

    return data


@router.post("/{streaming_session_id}/stop", dependencies=[Depends(get_api_key)])
async def stop(streaming_session_id: UUID,
               streaming_runtime_service: StreamingRuntimeService = Depends(get_streaming_runtime_service)):
    data = await streaming_runtime_service.stop(streaming_session_id=streaming_session_id)

    return data