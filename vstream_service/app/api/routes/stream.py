from uuid import UUID
from api.services.stream import StreamService
from api.services.stream_factory import get_stream_service
from fastapi import APIRouter, Depends, status
from api.schemas.sdp import SDPData
from core.security.api_key import get_api_key

router = APIRouter(prefix="/stream", tags=["stream"])


@router.post("/{streaming_session_id}/offer", response_model=SDPData, dependencies=[Depends(get_api_key)])
async def offer(streaming_session_id: UUID,
                sdp_data: SDPData,
                stream_service: StreamService = Depends(get_stream_service)):
    data = await stream_service.offer(streaming_session_id=streaming_session_id,
                                      sdp_data=sdp_data)

    return data


@router.post("/{streaming_session_id}/stop", dependencies=[Depends(get_api_key)])
async def stop(streaming_session_id: UUID,
               stream_service: StreamService = Depends(get_stream_service)):
    data = await stream_service.stop(streaming_session_id=streaming_session_id)

    return data