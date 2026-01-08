from uuid import UUID
from fastapi import APIRouter, Depends

from api.schemas.sdp import SDP
from core.entities.sdp_data import SDPData
from dataclasses import asdict
from core.security.api_key import get_api_key
from api.dependencies import StreamingRuntimeServiceDep


router = APIRouter(prefix="/stream", tags=["stream"])

@router.post("/{streaming_session_id}/offer", response_model=SDP, dependencies=[Depends(get_api_key)])
async def offer(streaming_session_id: UUID,
                sdp_data: SDP,
                streaming_runtime_service: StreamingRuntimeServiceDep):
    sdp_data = SDPData(**sdp_data.model_dump())
    sdp_data_answer = await streaming_runtime_service.offer(streaming_session_id=streaming_session_id,
                                                            sdp_data=sdp_data)

    return SDP(**asdict(sdp_data_answer))


@router.post("/{streaming_session_id}/stop", dependencies=[Depends(get_api_key)])
async def stop(streaming_session_id: UUID,
               streaming_runtime_service: StreamingRuntimeServiceDep):
    data = await streaming_runtime_service.stop(streaming_session_id=streaming_session_id)

    return data