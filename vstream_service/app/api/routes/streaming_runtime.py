from uuid import UUID
from fastapi import APIRouter, Depends

from api.dependencies import StreamingRuntimeServiceDep
from api.schemas.sdp import SDP
from core.entities.sdp_data import SDPEntity
from core.security.api_key import get_api_key


router = APIRouter(prefix="/stream", tags=["stream"])

@router.post("/{streaming_session_id}/offer", response_model=SDP, dependencies=[Depends(get_api_key)])
async def offer(streaming_session_id: UUID,
                sdp_data: SDP,
                streaming_runtime_service: StreamingRuntimeServiceDep):
    sdp_entity = SDPEntity(**sdp_data.model_dump())
    sdp_entity_answer = await streaming_runtime_service.offer(
        streaming_session_id=streaming_session_id,
        sdp_data=sdp_entity
    )

    return SDP.model_validate(sdp_entity_answer)


@router.post("/{streaming_session_id}/stop", dependencies=[Depends(get_api_key)])
async def stop(streaming_session_id: UUID,
               streaming_runtime_service: StreamingRuntimeServiceDep):
    data = await streaming_runtime_service.stop(streaming_session_id=streaming_session_id)

    return data