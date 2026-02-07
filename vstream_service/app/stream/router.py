from uuid import UUID
from fastapi import APIRouter, Depends

from app.stream.dependencies import StreamingServiceDep
from app.stream.schemas import SDP
from app.stream.entities import SDPEntity
from app.core.security.api_key import get_api_key


router = APIRouter(prefix="/stream", tags=["stream"], dependencies=[Depends(get_api_key)])

@router.post("/{streaming_session_id}/offer", response_model=SDP)
async def offer(streaming_session_id: UUID,
                sdp_data: SDP,
                streaming_runtime_service: StreamingServiceDep):
    sdp_entity = SDPEntity(**sdp_data.model_dump())
    sdp_entity_answer = await streaming_runtime_service.offer(
        streaming_session_id=streaming_session_id,
        sdp_data=sdp_entity
    )

    return sdp_entity_answer



@router.post("/{streaming_session_id}/stop")
async def stop(streaming_session_id: UUID,
               streaming_runtime_service: StreamingServiceDep):
    data = await streaming_runtime_service.stop(
        streaming_session_id=streaming_session_id
    )

    return data