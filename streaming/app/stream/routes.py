from uuid import UUID
from fastapi import APIRouter, Depends

from app.stream.dependencies import SignalingOfferUseCase, SignalingStopUseCase
from app.stream.schemas import SDP
from app.stream.entities import SDPEntity
from app.core.security.api_key import get_api_key


router = APIRouter(prefix="/stream", tags=["stream"], dependencies=[Depends(get_api_key)])

@router.post("/{session_id}/offer", response_model=SDP)
async def offer(
    session_id: UUID,
    sdp: SDP,
    use_case: SignalingOfferUseCase
):
    sdp_entity = SDPEntity(**sdp.model_dump())
    sdp_entity_answer = await use_case.execute(session_id, sdp_entity)

    return sdp_entity_answer


@router.post("/{session_id}/stop")
async def stop(
    session_id: UUID,
    use_case: SignalingStopUseCase
):
    data = await use_case.execute(session_id)

    return data