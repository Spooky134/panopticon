from typing import Annotated
from fastapi import Depends, Request

from app.session.repository import SessionRepository
from app.core.unit_of_work import UnitOfWork
from app.stream.use_cases.signaling_offer import SignalingOfferUseCase
from app.stream.use_cases.signaling_stop import SignalingStopUseCase

def get_signaling_offer_use_case(request: Request) -> SignalingOfferUseCase:
    return SignalingOfferUseCase(
        streaming_manager=request.app.state.streaming_manager,
        session_repo_factory=SessionRepository,
        uow_factory=UnitOfWork
    )

def get_signaling_stop_use_case(request: Request) -> SignalingStopUseCase:
    return SignalingStopUseCase(
        streaming_manager=request.app.state.streaming_manager,
        session_repo_factory=SessionRepository,
        uow_factory=UnitOfWork
    )

SignalingOfferUseCase = Annotated[SignalingOfferUseCase, Depends(get_signaling_offer_use_case)]
SignalingStopUseCase = Annotated[SignalingStopUseCase, Depends(get_signaling_stop_use_case)]
