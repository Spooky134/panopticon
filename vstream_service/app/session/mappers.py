from typing import List

from app.video.entities import StreamingVideoEntity
from app.core.mappers import BaseMapper
from app.session.entities import StreamingSessionEntity
from app.session.models import SessionModel


class StreamingSessionMapper(BaseMapper):
    @classmethod
    def to_entity(cls, model: SessionModel) -> StreamingSessionEntity:
        data = cls.model_to_dict(model)

        data["videos"] = (
            [StreamingVideoEntity(id=m.id, name=m.name) for m in model.videos]
            if cls.is_loaded(model, "videos")
            else []
        )

        return StreamingSessionEntity(**data)
    
    
    @classmethod
    def to_entities(cls, models: List[SessionModel]) -> List[StreamingSessionEntity]:
        return [cls.to_entity(model) for model in models]

    # @classmethod
    # def to_entity(cls, model: StreamingSessionModel) -> StreamingSessionEntity:
    #     data = cls.model_to_dict(model)
    #     return StreamingSessionEntity(**data)

