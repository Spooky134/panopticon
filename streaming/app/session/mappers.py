from typing import List

from app.video.entities import StreamingVideoEntity, VideoMetaEntity
from app.core.mappers import BaseMapper
from app.session.entities import StreamingSessionEntity
from app.session.models import SessionModel


class StreamingSessionMapper(BaseMapper):
    @classmethod
    def to_entity(cls, model: SessionModel) -> StreamingSessionEntity:
        data = cls.model_to_dict(model)

        data["videos"] = (
            [StreamingVideoEntity(
                s3_key=m.s3_key,
                streaming_session_id=m.session_id,
                created_at=m.created_at,
                meta=VideoMetaEntity(
                    width=m.width,
                    height=m.height,
                    duration=m.duration,
                    codec=m.meta.get("codec"),
                    file_size=m.file_size,
                    mime_type=m.mime_type,
                    fps=m.fps,
                    bit_rate=m.meta.get("bit_rate"),
                    frame_count=m.meta.get("frame_count")
                )
            ) for m in model.videos]
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

