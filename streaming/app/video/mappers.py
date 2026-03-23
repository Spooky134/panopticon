# TODO сначала добавление видео в базу а загрузка в фоне (первичный ключ это s3_key = uuid) тогда бд в таске не нужна

from app.video.entities import StreamingVideoEntity, VideoMetaEntity
from app.core.mappers import BaseMapper
from app.video.models import VideoModel


class VideoMapper(BaseMapper):
    @classmethod
    def to_entity(cls, model: VideoModel) -> StreamingVideoEntity:
        return StreamingVideoEntity(
            s3_key=model.s3_key,
            session_id=model.session_id,
            created_at=model.created_at,
            meta=VideoMetaEntity(
                width=model.width,
                height=model.height,
                duration=model.duration,
                codec=model.meta.get("codec"),
                file_size=model.file_size,
                mime_type=model.mime_type,
                fps=model.fps,
                bit_rate=model.meta.get("bit_rate"),
                frame_count=model.meta.get("frame_count")
            )
        )