from typing import Any

from sqlalchemy import inspect


class BaseMapper:
    @staticmethod
    def model_to_dict(model: Any) -> dict:
        if model is None:
            return {}

        return {c.key: getattr(model, c.key) for c in model.__table__.columns}

    @staticmethod
    def is_loaded(model: Any, field_name: str) -> bool:
        if model is None:
            return False

        inspected = inspect(model)
        return field_name not in inspected.unloaded
