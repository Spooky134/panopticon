import asyncio
import inspect
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union
from core.logger import get_logger
from infrastructure.db.models import StreamingSession

logger = get_logger(__name__)

class EventManager:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[..., Any]]] = {}

    def subscribe(self, event_type: str, handler: Callable[..., Any]) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.info(f"Subscribed to {handler.__name__} to event: {event_type}")

    async def emit(self, event_type: str, **kwargs) -> None:
        if event_type not in self._subscribers:
            return

        handlers = self._subscribers[event_type]
        tasks = []

        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(**kwargs))
            else:
                loop = asyncio.get_running_loop()
                tasks.append(loop.run_in_executor(None, lambda h=handler: h(**kwargs)))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    logger.error(f"Error in event handler for {event_type}: {res}")