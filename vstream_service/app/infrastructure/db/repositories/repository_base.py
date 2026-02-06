from abc import ABC, abstractmethod

class BaseRepository(ABC):
    @abstractmethod
    async def get_one(self):
        raise NotImplementedError

    @abstractmethod
    async def get_all(self):
        raise NotImplementedError

    @abstractmethod
    async def create(self):
        raise NotImplementedError

    @abstractmethod
    async def update(self):
        raise NotImplementedError

    @abstractmethod
    async def delete(self):
        raise NotImplementedError