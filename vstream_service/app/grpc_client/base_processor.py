import av


class BaseProcessor:
    def __init__(self, session_id):
        self.session_id = session_id

    async def start(self):
        raise NotImplementedError

    async def stop(self):
        raise NotImplementedError

    async def process_frame(self, frame: av.VideoFrame, ts) -> av.VideoFrame:
        raise NotImplementedError
