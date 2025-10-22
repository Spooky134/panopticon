import av


class BaseProcessor:
    async def process_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        raise NotImplementedError
