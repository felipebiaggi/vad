from common import (
    BsonModel,
)


class Config(BsonModel):
    samplerate: int = 8000
    start_window: int = 200
    end_window: int = 100
    head_margin: int = 0
    tail_margin: int = 0
    wait_end: int = 1000
    start_threshold: float = 1
    end_threshold: float = 0.08
    wlen: int = 10


class SendAudio(BsonModel):
    end_of_stream: bool
    audio: bytes
