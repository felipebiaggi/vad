from common import (
    BsonModel,
)


class StartOfSpeech(BsonModel):
    start_byte: int


class EndOfSpeech(BsonModel):
    end_byte: int


class EndOfStream(BsonModel):
    end_of_stream: bool


class NoSpeech(BsonModel):
    byte_count: int


class Speech(BsonModel):
    end_of_stream: bool
    speech: bytes


class Error(BsonModel):
    error_msg: str
