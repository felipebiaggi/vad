from endpointer import (
    Endpointer,
    WebRTCVad,
)
import methods as m
import events as e
import numpy as np
from pydantic import (
    ValidationError,
)
import websockets
import logging

logger = logging.getLogger(__name__)

async def vad(
    websocket,
    path,
):
    logger.warning("Session Started")
    """
    VAD endpoint to be served as a WS endpoint.

    Example of usage:
    > websockets.serve(vad, "localhost", 8765)

    Input:  any bson-encoded method from methods.py
    Output: any bson-encoded event from events.py
    """
    msg = await websocket.recv()

    # Check whether the first message is a config
    try:
        config = m.Config.from_bson(msg).dict()
        ep = Endpointer(
            WebRTCVad(3),
            **config,
        )
        logger.info(f"There is configuration in the first message, values {config}")
        msg = await websocket.recv()  # Get next message
    except ValidationError:  # Not a config message, init EP with default
        ep = Endpointer()
        logger.info(
            "There is no configuration in the first message, loading default parameters"
        )

    byte_remainder = b""
    started = False

    current_byte = 0

    try:
        while True:
            # Parse message
            method = m.SendAudio.from_bson(msg)
            end_of_stream = method.end_of_stream
            audio = method.audio

            # Byte alignment. Store remainder if the number of bytes is odd
            current_byte += len(audio)
            if byte_remainder:
                audio = b"".join(
                    byte_remainder,
                    audio,
                )
                byte_remainder = b""
            if len(audio) % 2 == 1:
                byte_remainder = audio[-1]
                audio = audio[:-1]

            # From bytes to samples
            # "<i2" means little-endian signed 2-byte int (short, 16bit)
            audio = np.frombuffer(
                audio,
                dtype="<i2",
            )
            samples = np.array(list(ep.process_samples(audio)))

            # Send speech if identified
            if len(samples) > 0:
                # Send start of speech if triggered
                if not started:
                    start_byte = ep.start * 2
                    started = True
                    await websocket.send(
                        e.StartOfSpeech(start_byte=start_byte).to_bson()
                    )
                await websocket.send(
                    e.Speech(
                        end_of_stream=end_of_stream,
                        speech=samples.tobytes(),
                    ).to_bson()
                )

            # Send end of speech if started and no speech is recgonzied.
            if started and (ep.state == "NO_SPEECH" or end_of_stream):
                if end_of_stream:
                    end_byte = ep.last_sample * 2
                else:
                    end_byte = ep.stop * 2
                ep._end_of_buffer = False
                started = False
                await websocket.send(e.EndOfSpeech(end_byte=end_byte).to_bson())

            if end_of_stream:  # Close connection
                await websocket.send(e.EndOfStream(end_of_stream=True).to_bson())
                return

            # Get message
            msg = await websocket.recv()
    except websockets.exceptions.ConnectionClosedOK:
        logger.warning("Client disconnect.")
    finally:
        logger.warning("Client disconnect.")
