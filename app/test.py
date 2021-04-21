# -*- coding: utf-8 -*-
import asyncio
import bson
import websockets
from pprint import (
    pprint,
)

import soundfile as sf
import methods as e

async def send_wav(websocket):
    path = "/app/tests/unit/res/nasceu-8k.wav"
    (wav, _,) = sf.read(path, dtype="<i2")  # PCM linear 16bit
    data = wav.tobytes()  # Converte para bytes
    pprint(f"Size WAV {len(data)}")

    for i in range(0, len(data) - 100, 100):
        message = e.SendAudio(audio=data[i: i + 100], end_of_stream=False).to_bson()
        await websocket.send(message)
    message = e.SendAudio(audio=data[i:], end_of_stream=True).to_bson()
    await websocket.send(message)


async def event_listener(websocket):
    buffer = bytearray()
    async for msg in websocket:
        res = bson.decode(msg)
        pprint(res)
        if "speech" in res:
            buffer = b"".join(
                [
                    buffer,
                    res["speech"],
                ]
            )
        if "start_byte" in res:
            pprint("start_byte: {}".format(res["start_byte"]))
        if "end_byte" in res:
            pprint("end_byte: {}".format(res["end_byte"]))
        if "end_of_stream" in res and res["end_of_stream"] is True:
            return buffer


async def client():
    async with websockets.connect("ws://localhost:8765") as websocket:
        consumer_task = asyncio.create_task(event_listener(websocket))
        producer_task = asyncio.create_task(send_wav(websocket))
        (value, _,) = await asyncio.gather(
            consumer_task,
            producer_task,
        )
        print(f"Final Audio Size {value}")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client())
