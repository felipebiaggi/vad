import websockets
import asyncio
import signal
import util
from ws import (
    vad,
)

util.set_logger_level()

logger = util.get_default_logger()

loop = asyncio.get_event_loop()


async def start_server(
    future,
):
    async with websockets.serve(
        vad,
        "0.0.0.0",
        8765,
        max_size=2 ** 23,
        max_queue=128,
    ):
        logger.info("Server Start")
        await future
        logger.info("Server Stop")


if __name__ == "__main__":
    future_signal = loop.create_future()

    loop.add_signal_handler(
        signal.SIGTERM,
        future_signal.set_result,
        None,
    )
    loop.run_until_complete(start_server(future_signal))
