import logging
import os

DATEFMT = "[%Y-%m-%d %H:%M:%S %z]"
FMT = "%(asctime)-15s [%(process)d] [%(thread)d] [%(filename)s] [%(funcName)s] [%(levelname)s] %(message)s"


def set_logger_level():
    type_level = [
        "CRITICAL",
        "ERROR",
        "WARNING",
        "INFO",
        "DEBUG",
        "NOTSET",
    ]
    try:
        log_level = os.environ["LOG_LEVEL"]
        if log_level in type_level:
            logging.basicConfig(
                level=log_level,
                format=FMT,
                datefmt=DATEFMT,
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format=FMT,
                datefmt=DATEFMT,
            )
    except KeyError:
        logging.basicConfig(
            level=logging.INFO,
            format=FMT,
            datefmt=DATEFMT,
        )


def get_default_logger():
    logger = logging.getLogger(__name__)
    return logger
