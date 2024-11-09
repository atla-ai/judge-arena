from dataclasses import dataclass
from datetime import datetime
import logging


def get_logger(sink_name: str = "core_utils") -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s "
        "[%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
        force=True,
    )
    logger = logging.getLogger(sink_name)
    return logger


@dataclass
class Vote:
    timestamp: str
    prompt: str
    response_a: str
    response_b: str
    model_a: str
    model_b: str
    winner: str
    judge_id: str
