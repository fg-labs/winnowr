import logging
import sys
from collections.abc import Callable

import defopt

from winnowr.tools.evaluate import evaluate
from winnowr.tools.extract import extract
from winnowr.tools.score import score
from winnowr.tools.strlen import strlen
from winnowr.tools.train import train

_tools: list[Callable[..., None]] = [
    extract,
    train,
    evaluate,
    score,
    strlen,
]


def setup_logging(level: str = "INFO") -> None:
    """
    Set up basic logging to print to the console.

    Args:
        level: the logging level.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s:%(funcName)s:%(lineno)s [%(levelname)s]: %(message)s",
    )


def run() -> None:
    """Set up logging, then hand over to defopt for running command line tools."""
    setup_logging()
    logger = logging.getLogger("winnowr")
    logger.info("Executing: " + " ".join(sys.argv))
    defopt.run(
        funcs=_tools,
        argv=sys.argv[1:],
        version=True,
    )
    logger.info("Finished executing successfully.")
