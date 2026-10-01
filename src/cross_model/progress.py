"""Consistent stderr logging and progress reporting for cross-model runs."""
import logging
import sys

from tqdm import tqdm


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def progress(iterable=None, *, desc, total=None, unit="item", leave=True):
    """Always display a stderr bar, including in redirected experiment logs."""
    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave,
                dynamic_ncols=True, mininterval=0.5, file=sys.stderr, disable=False)
