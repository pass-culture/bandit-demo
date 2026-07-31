import logging


def setup_logging(level: int = logging.INFO) -> None:
    """
    Lightweight logging configuration

    Features:
    - timestamp
    - level
    - module name
    - message

    Levels:
    - DEBUG: Detailed information, typically of interest only when diagnosing problems.
    - INFO: Confirmation that things are working as expected.
    - WARNING: An indication that something unexpected happened. The software is still working as expected, but it may indicate a potential problem in the future.
    - ERROR: Due to a more serious problem, the software has not been able to perform some function.
    - CRITICAL: A very serious error, indicating that the program itself may be unable to continue running.

    """  # noqa: E501, W505

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
