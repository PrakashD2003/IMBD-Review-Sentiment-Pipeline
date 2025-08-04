"""Custom exceptions and helpers for detailed error reporting."""

import logging
import sys
from typing import Optional


def format_and_log_exception(
    exc: Exception,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Build a detailed error message including file, line, and original exception,
    then log it at ERROR level.

    :param exc: The caught exception instance.
    :param logger: Logger to use (defaults to root logger).
    :return: The formatted error message.
    """
    # Grab the traceback and find the last frame
    _, _, tb = sys.exc_info()
    # Walk to the innermost frame
    while tb.tb_next:
        tb = tb.tb_next
    frame = tb.tb_frame
    filename = frame.f_code.co_filename
    lineno = tb.tb_lineno

    msg = f"Error in {filename}, line {lineno}: {exc}"

    if logger:
        logger.error(msg)
    else:
        logging.error(msg)

    return msg


class DetailedException(Exception):
    """
    Exception that automatically formats and logs its own traceback location.
    """

    def __init__(
        self,
        exc: Exception,
        logger: Optional[logging.Logger] = None
    ):
        """
        Wraps an existing exception, logs file/line context, and carries
        a detailed message forward.

        :param exc: The original exception to wrap.
        :param logger: Optional logger for error output.
        """
        # Format and log
        detailed_msg = format_and_log_exception(exc, logger)
        # Initialize base with the detailed message
        super().__init__(detailed_msg)



