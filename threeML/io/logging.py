import logging
import logging.handlers as handlers
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

from astromodels.utils.logging import (ColoredFormatter, LogFilter,
                                       _console_formatter, _dev_formatter,
                                       _usr_formatter,
                                       astromodels_console_log_handler,
                                       astromodels_dev_log_handler,
                                       astromodels_usr_log_handler)
from colorama import Back, Fore, Style

from threeML.config.config import threeML_config

# set up the console logging


def get_path_of_log_dir() -> Path:
    """
    get the path to the logging directory
    """

    log_path: Path = Path(threeML_config["logging"]["path"])

    if not log_path.exists():

        log_path.mkdir(parents=True)

    return log_path


_log_file_names = ["usr.log", "dev.log"]


def get_path_of_log_file(log_file: str) -> Path:
    """
    returns the path of the log files
    """
    assert log_file in _log_file_names, f"{log_file} is not one of {_log_file_names}"

    return get_path_of_log_dir() / log_file


# now create the developer handler that rotates every day and keeps
# 10 days worth of backup
threeML_dev_log_handler = handlers.TimedRotatingFileHandler(
    get_path_of_log_file("dev.log"), when="D", interval=1, backupCount=10
)


# lots of info written out

threeML_dev_log_handler.setFormatter(_dev_formatter)
threeML_dev_log_handler.setLevel(logging.DEBUG)


# now set up the usr log which will save the info

threeML_usr_log_handler = handlers.TimedRotatingFileHandler(
    get_path_of_log_file("usr.log"), when="D", interval=1, backupCount=10
)

threeML_usr_log_handler.setLevel(logging.INFO)

# lots of info written out

threeML_usr_log_handler.setFormatter(_usr_formatter)

# now set up the console logger


threeML_console_log_handler = logging.StreamHandler(sys.stdout)
threeML_console_log_handler.setFormatter(_console_formatter)
threeML_console_log_handler.setLevel(threeML_config["logging"]["level"])


astromodels_console_log_handler.setLevel(
    threeML_config["logging"]["level"])


warning_filter = LogFilter(logging.WARNING)


def silence_warnings():
    """
    supress warning messages in console and file usr logs
    """

    threeML_usr_log_handler.addFilter(warning_filter)
    threeML_console_log_handler.addFilter(warning_filter)

    astromodels_usr_log_handler.addFilter(warning_filter)
    astromodels_console_log_handler.addFilter(warning_filter)


def activate_warnings():
    """
    supress warning messages in console and file usr logs
    """

    threeML_usr_log_handler.removeFilter(warning_filter)
    threeML_console_log_handler.removeFilter(warning_filter)

    astromodels_usr_log_handler.removeFilter(warning_filter)
    astromodels_console_log_handler.removeFilter(warning_filter)


def update_logging_level(level):
    """
    update the logging level to the console
    """
    threeML_console_log_handler.setLevel(level)

    astromodels_console_log_handler.setLevel(level)


def silence_logs():
    """
    Turn off all logging 
    """

    log = logging.getLogger("threeML")

    for handler in log.handers:

        handler.setLevel(logging.CRITICAL)

    log = logging.getLogger("astromodels")

    for handler in log.handers:

        handler.setLevel(logging.CRITICAL)


def active_logs():
    """
    re-activate silenced logs
    """

    pass

        
@contextmanager
def silence_console_log():

    current_console_logging_level = threeML_console_log_handler.level
    current_usr_logging_level = threeML_usr_log_handler.level

    threeML_console_log_handler.setLevel(logging.ERROR)
    threeML_usr_log_handler.setLevel(logging.ERROR)

    try:
        yield

    finally:

        threeML_console_log_handler.setLevel(current_console_logging_level)
        threeML_usr_log_handler.setLevel(current_usr_logging_level)


def setup_logger(name):

    # A logger with name name will be created
    # and then add it to the print stream
    log = logging.getLogger(name)

    # this must be set to allow debug messages through
    log.setLevel(logging.DEBUG)

    # add the handlers

    if threeML_config["logging"]["developer"]:
        log.addHandler(threeML_dev_log_handler)

    else:

        # if we do not want to log developer
        # for 3ML, then lets not for astromodels

        astromodels_dev_log_handler.setLeveL(logging.CRITICAL)

    if threeML_config["logging"]["console"]:

        log.addHandler(threeML_console_log_handler)

    if threeML_config["logging"]["usr"]:
        log.addHandler(threeML_usr_log_handler)

    # we do not want to duplicate teh messages in the parents
    log.propagate = False

    return log
