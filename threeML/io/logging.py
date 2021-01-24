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

    log_path: Path = Path(threeML_config["logging"]["path"]).expanduser()

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


####
# These control the verbosity of 3ML
####


class LoggingState(object):

    def __init__(self, threeML_usr_log_handler, threeML_console_log_handler,
                 astromodels_usr_log_handler, astromodels_console_log_handler
                 ):
        """
        A container to store the stat of the logs
        """

        # attach the log handlers

        self.threeML_usr_log_handler = threeML_usr_log_handler
        self.threeML_console_log_handler = threeML_console_log_handler

        self.astromodels_usr_log_handler = astromodels_usr_log_handler
        self.astromodels_console_log_handler = astromodels_console_log_handler

        # store their current states

        self.threeML_usr_log_handler_state = threeML_usr_log_handler.level
        self.threeML_console_log_handler_state = threeML_console_log_handler.level

        self.astromodels_usr_log_handler_state = astromodels_usr_log_handler.level
        self.astromodels_console_log_handler_state = astromodels_console_log_handler.level

    def _store_state(self):

        self.threeML_usr_log_handler_state = threeML_usr_log_handler.level
        self.threeML_console_log_handler_state = threeML_console_log_handler.level

        self.astromodels_usr_log_handler_state = astromodels_usr_log_handler.level
        self.astromodels_console_log_handler_state = astromodels_console_log_handler.level

    def restore_last_state(self):

        self.threeML_usr_log_handler.setLevel(
            self.threeML_usr_log_handler_state)
        self.threeML_console_log_handler.setLevel(
            self.threeML_console_log_handler_state)

        self.astromodels_usr_log_handler.setLevel(
            self.astromodels_usr_log_handler_state)
        self.astromodels_console_log_handler.setLevel(
            self.astromodels_console_log_handler_state)

    def silence_logs(self):

        # store the state
        self._store_state()

        # silence the logs

        self.threeML_usr_log_handler.setLevel(
            logging.CRITICAL)
        self.threeML_console_log_handler.setLevel(
            logging.CRITICAL)

        self.astromodels_usr_log_handler.setLevel(
            logging.CRITICAL)
        self.astromodels_console_log_handler.setLevel(
            logging.CRITICAL)

    def loud_logs(self):

        # store the state
        self._store_state()

        # silence the logs

        self.threeML_usr_log_handler.setLevel(
            logging.INFO)
        self.threeML_console_log_handler.setLevel(
            logging.INFO)

        self.astromodels_usr_log_handler.setLevel(
            logging.INFO)
        self.astromodels_console_log_handler.setLevel(
            logging.INFO)

    def debug_logs(self):

        # store the state
        self._store_state()

        # silence the logs
        self.threeML_console_log_handler.setLevel(
            logging.DEBUG)

        self.astromodels_console_log_handler.setLevel(
            logging.DEBUG)


_log_state = LoggingState(threeML_usr_log_handler, threeML_console_log_handler,
                          astromodels_usr_log_handler, astromodels_console_log_handler)


def silence_progress_bars():
    """
    Turn off the progress bars
    """

    threeML_config["interface"]["show_progress_bars"] = False


def activate_progress_bars():
    """
    Turn on the progress bars
    """
    threeML_config["interface"]["show_progress_bars"] = True


def toggle_progress_bars():
    """
    toggle the state of the progress bars
    """
    state = threeML_config["interface"]["show_progress_bars"]

    threeML_config["interface"]["show_progress_bars"] = not state


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

    # handle dev logs independently
    threeML_dev_log_handler.setLevel(logging.CRITICAL)
    astromodels_dev_log_handler.setLevel(logging.CRITICAL)

    _log_state.silence_logs()


def quiet_mode():
    """
    turn off all logging and progress bars
    """

    silence_progress_bars()

    # save state and silence
    silence_logs()


def loud_mode():
    """
    turn on all progress bars and logging
    """

    activate_progress_bars()

    # save state and make loud
    _log_state.loud_logs()


def activate_logs():
    """
    re-activate silenced logs
    """

    # handle dev logs independently
    threeML_dev_log_handler.setLevel(logging.DEBUG)
    astromodels_dev_log_handler.setLevel(logging.DEBUG)

    _log_state.restore_last_state()


def debug_mode():
    """
    activate debug in the console
    """

    # store state and switch console to debug
    _log_state.debug_logs()


@contextmanager
def silence_console_log():
    """
    temporarily silence the console and progress bars
    """
    current_console_logging_level = threeML_console_log_handler.level
    current_usr_logging_level = threeML_usr_log_handler.level

    threeML_console_log_handler.setLevel(logging.ERROR)
    threeML_usr_log_handler.setLevel(logging.ERROR)

    progress_state = threeML_config["interface"]["show_progress_bars"]

    threeML_config["interface"]["show_progress_bars"] = False

    try:
        yield

    finally:

        threeML_console_log_handler.setLevel(current_console_logging_level)
        threeML_usr_log_handler.setLevel(current_usr_logging_level)

        threeML_config["interface"]["show_progress_bars"] = progress_state
       

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

        astromodels_dev_log_handler.setLevel(logging.CRITICAL)

    if threeML_config["logging"]["console"]:

        log.addHandler(threeML_console_log_handler)

    if threeML_config["logging"]["usr"]:
        log.addHandler(threeML_usr_log_handler)

    # we do not want to duplicate teh messages in the parents
    log.propagate = False

    return log
