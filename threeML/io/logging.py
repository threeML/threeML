import logging
import logging.handlers as handlers
import sys
from typing import Dict, Optional

from colorama import Back, Fore, Style

from threeML import threeML_config
from threeML.io.package_data import get_path_of_log_dir, get_path_of_log_file

## set up the console logging


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter.
    """

    def __init__(
        self, *args, colors: Optional[Dict[str, str]] = None, **kwargs
    ) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, "")
        record.reset = Style.RESET_ALL

        return super().format(record)


class MyFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno != self.__level


# make sure the logging directory is there
log_path = get_path_of_log_dir()
log_path.mkdir(parents=True, exist_ok=True)


# now create the developer handler that rotates every day and keeps
# 10 days worth of backup
threeML_dev_log_handler = handlers.TimedRotatingFileHandler(
    get_path_of_log_file("dev.log"), when="D", interval=1, backupCount=10
)

threeML_dev_log_handler.setLevel(logging.DEBUG)

# lots of info written out
_dev_formatter = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s| %(funcName)s | %(lineno)d | %(message)s"
)

threeML_dev_log_handler.setFormatter(_dev_formatter)

# now set up the usr log which will save the info

threeML_usr_log_handler = handlers.RotatingFileHandler(
    get_path_of_log_file("usr.log"), maxBytes=10000, backupCount=20
)

threeML_usr_log_handler.setLevel(logging.INFO)

# lots of info written out
_usr_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

threeML_usr_log_handler.setFormatter(_usr_formatter)

# now set up the console logger

_console_formatter = ColoredFormatter(
    "{asctime} |{color} {levelname:8} {reset}| {color} {message} {reset}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    colors={
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Back.WHITE + Style.BRIGHT,
    },
)


threeML_console_log_handler = logging.StreamHandler(sys.stdout)
threeML_console_log_handler.setFormatter(_console_formatter)


warning_filter = MyFilter(logging.WARNING)

def silence_warnings():
    """
    supress warning messages in console and file usr logs
    """

    threeML_usr_log_handler.addFilter(warning_filter)
    threeML_console_log_handler.addFilter(warning_filter)


def activate_warnings():
    """
    supress warning messages in console and file usr logs
    """

    threeML_usr_log_handler.removeFilter(warning_filter)
    threeML_console_log_handler.removeFilter(warning_filter)

    

def setup_logger(name):

    # A logger with name my_logger will be created
    # and then add it to the print stream
    log = logging.getLogger(name)

    log.setLevel(threeML_config["logging"]["level"])

    # add the handlers

    if threeML_config["logging"]["developer"]:
        log.addHandler(threeML_dev_log_handler)

    if threeML_config["logging"]["console"]:

        log.addHandler(threeML_console_log_handler)

    log.addHandler(threeML_usr_log_handler)

    return log
