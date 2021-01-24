import logging

from threeML.config.config import threeML_config
from threeML.io import (activate_logs, activate_progress_bars,
                        activate_warnings, debug_mode, loud_mode, quiet_mode,
                        silence_logs, silence_progress_bars, silence_warnings,
                        toggle_progress_bars, update_logging_level)
from threeML.io.logging import (astromodels_console_log_handler,
                                astromodels_dev_log_handler,
                                astromodels_usr_log_handler,
                                threeML_console_log_handler,
                                threeML_dev_log_handler,
                                threeML_usr_log_handler)
from threeML.utils.progress_bar import tqdm, trange


def test_all_toggles():

    toggle_progress_bars()

    activate_progress_bars()
    silence_progress_bars()

    silence_warnings()
    activate_warnings()

    update_logging_level("INFO")

    silence_logs()

    activate_logs()

    loud_mode()

    activate_logs()

    quiet_mode()

    activate_logs()


def test_progress_bars():

    threeML_config["interface"]["show_progress_bars"] = True

    toggle_progress_bars()

    assert not threeML_config["interface"]["show_progress_bars"]

    toggle_progress_bars()

    assert threeML_config["interface"]["show_progress_bars"]

    silence_progress_bars()

    for i in tqdm(range(10), desc="test"):
        pass

    for i in trange(1, 10, 1, desc="test"):
        pass

    assert not threeML_config["interface"]["show_progress_bars"]

    activate_progress_bars()

    for i in tqdm(range(10), desc="test"):
        pass

    for i in trange(1, 10, 1, desc="test"):
        pass

    assert threeML_config["interface"]["show_progress_bars"]


def test_logging_toggles():

    # restore base state
    activate_logs()

    assert threeML_console_log_handler.level == logging.INFO

    assert threeML_usr_log_handler.level == logging.INFO

    assert threeML_dev_log_handler.level == logging.DEBUG

    assert astromodels_console_log_handler.level == logging.INFO

    assert astromodels_usr_log_handler.level == logging.INFO

    assert astromodels_dev_log_handler.level == logging.DEBUG

    quiet_mode()

    assert threeML_console_log_handler.level == logging.CRITICAL

    assert threeML_usr_log_handler.level == logging.CRITICAL

    assert threeML_dev_log_handler.level == logging.CRITICAL

    assert astromodels_console_log_handler.level == logging.CRITICAL

    assert astromodels_usr_log_handler.level == logging.CRITICAL

    assert astromodels_dev_log_handler.level == logging.CRITICAL

    assert not threeML_config["interface"]["show_progress_bars"]

    activate_logs()

    assert threeML_console_log_handler.level == logging.INFO

    assert threeML_usr_log_handler.level == logging.INFO

    assert threeML_dev_log_handler.level == logging.DEBUG

    assert astromodels_console_log_handler.level == logging.INFO

    assert astromodels_usr_log_handler.level == logging.INFO

    assert astromodels_dev_log_handler.level == logging.DEBUG

    quiet_mode()

    loud_mode()

    assert threeML_console_log_handler.level == logging.INFO

    assert threeML_usr_log_handler.level == logging.INFO

    assert astromodels_console_log_handler.level == logging.INFO

    assert astromodels_usr_log_handler.level == logging.INFO

    # should restore back to quiet mode
    activate_logs()

    assert threeML_console_log_handler.level == logging.CRITICAL

    assert threeML_usr_log_handler.level == logging.CRITICAL

    assert astromodels_console_log_handler.level == logging.CRITICAL

    assert astromodels_usr_log_handler.level == logging.CRITICAL

    debug_mode()

    assert threeML_console_log_handler.level == logging.DEBUG

    assert threeML_usr_log_handler.level == logging.CRITICAL

    assert astromodels_console_log_handler.level == logging.DEBUG

    assert astromodels_usr_log_handler.level == logging.CRITICAL
