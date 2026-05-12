import os

from .logging import (
    activate_logs,
    activate_progress_bars,
    activate_warnings,
    debug_mode,
    loud_mode,
    quiet_mode,
    setup_logger,
    silence_logs,
    silence_progress_bars,
    silence_warnings,
    toggle_progress_bars,
    update_logging_level,
)
from .plotting.get_style import get_threeML_style, set_threeML_style

# need this for pickling
from .serialization import *  # noqa: F401,F403

__all__ = [
    "activate_logs",
    "activate_progress_bars",
    "activate_warnings",
    "debug_mode",
    "loud_mode",
    "quiet_mode",
    "setup_logger",
    "silence_logs",
    "silence_progress_bars",
    "silence_warnings",
    "toggle_progress_bars",
    "update_logging_level",
    "get_threeML_style",
    "set_threeML_style",
    "apply_startup_settings",
]


def apply_startup_settings(cfg):
    """
    Apply IO-related startup settings based on the provided config.
    Intentionally not executed at import time to avoid circular imports.
    Call this after threeML_config is created.
    """
    log = setup_logger(__name__)
    log.propagate = False

    # Startup warnings
    try:
        if cfg.get("logging", {}).get("startup_warnings", False):
            log.info("Starting 3ML!")
            log.warning("WARNINGs here are [red]NOT[/red] errors")
            log.warning(
                "but are inform you about optional packages that can be installed"
            )
            log.warning(
                "[red] to disable these messages, turn off start_warning in your"
                " config file[/red]"
            )
    except Exception:
        # Be resilient: don't fail if cfg shape is unexpected
        pass

    # Configure headless backend if DISPLAY is not set
    if os.environ.get("DISPLAY") is None:
        try:
            if cfg.get("logging", {}).get("startup_warnings", False):
                log.warning(
                    "no display variable set. using backend for graphics without "
                    "display (agg)"
                )
        except Exception:
            pass

        # Set Agg before pyplot is imported anywhere
        try:
            import matplotlib as mpl

            mpl.use("Agg")
        except Exception:
            # Don't crash if matplotlib isn't available yet
            pass
    return log
