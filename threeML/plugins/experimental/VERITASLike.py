import warnings
from .veritas_like import (  # noqa: F401
    VERITASLike as VERITASLike,
)

warnings.warn(
    f"Importing plugins like 'from {__name__} import {__name__.split('.')[-1]}' is "
    + f"deprecated; use 'from threeML.plugins import {__name__.split('.')[-1]}'.",
    DeprecationWarning,
    stacklevel=2,
)
