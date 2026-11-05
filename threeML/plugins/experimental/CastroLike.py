import warnings

from .castro_like import CastroLike as CastroLike  # noqa: F401

warnings.warn(
    f"Importing plugins like 'from {__name__} import {__name__.split('.')[-1]}' is "
    + f"deprecated; use 'from threeML.plugins import {__name__.split('.')[-1]}'.",
    DeprecationWarning,
    stacklevel=2,
)
