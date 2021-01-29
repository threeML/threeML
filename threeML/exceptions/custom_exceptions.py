import warnings as custom_warnings
import inspect
import functools

# Monkeypatch the print of warning so we can customize them


def my_format_warning(message, category, filename, lineo, line=None):
    """
    Override the default showwarning to customize the appearance of warnings
    :return:
    """
    # if message.message.find("may indicate binary incompatibility") >= 0:
    #     return ''
    return "\nWARNING %s: %s\n\n" % (category.__name__, message)


custom_warnings.formatwarning = my_format_warning


class NegativeBackground(RuntimeError):
    pass


class TriggerDoesNotExist(RuntimeError):
    pass


class ForbiddenRegionOfParameterSpace(RuntimeWarning):

    pass


class CppInterfaceNotAvailable(ImportWarning):

    pass


class CannotImportPlugin(ImportWarning):

    pass


class LikelihoodIsInfinite(RuntimeWarning):

    pass


class FitFailed(RuntimeError):
    pass


class ConfigurationFileCorrupt(RuntimeWarning):
    pass


class BadCovariance(RuntimeError):
    pass


class MinLargerMax(RuntimeError):
    pass


class NoFitYet(RuntimeError):
    pass


class DetDoesNotExist(RuntimeError):
    pass


class MinimizerNotAvailable(RuntimeError):
    pass


class TimeTypeNotKnown(RuntimeError):
    pass


class deprecated(object):
    def __init__(self, reason):
        if inspect.isclass(reason) or inspect.isfunction(reason):
            raise TypeError("Reason for deprecation must be supplied")
        self.reason = reason

    def __call__(self, cls_or_func):
        if inspect.isfunction(cls_or_func):
            if hasattr(cls_or_func, "func_code"):
                _code = cls_or_func.func_code
            else:
                _code = cls_or_func.__code__
            fmt = "Call to deprecated function or method {name} ({reason})."
            filename = _code.co_filename
            lineno = _code.co_firstlineno + 1

        elif inspect.isclass(cls_or_func):
            fmt = "Call to deprecated class {name} ({reason})."
            filename = cls_or_func.__module__
            lineno = 1

        else:
            raise TypeError(type(cls_or_func))

        msg = fmt.format(name=cls_or_func.__name__, reason=self.reason)

        @functools.wraps(cls_or_func)
        def new_func(*args, **kwargs):
            custom_warnings.simplefilter(
                "always", DeprecationWarning
            )  # turn off filter
            custom_warnings.warn_explicit(
                msg, category=DeprecationWarning, filename=filename, lineno=lineno
            )
            custom_warnings.simplefilter("default", DeprecationWarning)  # reset filter
            return cls_or_func(*args, **kwargs)

        return new_func
