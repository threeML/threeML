import warnings as custom_warnings


# Monkeypatch the print of warning so we can customize them

def my_format_warning(message, category, *args):
    """
    Override the default showwarning to customize the appearance of warnings
    :return:
    """

    return "\nWARNING %s: %s\n\n" % (category.__name__, message)


custom_warnings.formatwarning = my_format_warning


class ForbiddenRegionOfParameterSpace(Warning):

    pass


class CppInterfaceNotAvailable(Warning):

    pass


class CannotImportPlugin(Warning):

    pass


class LikelihoodIsInfinite(Warning):

    pass


class FitFailed(RuntimeError):
    pass