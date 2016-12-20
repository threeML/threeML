import warnings as custom_warnings

# Monkeypatch the print of warning so we can customize them

def my_format_warning(message, category, *args):
    """
    Override the default showwarning to customize the appearance of warnings
    :return:
    """

    if message.message.find("may indicate binary incompatibility") >= 0:

        return ''

    return "\nWARNING %s: %s\n\n" % (category.__name__, message)


custom_warnings.formatwarning = my_format_warning


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
