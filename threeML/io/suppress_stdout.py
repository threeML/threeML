from builtins import object
import contextlib
import sys


class _DummyFile(object):
    def write(self, x):
        pass

    def flush(self, *args, **kwargs):
        pass


@contextlib.contextmanager
def suppress_stdout():
    """
    Temporarily suppress the output from a function

    :return: None
    """

    save_stdout = sys.stdout

    sys.stdout = _DummyFile()

    yield

    sys.stdout = save_stdout
