import contextlib
import sys


class DummyFile(object):
    def write(self, x): pass


@contextlib.contextmanager
def suppress_stdout():
    """
    Temporarily suppress the output from a function

    :return: None
    """

    save_stdout = sys.stdout

    sys.stdout = DummyFile()

    yield

    sys.stdout = save_stdout