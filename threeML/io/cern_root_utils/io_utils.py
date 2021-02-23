import ROOT
import contextlib


def get_list_of_keys(root_file, dir=""):
    """
    Given a ROOT file, it returns the list of object names contained in the file in the provided directory.

    :param root_file: a ROOT.TFile instance
    :param dir: the directory (default: "", i.e., the root of the file)
    :return: a list of object names
    """

    root_file.cd(dir)

    return [key.GetName() for key in ROOT.gDirectory.GetListOfKeys()]


@contextlib.contextmanager
def open_ROOT_file(filename):
    """
    Open a ROOT file in a context. Will close it no matter what, even if there are exceptions

    :param filename:
    :return:
    """

    f = ROOT.TFile(filename)

    try:

        yield f

    finally:

        f.Close()

        del f
