from builtins import range
import root_numpy
import numpy as np


def tree_to_ndarray(tree, *args, **kwargs):

    return root_numpy.tree2array(tree, *args, **kwargs)  # type: np.ndarray


def tgraph_to_arrays(tgraph):

    # To read a TGraph we need to iterate over the points

    n_points = tgraph.GetN()

    x_buffer = tgraph.GetX()
    y_buffer = tgraph.GetY()

    x = np.array([float(x_buffer[i]) for i in range(n_points)])  # type: np.ndarray
    y = np.array([float(y_buffer[i]) for i in range(n_points)])  # type: np.ndarray

    return x, y


def _get_edges(taxis, n):

    edges = np.zeros(n+1)

    for i in range(n):

        edges[i] = taxis.GetBinLowEdge(i)

    edges[-1] = taxis.GetBinUpEdge(n - 1)

    return edges  # type: np.array


def th2_to_arrays(th2):

    # NOTE: man how much I hate ROOT!
    # So much effort to do the simplest thing...

    # Get first all the lower edges of the bins, then the last edge is the upper edge of the last bin
    # (so these edges can be used in np.histogram)
    n_x = th2.GetNbinsX()
    xax = th2.GetXaxis()
    x_edges = _get_edges(xax, n_x)

    # Same for y
    n_y = th2.GetNbinsY()
    yax = th2.GetYaxis()
    y_edges = _get_edges(yax, n_y)

    # Finally get the array of values
    matrix = root_numpy.hist2array(th2)

    return x_edges, y_edges, matrix
