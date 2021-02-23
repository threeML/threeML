from builtins import range

__author__ = "grburgess"

import matplotlib.pyplot as plt
import numpy as np


# reverse these colormaps so that it goes from light to dark

REVERSE_CMAP = ["summer", "autumn", "winter", "spring", "copper"]

# clip some colormaps so the colors aren't too light

CMAP_RANGE = dict(
    gray={"start": 200, "stop": 0},
    Blues={"start": 60, "stop": 255},
    Oranges={"start": 100, "stop": 255},
    OrRd={"start": 60, "stop": 255},
    BuGn={"start": 60, "stop": 255},
    PuRd={"start": 60, "stop": 255},
    YlGn={"start": 60, "stop": 255},
    YlGnBu={"start": 60, "stop": 255},
    YlOrBr={"start": 60, "stop": 255},
    YlOrRd={"start": 60, "stop": 255},
    hot={"start": 230, "stop": 0},
    bone={"start": 200, "stop": 0},
    pink={"start": 160, "stop": 0},
)


def cmap_intervals(length=50, cmap="YlOrBr", start=None, stop=None):
    """
    Return evenly spaced intervals of a given colormap `cmap`.

    Colormaps listed in REVERSE_CMAP will be cycled in reverse order.
    Certain colormaps have pre-specified color ranges in CMAP_RANGE. These module
    variables ensure that colors cycle from light to dark and light colors are
    not too close to white.


    :param length: int the number of colors used before cycling back to first color. When
    length is large (> ~10), it is difficult to distinguish between
    successive lines because successive colors are very similar.
    :param cmap: str name of a matplotlib colormap (see matplotlib.pyplot.cm)
    """
    cm = plt.cm.get_cmap(cmap)

    # qualitative color maps
    if cmap in [
        "Accent",
        "Dark2",
        "Paired",
        "Pastel1",
        "Pastel2",
        "Set1",
        "Set2",
        "Set3",
        "Vega10",
        "Vega20",
        "Vega20b",
        "Vega20c",
    ]:

        base_n_colors = cm.N

        cmap_list = cm(list(range(base_n_colors)))

        if base_n_colors < length:

            factor = int(np.floor_divide(length, base_n_colors))

            cmap_list = np.tile(cmap_list, (factor, 1))

        return cmap_list

    crange = CMAP_RANGE.get(cmap, dict(start=0, stop=255))
    if cmap in REVERSE_CMAP:
        crange = dict(start=crange["stop"], stop=crange["start"])
    if start is not None:
        crange["start"] = start
    if stop is not None:
        crange["stop"] = stop

    idx = np.linspace(crange["start"], crange["stop"], length).astype(np.int)
    return cm(idx)
