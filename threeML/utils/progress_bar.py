from matplotlib import cm
from matplotlib.colors import to_hex
import numpy as np

from tqdm.auto import tqdm as _tqdm
from tqdm.auto import trange as _trange


from threeML.config.config import threeML_config

# _colors = ["#9C04FF","#E0DD18","#0B92FC","#06F86D","#FD4409"]

#_colors = []

c_itr = 0


class _Get_Color(object):

    def __init__(self, n_colors=5):

        cmap = cm.get_cmap(
            threeML_config["interface"]["multi_progress_cmap"])

        self._colors = [to_hex(c) for c in cmap(np.linspace(0,1,n_colors))]

        self.c_itr = 0
        self.n_colors = n_colors

    def color(self):

        if threeML_config["interface"]["multi_progress_color"]:

            color = self._colors[self.c_itr]

            if self.c_itr < self.n_colors - 1:

                self.c_itr += 1

            else:

                self.c_itr = 0

        else:

            color = threeML_config["interface"]["progress_bar_color"]

        return color


_get_color = _Get_Color(n_colors=8)


def tqdm(itr=None, **kwargs):

    color = _get_color.color()

    return (_tqdm(itr, colour=color, **kwargs) if threeML_config["interface"]["show_progress_bars"] else itr)


def trange(*args, **kwargs):

    color = _get_color.color()

    return (_trange(*args, colour=color, **kwargs) if threeML_config["interface"]["show_progress_bars"] else range(*args))


__all__ = ["tqdm", "trange"]



