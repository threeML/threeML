# This module handle the lazy dependence on IPython
import pandas as pd
import numpy as np
def fallback_display(x):

    print(x)

try:

    from IPython.core.display import display

except ImportError:

    display = fallback_display


## Panda table styles


def hover(hover_color="#98DBFF"):
    return dict(selector="tr:hover",
                props=[("background-color", "%s" % hover_color)])


def panda_table_styler(panda_table, caption, color="#FF8B76" ,precision=None,):
    """ Display a table pandas styled"""

    row_style = [
        hover(),
        dict(selector="th", props=[("font-size", "110%"),
                                   ("text-align", "center"),
                                   ('color', 'y'),
                                   ('background-color', color)]),
        dict(selector="td", props=[("font-size", "100%"),
                                   ("text-align", "center"),
                                   ('color', 'k')
                                   ]),
        dict(selector="caption", props=[("caption-side", "top")])]

    if precision is not None:

        return panda_table.style.set_table_styles(row_style).set_caption(caption).set_precision(precision)

    else:

        return panda_table.style.set_table_styles(row_style).set_caption(caption)




def panda_matrix_styler(panda_table, caption,cmap=None, precision=None):
    """ Display a matrix pandas styled"""

    matrix_style = [
        dict(selector="caption", props=[("caption-side", "top")])]


    # Scale the values so that the table is readable and colors
    # correspond to realistic correlations
    vals = panda_table.values[np.isfinite(panda_table.values)]

    delta = vals.max() - vals.min()


    low = (-2.-vals.min())/(-delta)

    high = (2.-vals.max())/(delta)

    if precision is not None:


        pd.set_option('float_format', '{:2.2f}'.format)

        return panda_table.style.set_table_styles(matrix_style).set_caption(caption).set_precision(precision).background_gradient(cmap=cmap,low=low,high=high).highlight_null('yellow')

    else:

        return panda_table.style.set_table_style(matrix_style).set_caption(caption).background_gradient(cmap=cmap,low=low,high=high).highlight_null('yellow')


