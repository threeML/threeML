import pytest
from threeML import *
import matplotlib.pyplot as plt


def a_plot():

    fig, subs_ = plt.subplots(2, 2)

    subs = subs_.flatten()

    x = [1, 2, 3]
    y = [4, 5, 6]

    subs[0].plot(x, y)

    subs[1].plot(x, y, '.')

    subs[2].plot(x, y, 'o')

    subs[3].hist(np.random.normal(size=100))

    subs[3].set_xlabel("Test label")

    subs[0].set_ylabel("Test label")

    fig.suptitle("TEST")


def get_new_style():

    my_style = create_new_plotting_style()

    # Change something
    my_style['lines.linewidth'] = 2.7

    return my_style


def test_using_plot_style_context_manager():

    # Check that
    pass