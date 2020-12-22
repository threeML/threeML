import pytest
from threeML import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def test_plot():

    fig, subs_ = plt.subplots(2, 2)

    subs = subs_.flatten()

    x = [1, 2, 3]
    y = [4, 5, 6]

    subs[0].plot(x, y)

    subs[1].plot(x, y, ".")

    subs[2].plot(x, y, "o")

    subs[3].hist(np.random.normal(size=100))

    subs[3].set_xlabel("Test label")

    subs[0].set_ylabel("Test label")

    fig.suptitle("TEST")


def test_create_new_style():

    my_style = create_new_plotting_style()

    # Change something
    my_style["lines.linewidth"] = 2.7

    # Save it, but first make sure the file doesn't already exists (maybe from a crashed previous test)
    if os.path.exists("__test_style"):

        os.remove("__test_style")

    pathname = my_style.save("__test_style")

    assert os.path.exists(pathname)

    # Make sure we throw an exception if the file already exists
    with pytest.raises(IOError):

        my_style.save("__test_style")

    # Now make sure we are able to overwrite
    my_style.save("__test_style", overwrite=True)

    # Remove the file to avoid leaving files around
    os.remove(pathname)


def test_using_plot_style_context_manager():

    # Now create a new style with something different from the default one
    new_style = create_new_plotting_style()

    # Set axes green in the new style
    new_style["axes.edgecolor"] = "green"

    # Save
    style_filename = new_style.save("__green")

    assert os.path.exists(style_filename)

    # Now let's try to use it
    # Copy what we have in the default style (probably black)
    old_value = mpl.rcParams["axes.edgecolor"]

    with plot_style("__green"):

        # Verify that at the moment the edgecolor is indeed green
        assert mpl.rcParams["axes.edgecolor"] == "green"

        # Make a plot just to make sure
        test_plot()

    # Verify that we are back to what we had before
    assert mpl.rcParams["axes.edgecolor"] == old_value

    assert "__green" in get_available_plotting_styles()

    # Remove the file we created
    os.remove(style_filename)
