from __future__ import print_function
from builtins import object
import matplotlib as mpl
import matplotlib.pyplot as plt
import contextlib
import glob
import os
import yaml

from threeML.io.package_data import get_path_of_data_file, get_path_of_user_dir
from astromodels.utils.valid_variable import is_valid_variable_name


def check_legal_plot_style_name(style_name):

    if style_name not in defined_styles:

        raise NameError(
            "Style '%s' is not known. Valid styles: %s"
            % (style_name, ",".join(list(defined_styles.keys())))
        )


class PlotStyle(object):
    """
    Contains the styles for the plots. It embeds the matplotlib style, so that by choosing
    an instance of PlotStyle the user can set at the same time the matplotlib style and
    all the elements of the 3ML style.
    """

    def __init__(
        self,
        matplotlib_base_style="seaborn-notebook",
        matplotlib_overrides=None,
        threeml_style=None,
    ):

        assert matplotlib_base_style in plt.style.available, (
            "Style %s is not among the known matplotlib styles" % matplotlib_base_style
        )

        self._matplotlib_base_style = matplotlib_base_style

        self._matplotlib_overrides = (
            {} if matplotlib_overrides is None else matplotlib_overrides
        )

        self._threeml_style = {} if threeml_style is None else threeml_style

    @classmethod
    def from_style_file(cls, filename):

        # Read style file
        with open(filename) as f:

            d = yaml.load(f, Loader=yaml.SafeLoader)

        return cls(
            matplotlib_base_style=d["matplotlib_base_style"],
            matplotlib_overrides=d["matplotlib_overrides"],
            threeml_style=d["threeml_style"],
        )

    def clone(self):
        """
        Clone this style
        """

        clone = PlotStyle(
            matplotlib_base_style=self._matplotlib_base_style,
            matplotlib_overrides=dict(self._matplotlib_overrides),
            threeml_style=dict(self._threeml_style),
        )

        return clone

    def activate(self):
        """
        Activate this style so that it becomes the default style for any plot. This is mainly useful for the
        default style. For any other style, use the `with plot_style([style name])` context manager instead.

        :return: None
        """

        # Activate matplotlib base style
        mpl.style.use(self._matplotlib_base_style)

        # Override some settings if needed
        mpl.rcParams.update(self._matplotlib_overrides)

        # Use this style as active style
        global current_style

        current_style = self

    @staticmethod
    def deactivate():
        """
        Deactivate the current style and restore the default. Do not use this directly. Use the
        `with plot_style([style name])` context manager instead.

        :return: None
        """

        # Restore matplotlib defaults

        mpl.rcdefaults()

        # Restore 3ML default

        global current_style

        current_style = defined_styles["default"]

    @staticmethod
    def _check_name(name):

        if not is_valid_variable_name(name):
            raise NameError(
                "The name '%s' is not valid. Please use a simple name with no spaces nor "
                "special characters." % (name)
            )

    def save(self, name, overwrite=False):
        """
        Save the style with the provided name, so it will be made available also in future sessions of 3ML.

        :param name: the name to give to the new style
        :param overwrite: whether to overwrite an existing style with the same name or not
        :return: the path of the YAML file in which the style has been saved for future use
        """

        # Make sure name is legal
        self._check_name(name)

        # Make sure we are not trying to overwrite the default style

        assert name != "default", "You cannot overwrite the default style"

        # Get the list of existing styles

        defined_styles = _discover_styles()

        # Prepare dictionary to be written
        d = {}
        d["matplotlib_base_style"] = self._matplotlib_base_style
        d["matplotlib_overrides"] = self._matplotlib_overrides
        d["threeml_style"] = self._threeml_style

        # Write it
        # Save in the style directory
        this_path = os.path.join(_get_styles_directory(), "%s.yml" % name)

        # Check whether it exists already.

        if this_path in defined_styles and not overwrite:
            raise IOError(
                "Style %s already exists. Use 'overwrite=True' to overwrite it." % name
            )

        # If necessary, create the styles directory (needed the first time that the user
        # save a custom style)
        if not os.path.exists(_get_styles_directory()):

            os.makedirs(_get_styles_directory())

        # At this point, either the file is new or we are overwriting, so we can open with "w+"
        with open(this_path, "w+") as f:

            yaml.dump(d, f)

        print("Successfully written style into %s" % this_path)

        # Refresh the list of defined styles so the new style can be used immediately
        _refresh_defined_styles()

        # Return the path
        return this_path

    @staticmethod
    def _raise_unknown_element(item):

        raise NameError("'%s' is not a known style element" % item)

    def __setitem__(self, item, setting):

        if item not in self._threeml_style:

            if item in mpl.rcParams:

                self._matplotlib_overrides[item] = setting

            else:

                self._raise_unknown_element(item)

        else:

            self._threeml_style[item] = setting

    def __getitem__(self, item):

        if item in self._threeml_style:

            return self._threeml_style[item]

        else:

            if item in mpl.rcParams:

                return mpl.rcParams[item]

            elif item in self._matplotlib_overrides:

                return self._matplotlib_overrides[item]

            else:

                self._raise_unknown_element(item)


@contextlib.contextmanager
def plot_style(style_name):
    """
    A context manager to temporarily change the plotting style to the provided style.

    Examples:

    Say we have defined a style 'plain'::

        with plot_style('plain'):

            # plots generated here will have the 'plain' style

            ...

        # plots generated here will have the default style
        ...

    You can also temporarily change an attribute of the style within the `with` context::

        with plot_style('plain') as my_style:

            # Temporarily change the width of the lines. Outside of this particular "with" context
            # reusing the "plain" style will result in normal lines

            my_style['lines.linewidth'] = 2

            # plots generated here will have the 'plain' style with lines with double width

        # Plots generated here will have the default style

        ...


    :param style_name: name of the style. Use `get_available_plotting_styles()` to get a list of known styles.
    :return: style instance
    """

    check_legal_plot_style_name(style_name)

    # Get the PlotStyle instances corresponding to the provided style.
    # We clone the style so that the user can temporarily change anything in the style
    # within the `with` statement only temporarily affecting the plots. After the `with`
    # context is done, the original style will be unaffected

    style = defined_styles[style_name].clone()

    # Activate

    style.activate()

    # Return control to caller yielding the clone of the style instance.

    yield style

    # After the caller is done, restore default

    style.deactivate()


def create_new_plotting_style(based_on="default"):
    """
    Create a new plotting style ready for customization, based on an existing plotting style. By default, the
    default plotting style is used.

    :param based_on: the plot style to clone. By default, the default plotting style is used.
    :return: a PlotStyle instance ready for customization
    """

    check_legal_plot_style_name(based_on)

    return defined_styles[based_on].clone()


def _get_styles_directory():
    return os.path.join(get_path_of_user_dir(), "styles")


def _discover_styles():
    # Scan the 3ML styles directory for styles

    styles = glob.glob(os.path.join(_get_styles_directory(), "*.yml"))

    return styles


def _load_styles():

    # Discover defined styles

    styles = _discover_styles()

    # Load them

    defined_styles = {}

    for style_file in styles:
        this_style = PlotStyle.from_style_file(style_file)

        # The name of the style is just the file name without the .yml extension
        style_name = os.path.splitext(os.path.basename(style_file))[0]

        defined_styles[style_name] = this_style

    # Now load the default style
    default_style_filename = get_path_of_data_file("default_style.yml")

    defined_styles["default"] = PlotStyle.from_style_file(default_style_filename)

    return defined_styles


def get_available_plotting_styles():
    return list(defined_styles.keys())


# Load them on import
defined_styles = _load_styles()

current_style = defined_styles["default"]


# This is used to refresh the list on demand
def _refresh_defined_styles():

    global defined_styles

    defined_styles = _load_styles()
