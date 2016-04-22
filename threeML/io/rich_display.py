# This module handle the lazy dependence on IPython


def fallback_display(x):

    print(x)

try:

    from IPython.core.display import display

except ImportError:

    display = fallback_display


