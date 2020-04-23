# Functions to detect whether we are running inside a notebook or not

from IPython import get_ipython


def is_inside_notebook():

    ip = get_ipython()

    if ip is None:

        # This happens if we are running in a python session, not a IPython one (for example in a script)
        return False

    else:

        # We are running in a IPython session, either in a console or in a notebook
        if ip.has_trait("kernel"):

            # We are in a notebook
            return True

        else:

            # We are not in a notebook
            return False
