from contextlib import contextmanager

has_parallel = False

try:

    from ipyparallel import Client

except ImportError:

    # Try with the old ipython, which contained the parallel code
    try:

        from IPython.parallel import Client

    except ImportError:

        has_parallel = False

    else:

        has_parallel = True

else:

    has_parallel = True

from threeML.config.config import threeML_config

# Custom warning
import warnings
import exceptions

class NoParallelEnvironment(exceptions.UserWarning):
    pass

# Set up the warnings module to always display our custom warning (otherwise it would only be displayed once)
warnings.simplefilter('always', NoParallelEnvironment)

@contextmanager
def parallel_computation(profile=None):
    """
    A context manager which turns on parallel execution temporarily

    :param profile: the profile to use, if different from the default
    :return:
    """

    # Memorize the state of the use-parallel config

    old_state = bool(threeML_config['parallel']['use-parallel'])

    old_profile = str(threeML_config['parallel']['IPython profile name'])

    # Set the use-parallel feature on, if available

    if has_parallel:

        threeML_config['parallel']['use-parallel'] = True

    else:

        # No parallel environment available. Issue a warning and continue with serial computation

        warnings.warn("You requested parallel computation, but no parallel environment is available. You need "
                      "to install the ipyparallel package. Continuing with serial computation...",
                      NoParallelEnvironment)

        threeML_config['parallel']['use-parallel'] = False

    # Now use the specified profile (if any), otherwise the default one

    if profile is not None:

        threeML_config['parallel']['IPython profile name'] = str(profile)

    # Here is where the content of the with parallel_computation statement gets
    # executed

    try:

        yield

    finally:

        # This gets executed in any case, even if there is an exception

        # Revert back
        threeML_config['parallel']['use-parallel'] = old_state

        threeML_config['parallel']['IPython profile name'] = old_profile


if has_parallel:
    class ParallelClient(Client):

        def __init__(self, *args, **kwargs):
            """
            Wrapper around the IPython Client class, which forces the use of dill for object serialization

            :param args: same as IPython Client
            :param kwargs: same as IPython Client
            :return:
            """

            import pdb;pdb.set_trace()

            # Just a wrapper around the IPython Client class
            # forcing the use of dill for object serialization
            # (more robust, and allows for serialization of class
            # methods)

            if 'profile' not in kwargs.keys():

                kwargs['profile'] = threeML_config['parallel']['IPython profile name']

            super(ParallelClient, self).__init__(*args, **kwargs)

            # This will propagate the use_dill to all running
            # engines
            _ = self.direct_view().use_dill()

        def get_number_of_engines(self):

            return len(self.direct_view())

else:

    # NO parallel environment available. Make a dumb object to avoid import problems, but this object will never
    # be really used because the context manager will not activate the parallel mode (see above)
    class ParallelClient(object):

        def __init__(self, *args, **kwargs):

            raise RuntimeError("No parallel environment and attempted to use the ParallelClient class, which should "
                               "never happen. Please open an issue at https://github.com/giacomov/3ML/issues")
