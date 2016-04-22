from contextlib import contextmanager

from IPython.parallel import Client

from threeML.config.config import threeML_config


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

    # Set the use-parallel feature on

    threeML_config['parallel']['use-parallel'] = True

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


class ParallelClient(Client):

    def __init__(self, *args, **kwargs):
        """
        Wrapper around the IPython Client class, which forces the use of dill for object serialization

        :param args: same as IPython Client
        :param kwargs: same as IPython Client
        :return:
        """

        # Just a wrapper around the IPython Client class
        # forcing the use of dill for object serialization
        # (more robust, and allows for serialization of class
        # methods)

        if 'profile' not in kwargs.keys():

            kwargs['profile'] = threeML_config['parallel']['IPython profile name']

        super(ParallelClient, self).__init__(*args, **kwargs)

        # This will propagate the use_dill to all running
        # engines
        self.direct_view().use_dill()

    def get_number_of_engines(self):

        return len(self.direct_view())
