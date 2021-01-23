# Custom warning
import math
import re
import signal
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from distutils.spawn import find_executable

from threeML.config.config import threeML_config
from threeML.io.logging import setup_logger
from threeML.utils.progress_bar import tqdm

log = setup_logger(__name__)

try:
    from subprocess import DEVNULL  # py3k
except ImportError:
    import os

    DEVNULL = open(os.devnull, "wb")

# Check whether we have a parallel system or not

has_parallel = False

try:

    from ipyparallel import Client

except ImportError:

    has_parallel = False

else:

    has_parallel = True


class NoParallelEnvironment(UserWarning):
    pass


# Set up the warnings module to always display our custom warning (otherwise it would only be displayed once)
warnings.simplefilter("always", NoParallelEnvironment)


@contextmanager
def parallel_computation(profile=None, start_cluster=True):
    """
    A context manager which turns on parallel execution temporarily

    :param profile: the profile to use, if different from the default
    :param start_cluster: True or False. Whether to start a new cluster. If False, try to use an existing one for the
    same profile
    :return:
    """

    # Memorize the state of the use-parallel config

    old_state = bool(threeML_config["parallel"]["use-parallel"])

    old_profile = str(threeML_config["parallel"]["IPython profile name"])

    # Set the use-parallel feature on, if available

    if has_parallel:

        threeML_config["parallel"]["use-parallel"] = True

    else:

        # No parallel environment available. Issue a warning and continue with serial computation

        log.warning(
            "You requested parallel computation, but no parallel environment is available. You need "
            "to install the ipyparallel package. Continuing with serial computation...",

        )

        threeML_config["parallel"]["use-parallel"] = False

    # Now use the specified profile (if any), otherwise the default one

    if profile is not None:

        threeML_config["parallel"]["IPython profile name"] = str(profile)

    # Here is where the content of the with parallel_computation statement gets
    # executed

    # See if we need to start the ipyparallel cluster first

    if start_cluster:

        # Get the command line together

        # First find out path of ipcluster
        ipcluster_path = find_executable("ipcluster")

        cmd_line = [ipcluster_path, "start"]

        if profile is not None:

            cmd_line.append(" --profile=%s" % profile)

        # Start process asynchronously with Popen, suppressing all output
        print("Starting ipyparallel cluster with this command line:")
        print(" ".join(cmd_line))

        ipycluster_process = subprocess.Popen(cmd_line)

        rc = Client()

        # Wait for the engines to become available

        while True:

            try:

                view = rc[:]

            except:

                time.sleep(0.5)

                continue

            else:

                print("%i engines are active" % (len(view)))

                break

        # Do whatever we need to do
        try:

            yield

        finally:

            # This gets executed in any case, even if there is an exception

            print("\nShutting down ipcluster...")

            ipycluster_process.send_signal(signal.SIGINT)

            ipycluster_process.wait()

    else:

        # Using an already started cluster

        yield

    # Revert back
    threeML_config["parallel"]["use-parallel"] = old_state

    threeML_config["parallel"]["IPython profile name"] = old_profile


def is_parallel_computation_active():

    return bool(threeML_config["parallel"]["use-parallel"])


if has_parallel:

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

            if "profile" not in kwargs.keys():

                kwargs["profile"] = threeML_config["parallel"]["IPython profile name"]

            super(ParallelClient, self).__init__(*args, **kwargs)

            # This will propagate the use_dill to all running
            # engines
            _ = self.direct_view().use_dill()

        def get_number_of_engines(self):

            return len(self.direct_view())

        def _interactive_map(
            self, worker, items_to_process, ordered=True, chunk_size=None
        ):
            """
            Subdivide the work among the active engines, taking care of dividing it among them

            :param worker: the function to be applied
            :param items_to_process: the items to apply the function to
            :param ordered: whether to keep the order of output (default: True). Using False can be much faster, but
            you need to have a way to re-estabilish the order if you care about it, after the fact.
            :param chunk_size: determine how many items should an engine process before reporting back. Use None for
            an automatic choice.
            :return: a AsyncResult object
            """

            # Split the work evenly between the engines
            n_total_engines = self.get_number_of_engines()

            n_items = len(items_to_process)

            # Get a load-balanced view with the appropriate number of engines

            if n_items < n_total_engines:

                log.warning("More engines than items to process")

                # Limit the view to the needed engines

                lview = self.load_balanced_view(range(n_items))

                n_active_engines = n_items

                chunk_size = 1

            else:

                # Use all engines

                lview = self.load_balanced_view()

                n_active_engines = n_total_engines

                if chunk_size is None:

                    chunk_size = int(
                        math.ceil(n_items / float(n_active_engines) / 20))

            # We need this to keep the instance alive
            self._current_amr = lview.imap(
                worker, items_to_process, chunksize=chunk_size, ordered=ordered
            )

            return self._current_amr

        def execute_with_progress_bar(self, worker, items, chunk_size=None, name="progress"):

            # Let's make a wrapper which will allow us to recover the order
            def wrapper(x):

                (id, item) = x

                return (id, worker(item))

            items_wrapped = [(i, item) for i, item in enumerate(items)]

            amr = self._interactive_map(
                wrapper, items_wrapped, ordered=False, chunk_size=chunk_size
            )

            results = []

            for i, res in enumerate(tqdm(amr, desc=name)):

                results.append(res)

            # Reorder the list according to the id
            return list(map(lambda x: x[1], sorted(results, key=lambda x: x[0])))


else:

    # NO parallel environment available. Make a dumb object to avoid import problems, but this object will never
    # be really used because the context manager will not activate the parallel mode (see above)
    class ParallelClient(object):
        def __init__(self, *args, **kwargs):

            raise RuntimeError(
                "No parallel environment and attempted to use the ParallelClient class, which should "
                "never happen. Please open an issue at https://github.com/giacomov/3ML/issues"
            )
