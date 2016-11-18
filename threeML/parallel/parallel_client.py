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
from threeML.io.progress_bar import progress_bar, multiple_progress_bars, CannotGenerateHTMLBar

# Custom warning
import warnings
import exceptions

import sys
import time
import re
import math


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

        def interactive_map(self, worker, items_to_process, ordered=True):
            """
            Subdivide the work among the active engines, taking care of dividing it among them

            :param worker: the function to be applied
            :param items_to_process: the items to apply the function to
            :param ordered: whether to keep the order of output (default: True). Using False can be much faster, but
            you need to have a way to re-estabilish the order if you care about it, after the fact.
            :return: a AsyncResult object
            """

            # Split the work evenly between the engines
            n_total_engines = self.get_number_of_engines()

            n_items = len(items_to_process)

            # Get a load-balanced view with the appropriate number of engines

            if n_items < n_total_engines:

                warnings.warn("More engines than items to process")

                # Limit the view to the needed engines

                lview = self.load_balanced_view(range(n_items))

                n_active_engines = n_items

                chunk_size = 1

            else:

                # Use all engines

                lview = self.load_balanced_view()

                n_active_engines = n_total_engines

                chunk_size = int(math.ceil(n_items / float(n_active_engines) / 20))

            return lview.imap(worker, items_to_process, chunksize=chunk_size, ordered=ordered)

        def execute_with_progress_bar(self, worker, items):

            # Let's make a wrapper which will allow us to recover the order
            def wrapper(x):

                (id, item) = x

                return (id, worker(item))

            items_wrapped = [(i, item) for i, item in enumerate(items)]

            amr = self.interactive_map(wrapper, items_wrapped, ordered=False)

            results = []

            n_iterations = len(items)

            with progress_bar(n_iterations) as p:

                for i, res in enumerate(amr):
                    results.append(res)

                    p.increase()

            # Reorder the list according to the id
            return map(lambda x:x[1], sorted(results, key=lambda x:x[0]))

        @staticmethod
        def fetch_progress_from_progress_bars(ar):

            while not ar.ready():

                stdouts = ar.stdout

                if not any(stdouts):

                    continue

                percentage_completed_engines = []

                for stdout in ar.stdout:

                    # Default value is 0

                    percentage_completed_engines.append(0)

                    if stdout:

                        idx = stdout.find('[')

                        if idx < 0:

                            continue

                        # Find the progress bar (if any)
                        tokens = re.findall('(\[[^\r^\)]+[\r\)]|\[.+completed.+\Z)', stdout[idx:][-1000:])

                        if len(tokens) > 0:

                            last_progress_bar = tokens[-1]

                            # Now extract the progress
                            percentage_completed = re.match('\[[\*\s]+([0-9]*(\.[0-9]+)?)\s?%[\*\s]+',
                                                            last_progress_bar)

                            if percentage_completed is None:

                                sys.stderr.write("\nCould not understand progress bar from engine: %s" %
                                                 last_progress_bar)

                            else:

                                percentage_completed_engines[-1] = float(percentage_completed.groups()[0])

                yield percentage_completed_engines

        def wait_watching_progress(self, ar, dt=5.0):
            """
            Report progress from the different engines

            :param ar:
            :param dt:
            :return:
            """

            n_engines = self.get_number_of_engines()

            print("Found %s engines" % n_engines)

            try:

                with multiple_progress_bars(iterations=100, n=n_engines) as bars:

                    # We are in the notebook, display a report of all the engines

                    for progress in self.fetch_progress_from_progress_bars(ar):

                        for i in range(n_engines):

                            bars[i].animate(progress[i])

                        time.sleep(dt)

            except CannotGenerateHTMLBar:

                # Fall back to text progress and one bar

                with progress_bar(100) as bar:

                    progress = self.fetch_progress_from_progress_bars(ar)

                    global_progress = sum(progress) / float(n_engines)

                    bar.animate(global_progress)

                    time.sleep(dt)


else:

    # NO parallel environment available. Make a dumb object to avoid import problems, but this object will never
    # be really used because the context manager will not activate the parallel mode (see above)
    class ParallelClient(object):

        def __init__(self, *args, **kwargs):

            raise RuntimeError("No parallel environment and attempted to use the ParallelClient class, which should "
                               "never happen. Please open an issue at https://github.com/giacomov/3ML/issues")
