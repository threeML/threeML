# Author: Giacomo Vianello (giacomov@stanford.edu)


import sys

from threeML.utils.progress_bar import tqdm
import numexpr
import numpy as np


from threeML.io.logging import setup_logger

logger = setup_logger(__name__)

__all__ = ["bayesian_blocks", "bayesian_blocks_not_unique"]


def bayesian_blocks_not_unique(tt, ttstart, ttstop, p0):
    # Verify that the input array is one-dimensional
    tt = np.asarray(tt, dtype=float)

    assert tt.ndim == 1

    # Now create the array of unique times

    unique_t = np.unique(tt)

    t = tt
    tstart = ttstart
    tstop = ttstop

    # Create initial cell edges (Voronoi tessellation) using the unique time stamps

    edges = np.concatenate([[tstart], 0.5 * (unique_t[1:] + unique_t[:-1]), [tstop]])

    # The last block length is 0 by definition
    block_length = tstop - edges

    if np.sum((block_length <= 0)) > 1:
        raise RuntimeError(
            "Events appears to be out of order! Check for order, or duplicated events."
        )

    N = unique_t.shape[0]

    # arrays to store the best configuration
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    # Pre-computed priors (for speed)
    # eq. 21 from Scargle 2012

    priors = 4 - np.log(73.53 * p0 * np.power(np.arange(1, N + 1), -0.478))

    # Count how many events are in each Voronoi cell

    x, _ = np.histogram(t, edges)

    # Speed tricks: resolve once for all the functions which will be used
    # in the loop
    cumsum = np.cumsum
    log = np.log
    argmax = np.argmax
    numexpr_evaluate = numexpr.evaluate
    arange = np.arange

    # Decide the step for reporting progress
    incr = max(int(float(N) / 100.0 * 10), 1)

    logger.debug("Finding blocks...")

    # This is where the computation happens. Following Scargle et al. 2012.
    # This loop has been optimized for speed:
    # * the expression for the fitness function has been rewritten to
    #  avoid multiple log computations, and to avoid power computations
    # * the use of scipy.weave and numexpr has been evaluated. The latter
    #  gives a big gain (~40%) if used for the fitness function. No other
    #  gain is obtained by using it anywhere else

    # Set numexpr precision to low (more than enough for us), which is
    # faster than high
    oldaccuracy = numexpr.set_vml_accuracy_mode("low")
    numexpr.set_num_threads(1)
    numexpr.set_vml_num_threads(1)

    

    for R in tqdm(range(N)):
        br = block_length[R + 1]
        T_k = block_length[: R + 1] - br

        # N_k: number of elements in each block
        # This expression has been simplified for the case of
        # unbinned events (i.e., one element in each block)
        # It was:
        N_k = cumsum(x[: R + 1][::-1])[::-1]
        # Now it is:
        # N_k = arange(R + 1, 0, -1)

        # Evaluate fitness function
        # This is the slowest part, which I'm speeding up by using
        # numexpr. It provides a ~40% gain in execution speed.

        fit_vec = numexpr_evaluate(
            """N_k * log(N_k/ T_k) """,
            optimization="aggressive",
            local_dict={"N_k": N_k, "T_k": T_k},
        )

        p = priors[R]

        A_R = fit_vec - p

        A_R[1:] += best[:R]

        i_max = argmax(A_R)

        last[R] = i_max
        best[R] = A_R[i_max]

        

    numexpr.set_vml_accuracy_mode(oldaccuracy)

    logger.debug("Done\n")

    # Now find blocks
    change_points = np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind

        if ind == 0:
            break

        ind = last[ind - 1]

    change_points = change_points[i_cp:]

    finalEdges = edges[change_points]

    return np.asarray(finalEdges)


def bayesian_blocks(tt, ttstart, ttstop, p0, bkg_integral_distribution=None):
    """
    Divide a series of events characterized by their arrival time in blocks
    of perceptibly constant count rate. If the background integral distribution
    is given, divide the series in blocks where the difference with respect to
    the background is perceptibly constant.

    :param tt: arrival times of the events
    :param ttstart: the start of the interval
    :param ttstop: the stop of the interval
    :param p0: the false positive probability. This is used to decide the penalization on the likelihood, so this
    parameter affects the number of blocks
    :param bkg_integral_distribution: (default: None) If given, the algorithm account for the presence of the background and
    finds changes in rate with respect to the background
    :return: the np.array containing the edges of the blocks
    """

    # Verify that the input array is one-dimensional
    tt = np.asarray(tt, dtype=float)

    assert tt.ndim == 1

    if bkg_integral_distribution is not None:

        # Transforming the inhomogeneous Poisson process into an homogeneous one with rate 1,
        # by changing the time axis according to the background rate
        logger.debug(
            "Transforming the inhomogeneous Poisson process to a homogeneous one with rate 1..."
        )
        t = np.array(bkg_integral_distribution(tt))
        logger.debug("done")

        # Now compute the start and stop time in the new system
        tstart = bkg_integral_distribution(ttstart)
        tstop = bkg_integral_distribution(ttstop)

    else:

        t = tt
        tstart = ttstart
        tstop = ttstop

    # Create initial cell edges (Voronoi tessellation)
    edges = np.concatenate([[t[0]], 0.5 * (t[1:] + t[:-1]), [t[-1]]])

    # Create the edges also in the original time system
    edges_ = np.concatenate([[tt[0]], 0.5 * (tt[1:] + tt[:-1]), [tt[-1]]])

    # Create a lookup table to be able to transform back from the transformed system
    # to the original one
    lookup_table = {key: value for (key, value) in zip(edges, edges_)}

    # The last block length is 0 by definition
    block_length = tstop - edges

    if np.sum((block_length <= 0)) > 1:

        raise RuntimeError(
            "Events appears to be out of order! Check for order, or duplicated events."
        )

    N = t.shape[0]

    # arrays to store the best configuration
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    # eq. 21 from Scargle 2012
    prior = 4 - np.log(73.53 * p0 * (N ** -0.478))

    logger.debug("Finding blocks...")

    # This is where the computation happens. Following Scargle et al. 2012.
    # This loop has been optimized for speed:
    # * the expression for the fitness function has been rewritten to
    #  avoid multiple log computations, and to avoid power computations
    # * the use of scipy.weave and numexpr has been evaluated. The latter
    #  gives a big gain (~40%) if used for the fitness function. No other
    #  gain is obtained by using it anywhere else

    # Set numexpr precision to low (more than enough for us), which is
    # faster than high
    oldaccuracy = numexpr.set_vml_accuracy_mode("low")
    numexpr.set_num_threads(1)
    numexpr.set_vml_num_threads(1)

    # Speed tricks: resolve once for all the functions which will be used
    # in the loop
    numexpr_evaluate = numexpr.evaluate
    numexpr_re_evaluate = numexpr.re_evaluate

    # Pre-compute this

    aranges = np.arange(N + 1, 0, -1)

    for R in range(N):
        br = block_length[R + 1]
        T_k = (
            block_length[: R + 1] - br
        )  # this looks like it is not used, but it actually is,
        # inside the numexpr expression

        # N_k: number of elements in each block
        # This expression has been simplified for the case of
        # unbinned events (i.e., one element in each block)
        # It was:
        # N_k = cumsum(x[:R + 1][::-1])[::-1]
        # Now it is:
        N_k = aranges[N - R :]
        # where aranges has been pre-computed

        # Evaluate fitness function
        # This is the slowest part, which I'm speeding up by using
        # numexpr. It provides a ~40% gain in execution speed.

        # The first time we need to "compile" the expression in numexpr,
        # all the other times we can reuse it

        if R == 0:

            fit_vec = numexpr_evaluate(
                """N_k * log(N_k/ T_k) """,
                optimization="aggressive",
                local_dict={"N_k": N_k, "T_k": T_k},
            )

        else:

            fit_vec = numexpr_re_evaluate(local_dict={"N_k": N_k, "T_k": T_k})

        A_R = fit_vec - prior  # type: np.ndarray

        A_R[1:] += best[:R]

        i_max = A_R.argmax()

        last[R] = i_max
        best[R] = A_R[i_max]

    numexpr.set_vml_accuracy_mode(oldaccuracy)

    logger.debug("Done\n")

    # Now peel off and find the blocks (see the algorithm in Scargle et al.)
    change_points = np.zeros(N, dtype=int)
    i_cp = N
    ind = N

    while True:

        i_cp -= 1

        change_points[i_cp] = ind

        if ind == 0:

            break

        ind = last[ind - 1]

    change_points = change_points[i_cp:]

    edg = edges[change_points]

    # Transform the found edges back into the original time system

    if bkg_integral_distribution is not None:

        final_edges = [lookup_table[x] for x in edg]

    else:

        final_edges = edg

    # Now fix the first and last edge so that they are tstart and tstop
    final_edges[0] = ttstart
    final_edges[-1] = ttstop

    return np.asarray(final_edges)


# To be run with a profiler
if __name__ == "__main__":

    tt = np.random.uniform(0, 1000, int(sys.argv[1]))
    tt.sort()

    with open("sim.txt", "w+") as f:
        for t in tt:
            f.write("%s\n" % (t))

    res = bayesian_blocks(tt, 0, 1000, 1e-3, None)
    print(res)
