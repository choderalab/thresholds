import numpy as np
from tqdm import tqdm


def probabilistic_bisection(noisy_oracle, search_interval=(0, 1), p=0.6, max_iterations=1000, resolution=100000,
                            early_termination_width=0):
    """Query the noisy_oracle at the median of the current belief distribution, then update the belief accordingly.
    Start from a uniform belief over the search_interval, and repeat n_iterations times.

    Parameters
    ----------
        noisy_oracle : stochastic function that accepts a float and returns a bool
            we assume that E[noisy_oracle(x)] is non-decreasing in x, and crosses 0.5 within the search_interval
        search_interval : tuple of floats
            left and right bounds on the search interval
        p : float
            assumed constant known probability of correct responses from noisy_oracle (must be > 0.5)
        max_iterations : int
            maximum number of times to query the noisy_oracle
        resolution : int
            how many bins to use when discretizing the search_interval
        early_termination_width : float
            if 95% of our belief is in an interval of this width or smaller, stop early

    Returns
    -------
        x : numpy.ndarray
            discretization of search_interval
        zs : list of bools
            oracle responses after each iteration
        fs : list of numpy.ndarrays
            belief pdfs after each iteration, including initial belief pdf

    References
    ----------
    [1] Bisection search with noisy responses (Waeber et al., 2013)
        http://epubs.siam.org/doi/abs/10.1137/120861898

    Notes
    -----
        For convenience / clarity / ease of implementation, we represent the belief pdf numerically, by uniformly
        discretizing the search_interval. This puts a cap on precision of the solution, which could be reached in as
        few as log_2(resolution) iterations (in the noiseless case). It is also wasteful of memory.
        Later, it would be better to represent the belief pdf using the recursive update equations, but I haven't yet
        figured out how to use them to find the median efficiently.
    """

    if p <= 0.5:
        raise (ValueError('the probability of correct responses must be > 0.5'))

    # initialize a uniform belief over the search interval
    start, stop = sorted(search_interval)
    x = np.linspace(start, stop, resolution)
    f = np.ones(len(x))
    f /= np.trapz(f, x)

    # initialize list of (discretized) belief pdfs
    fs = [f]

    # initialize empty list of oracle responses
    zs = []

    def get_quantile(f, alpha=0.05):
        return x[np.argmin(np.abs(np.cumsum(f) / np.sum(f) - alpha))]

    def get_belief_interval(f, fraction=0.95):
        eps = 0.5 * (1 - fraction)
        left, right = get_quantile(f, eps), get_quantile(f, 1 - eps)
        return left, right

    def describe_belief_interval(f, fraction=0.95):
        median = get_quantile(f, 0.5)
        left, right = get_belief_interval(f, fraction)
        description = "median: {:.3f}, {}% belief interval: ({:.3f}, {:.3f})".format(
            median, fraction * 100, left, right)
        return description

    trange = tqdm(range(max_iterations))
    for _ in trange:
        f = fs[-1]

        # query the oracle at median of previous belief pdf
        median = get_quantile(f, 0.5)
        z = noisy_oracle(median)
        zs.append(z)

        # update belief
        new_f = np.array(f)
        if z > 0:  # to handle noisy_oracles that return bools or binary
            new_f[np.where(x >= median)] *= p
            new_f[np.where(x < median)] *= (1 - p)
        else:
            new_f[np.where(x >= median)] *= (1 - p)
            new_f[np.where(x < median)] *= p
        new_f /= np.trapz(new_f, x)
        fs.append(new_f)

        trange.set_description(describe_belief_interval(new_f))

        belief_interval = get_belief_interval(new_f, 0.95)
        if (belief_interval[1] - belief_interval[0]) <= early_termination_width:
            break

    return x, zs, fs
