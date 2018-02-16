import numpy as np
import pytest

from thresholds import bisect


def test_probabilistic_bisection():
    # check that we converge to within discretization error of the correct answer in a noise-free case
    x_star = 1.0 / 3

    def noiseless_oracle(x):
        return x < x_star

    x, zs, fs = bisect.probabilistic_bisection(noiseless_oracle)
    dx = x[1] - x[0]
    solution = x[np.argmax(fs[-1])]
    assert (abs(solution - x_star) <= dx)

    def noisy_oracle(x):
        return (x + np.random.randn()) < x_star

    early_termination_width = 0.1
    x, zs, fs = bisect.probabilistic_bisection(noisy_oracle, early_termination_width=early_termination_width)
    solution = x[np.argmax(fs[-1])]
    assert (abs(solution - x_star) <= early_termination_width)

    with pytest.raises(ValueError):
        bisect.probabilistic_bisection(noiseless_oracle, p=0.4)
