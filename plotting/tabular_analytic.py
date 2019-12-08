import warnings

import numpy as np
import scipy.misc


def chain_p_solved():
    n = 19
    return np.sum([0.5 ** n * scipy.misc.comb(N=n, k=k) for k in range(15, 20)])


def tree_frac_solved(x):
    return 1 - (1 - 0.5 ** 10) ** x


def chain_frac_solved(x):
    p = chain_p_solved()
    return 1 - (1 - p) ** x


def median_solve_time_tree(n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.log(1 / 2) / np.log(1 - (1 / 2) ** n)
