# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    p = np.random.permutation(len(x))
    x, y = x[p], y[p]
    split_ind = int(ratio * len(x))
    return x[:split_ind], x[split_ind:], y[:split_ind], y[split_ind:]
    