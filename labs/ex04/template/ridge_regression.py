# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_mse


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    tx_t = np.transpose(tx)
    N = tx_t.shape[0]
    linear_func = tx_t.dot(tx) + (2 * N * lambda_ * np.identity(N))
    Xty = tx_t.dot(y)
    weights = np.linalg.solve(linear_func, Xty)
    mse = compute_mse(y, tx, weights)
    return weights, mse
