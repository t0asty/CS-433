# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_mse


def least_squares(y, tx):
    """
        calculate the least squares.
        returns optimal weights, MSE
    """
    tx_t = np.transpose(tx)
    weights = np.linalg.solve(tx_t.dot(tx), tx_t.dot(y))
    mse = compute_mse(y, tx, weights)

    return weights, mse
