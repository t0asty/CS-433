# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
from helpers import batch_iter
from costs import compute_loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - tx.dot(w)
    
    return - 1/len(y) * np.transpose(tx).dot(e)



def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, shuffle=True):
            grad = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w
            w = w - gamma * grad
            # store w and loss
            ws.append(w)
        loss = compute_loss(y, tx, w)
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws