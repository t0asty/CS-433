# -*- coding: utf-8 -*-
"""Gradient Descent"""
import costs

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    
    return - 1/len(y) * np.transpose(tx).dot(e)


def gradient_descent(y, tx, initial_w, max_iters, gamma, verbose=True):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = costs.compute_loss(y, tx, w)
        # update w
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if verbose:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws