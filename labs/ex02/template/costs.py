# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w, loss_function='mse'):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    if loss_function == 'mse':   
        e = y - tx.dot(w)
    
        return 1/(2 * len(y)) * np.transpose(e).dot(e)

    elif loss_function == 'mae':
        e = y - tx.dot(w)
        
        return 1/len(y) * sum(np.abs(e))
    
    else:
        raise NotImplementedError