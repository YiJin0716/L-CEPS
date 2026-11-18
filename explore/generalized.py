"""Complete-tie prefix selection used by the current model."""
import numpy as np

def prefix_selection(probability, targets):
    """Select the largest complete score-tie prefix with mean score <= alpha."""
    probability = np.asarray(probability, dtype=float)
    targets = np.atleast_1d(np.asarray(targets, dtype=float))
    if (probability.ndim != 1 or not len(probability)
            or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError('Variable scores must be a nonempty vector in [0, 1].')
    if (targets.ndim != 1 or not np.isfinite(targets).all()
            or np.any((targets < 0) | (targets > 1))):
        raise ValueError('Target FDR values must be finite numbers in [0, 1].')
    order = np.argsort(probability, kind='stable')
    ordered = probability[order]
    endpoints = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True]) + 1
    averages = np.cumsum(ordered, dtype=float)[endpoints - 1] / endpoints
    selected = []
    for target in targets:
        valid = endpoints[averages <= target]
        selected.append(np.sort(order[:valid[-1]]) if len(valid) else np.empty(0, dtype=int))
    return selected, averages, endpoints
