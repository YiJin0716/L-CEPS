"""Compute the two IPSS paths used by the current model."""
from threading import RLock
import numpy as np
import torch
from .ipss_path import compute_stability_paths
_PATH_LOCK=RLock()

def compute_paths(x, y, seed=302):
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an integer in [0, 2**32).')
    paths = {}
    with _PATH_LOCK:
        numpy_state, torch_state = np.random.get_state(), torch.get_rng_state()
        try:
            for selector in ('gb', 'l1'):
                np.random.seed(seed)
                torch.manual_seed(seed)
                paths[selector] = compute_stability_paths(x, y.copy(), selector=selector, n_jobs=1)
        finally:
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
    return paths
