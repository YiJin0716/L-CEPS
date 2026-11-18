"""Reuse the repository's IPSS paths and preserve its EFP/q-value conventions."""

import numpy as np


def efp_scores(paths, delta):
    from ipss.processing import ipss_scores
    p_full = paths['p_full']
    if paths['null']:
        return np.full(p_full, float(p_full))
    scores, integral, _, _ = ipss_scores(paths['stability_paths'], paths['b'],
        paths['alphas'], paths['average_selected'], paths['ipss_function'], float(delta), .05)
    p = paths['p_preselected']
    efp = np.full(p_full, float(p))
    efp[paths['preselect_indices']] = np.round(integral / np.maximum(scores, integral/p), 8)
    if p != p_full:
        efp[efp >= p-1] = p_full
    return efp


def q_from_efp(efp):
    """Exact tied-rank implementation of ipss.helpers.compute_qvalues."""
    efp = np.asarray(efp)
    order = np.argsort(efp, kind='stable')
    scores = efp[order]
    # Every member of a tied group uses the full group size as denominator.
    right_ranks = np.searchsorted(scores, scores, side='right')
    q_sorted = np.minimum.accumulate((scores/right_ranks)[::-1])[::-1]
    q = np.empty_like(scores, dtype=float)
    q[order] = q_sorted
    return q


def counts_at_q(q, truth, targets):
    is_true = np.zeros(len(q), dtype=bool)
    is_true[np.asarray(truth, dtype=int)] = True
    selected = np.asarray(q)[:, None] <= np.asarray(targets)[None, :]
    tp = selected[is_true].sum(0)
    fp = selected[~is_true].sum(0)
    return tp, fp
