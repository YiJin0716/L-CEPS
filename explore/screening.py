"""Screen by the union of ten IPSS selections, then apply the small MLP."""
import numpy as np

from .expanded_f import ExpandedF
from .generalized import prefix_selection
from .observed import observed_records
from .paths import efp_scores, q_from_efp
from .predict import compute_paths

DELTAS = (-1., 0., 1., 2., 4.)
RULE = 'union10_at_target'


def union_qvalues(paths):
    """Minimum IPSS q-value across GB/Lasso and the five fixed deltas."""
    if set(paths) != {'gb', 'l1'}:
        raise ValueError('Both GB and Lasso IPSS paths are required.')
    p = paths['gb']['p_full']
    if paths['l1']['p_full'] != p:
        raise ValueError('IPSS paths have inconsistent column counts.')
    q = np.full(p, np.inf)
    for selector in ('gb', 'l1'):
        for delta in DELTAS:
            q = np.minimum(q, q_from_efp(efp_scores(paths[selector], delta)))
    if not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('Nonfinite or negative screening q-values.')
    return q


def screened_selection(probability, screening_q, targets):
    """Recompute complete-tie prefixes within the candidate set at each target."""
    probability = np.asarray(probability, dtype=float)
    screening_q = np.asarray(screening_q, dtype=float)
    targets = np.atleast_1d(np.asarray(targets, dtype=float))
    prefix_selection(probability, targets)  # Shared input validation.
    if (screening_q.shape != probability.shape or not np.isfinite(screening_q).all()
            or np.any(screening_q < 0)):
        raise ValueError('Require one finite, nonnegative screening q-value per variable.')
    selected, averages, endpoints, counts = [], [], [], []
    for alpha in targets:
        indices = np.flatnonzero(screening_q <= alpha)
        counts.append(len(indices))
        if len(indices):
            sets, risk, ends = prefix_selection(probability[indices], [alpha])
            chosen = indices[sets[0]]
        else:
            chosen = np.empty(0, dtype=int)
            risk = np.empty(0, dtype=float)
            ends = np.empty(0, dtype=int)
        selected.append(chosen)
        averages.append(risk)
        endpoints.append(ends)
    return selected, averages, endpoints, counts


class ScreenedMLP(ExpandedF):
    def select(self, x, y, targets=(.05, .1, .2), *, random_seed=302, paths=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if x.ndim != 2 or len(x) != len(y) or len(y) < 5 or x.shape[1] < 2:
            raise ValueError('Require n-by-p X and length-n y, n>=5 and p>=2.')
        if not np.isfinite(x).all() or not np.isfinite(y).all() or y.std() <= 1e-10:
            raise ValueError('Require finite X/y and nonconstant y.')
        targets = np.atleast_1d(np.asarray(targets, dtype=float))
        prefix_selection(np.array([.5]), targets)
        paths = compute_paths(x, y, random_seed) if paths is None else paths
        if set(paths) != {'gb', 'l1'} or any(v['p_full'] != x.shape[1] for v in paths.values()):
            raise ValueError('Both correctly sized IPSS paths are required.')
        screening_q = union_qvalues(paths)
        # Shared measurements always use full X, preserving the fitted input meaning.
        required = np.flatnonzero(screening_q <= max(targets, default=-1.))
        probability = np.ones(x.shape[1], dtype=np.float32)
        if len(required):
            record = observed_records(x, y, paths)['gb']
            probability[required] = self.estimator.predict_proba(self.matrix(record)[required])[:, 1]
        selected, risk, ends, counts = screened_selection(probability, screening_q, targets)
        return dict(targets=targets.tolist(), selected=[s.tolist() for s in selected],
                    final_counts=[len(s) for s in selected], candidate_counts=counts,
                    total_variables=x.shape[1], variables_scored=len(required),
                    screening_rule=RULE, screening_deltas=list(DELTAS),
                    estimated_prefix_fdp=[r.tolist() for r in risk],
                    prefix_counts=[e.tolist() for e in ends], random_seed=int(random_seed))

