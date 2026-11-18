"""Observed X/y feature construction for the final uncalibrated model."""

import numpy as np

from .features import dataset_features, path_features
from .weak import weak_conditional_features
from .ols import ols_features, signal_density_features
from .sparse import sparse_features
from .saturating import saturating_features
from .paths import efp_scores
from .conditional import conditional_features, ranking_features



def observed_records(x, y, paths, features=None):
    if features is None:
        features = dataset_features(x, y)
    base, names, extra = conditional_features(x, y)
    weak, weak_names, weak_extra = weak_conditional_features(x, y)
    ols, ols_names, ols_extra = ols_features(x, y)
    sparse, sparse_names, sparse_extra = sparse_features(x, y)
    tanh, tanh_names, tanh_extra = saturating_features(x, y)
    global_features = {**features, **extra, **weak_extra, **ols_extra, **sparse_extra, **tanh_extra}
    global_features['all_nonlinear_cv_gain'] = max(global_features['nonlinear_cv_gain'],
        global_features['sin_cv_r2'] - global_features['linear_cv_r2'])
    records = {}
    for selector, path in paths.items():
        matrix, feature_names = ranking_features(base, names, path)
        matrix = np.column_stack([matrix, weak, ols, sparse, tanh])
        feature_names += weak_names+ols_names+sparse_names+tanh_names
        context = {**global_features, **signal_density_features(matrix, feature_names),
                   **path_features(path), 'selector_gb':float(selector == 'gb')}
        records[selector] = dict(selector=selector, features=context, feature_x=matrix,
            feature_names=feature_names, feature_version=6, paths=path, efp=efp_scores(path, 1.))
    if set(records) == {'l1', 'gb'}:
        for selector, record in records.items():
            peer = records['l1' if selector == 'gb' else 'gb']
            record.update(peer_paths=peer['paths'], peer_feature_x=peer['feature_x'])
    return records
