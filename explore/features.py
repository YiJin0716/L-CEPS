"""Observable dataset and selection-path features; never accepts simulation truth."""

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict


def standardized(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-10)


def describe(values, prefix):
    values = np.asarray(values).ravel()
    if not values.size:
        values = np.zeros(1)
    return {prefix + suffix: float(value) for suffix, value in zip(
        ('_mean', '_sd', '_q50', '_q90', '_q99', '_max'),
        [values.mean(), values.std(), *np.quantile(values, [.5, .9, .99]), values.max()])}


def dataset_features(x, y):
    """Scale/order invariant summaries, including even and odd nonlinear signal.

    CV uses fixed folds independent of simulation seeds. No rho, SNR, transform,
    true-support size, or response-family metadata is supplied to the model.
    """
    x, y = standardized(x), standardized(np.asarray(y).ravel())
    n, p = x.shape
    if n < 5 or p < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Require finite X/y, n >= 5, and p >= 2.')
    corr = x.T @ x / n
    absolute = np.abs(corr)
    np.fill_diagonal(absolute, 0)
    offdiag = absolute[~np.eye(p, dtype=bool)]
    eig = np.maximum(np.linalg.eigvalsh(corr), 0)
    features = dict(log_n=np.log(n), log_p=np.log(p), p_over_n=p/n,
                    effective_rank=p*p/max(np.sum(eig**2), 1e-10)/p,
                    top_eigen_fraction=eig[-1]/p,
                    top5_eigen_fraction=eig[-min(5, p):].sum()/p,
                    x_kurtosis=float(np.mean(x**4)), y_kurtosis=float(np.mean(y**4)))
    features.update(describe(offdiag, 'corr'))
    features.update(describe(absolute.max(axis=1), 'neighbor'))
    features.update(describe((absolute**2).sum(axis=1), 'corr_energy'))
    for threshold in (.3, .5, .7, .9):
        features[f'corr_fraction_{threshold}'] = float(np.mean(offdiag > threshold))
    cv = KFold(5, shuffle=True, random_state=417)
    for name, transformed in (('linear', x), ('square', x*x),
                              ('bump', np.exp(-.5*x*x)), ('sin', np.sin(2*x)),
                              ('rank', rankdata(x, axis=0))):
        z = standardized(transformed)
        assoc = np.abs(z.T @ y / n)
        features.update(describe(assoc, name + '_assoc'))
        if name in ('linear', 'square', 'bump'):
            # Regularization scales with sample size and dimension.
            prediction = cross_val_predict(Ridge(alpha=max(n, p)*.1), z, y, cv=cv)
            features[name + '_cv_r2'] = float(1 - np.mean((y-prediction)**2))
    features['nonlinear_cv_gain'] = max(features['square_cv_r2'], features['bump_cv_r2']) - features['linear_cv_r2']
    return features


def path_features(paths):
    features = dict(preselected_fraction=0., path_length=0., default_delta=paths['default_delta'])
    probabilities = np.zeros(1) if paths['null'] else paths['stability_paths'].max(axis=0)
    features.update(describe(probabilities, 'stability_max'))
    if not paths['null']:
        features['preselected_fraction'] = paths['p_preselected']/paths['p_full']
        features['path_length'] = len(paths['alphas'])
    return features

