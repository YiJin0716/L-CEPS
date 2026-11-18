"""Observed saturating-response evidence and local correlation adjustments.

All statistics are predictive features,
not asserted p-values. Local regressions use correlations in observed X only;
they do not receive the covariance family, block size, rho, or true support.
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold

from .features import standardized
from .sparse import canonical_coefficients


def local_precision(gram, max_neighbors=10, threshold=.5):
    """Regress each column on its strongest observed correlated neighbors.

    The returned rows have unit covariance with their own column and remove
    the linear projection on included neighbors, up to a numerical ridge.
    This remains usable when the full p-by-p empirical Gram matrix is singular.
    """
    p = len(gram)
    weights = np.eye(p)
    counts = np.zeros(p)
    for j in range(p):
        correlation = np.abs(gram[j]).copy()
        correlation[j] = 0.
        indices = np.flatnonzero(correlation > threshold)
        if len(indices) > max_neighbors:
            cutoff = np.sort(correlation[indices])[-max_neighbors]
            # Keep boundary ties together to preserve column permutation symmetry.
            indices = indices[correlation[indices] >= cutoff]
        counts[j] = len(indices)
        if len(indices):
            block = gram[np.ix_(indices, indices)] + 1e-6*np.eye(len(indices))
            weights[j, indices] = -np.linalg.solve(block, gram[indices, j])
    normalizer = np.maximum(np.sum(weights*gram, axis=1), 1e-10)
    weights /= normalizer[:, None]
    variance = np.maximum(np.sum((weights@gram)*weights, axis=1), 1e-15)
    return weights, variance, counts


def saturating_features(x, y):
    x = standardized(x)
    y = standardized(np.asarray(y).ravel())
    z = np.asfortranarray(standardized(np.tanh(x)))
    n, p = z.shape
    cv = KFold(5, shuffle=True, random_state=417)
    alphas = max(float(np.max(np.abs(z.T@y))/n), 1e-6)*np.geomspace(1., .01, 40)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        fitted = LassoCV(alphas=alphas, cv=cv, n_jobs=1, max_iter=5000, tol=1e-4).fit(z, y)
        mse = fitted.mse_path_.mean(axis=1)
        best = int(np.argmin(mse))
        cutoff = mse[best]+fitted.mse_path_[best].std(ddof=1)/np.sqrt(5)
        conservative = int(np.flatnonzero(mse <= cutoff)[0])
        sparse = Lasso(alpha=float(fitted.alphas_[conservative]), max_iter=10000, tol=1e-5).fit(z, y)
    fitted.coef_ = canonical_coefficients(fitted.coef_)
    sparse.coef_ = canonical_coefficients(sparse.coef_)
    beta = sparse.coef_
    residual = y-sparse.predict(z)
    active = np.count_nonzero(beta)
    sigma2 = float(np.sum(residual**2)/max(n-active-1, .1*n))
    gram = z.T@z/n
    score = z.T@residual/n
    names = ['tanh_lasso_beta', 'tanh_lasso_1se_beta']
    columns = [np.abs(fitted.coef_), np.abs(beta)]
    extra = {'tanh_sparse_cv_r2': float(1-mse[best]),
             'tanh_sparse_1se_cv_r2': float(1-mse[conservative]),
             'tanh_active_fraction': float(np.count_nonzero(fitted.coef_)/p),
             'tanh_1se_active_fraction': float(active/p),
             'tanh_residual_variance': sigma2,
             'tanh_lasso_alpha': float(fitted.alpha_),
             'tanh_lasso_1se_alpha': float(sparse.alpha),
             'tanh_convergence_warnings': float(sum(isinstance(w.message, ConvergenceWarning) for w in caught))}
    values, vectors = np.linalg.eigh(gram)
    for penalty in (.1, .01):
        inverse = (vectors/(np.maximum(values, 0)+penalty))@vectors.T
        inverse /= np.maximum(np.sum(inverse*gram, axis=1), 1e-10)[:, None]
        variance = np.maximum(np.sum((inverse@gram)*inverse, axis=1), 1e-15)
        corrected = beta+inverse@score
        columns.extend([np.abs(corrected), np.abs(corrected)/np.sqrt(sigma2*variance/n+1e-15)])
        names.extend([f'tanh_corrected_beta_{penalty}', f'tanh_corrected_t_{penalty}'])
    for neighbors in (5, 10):
        inverse, variance, counts = local_precision(gram, max_neighbors=neighbors)
        corrected = beta+inverse@score
        t = np.abs(corrected)/np.sqrt(sigma2*variance/n+1e-15)
        columns.extend([np.abs(corrected), t, counts])
        names.extend([f'tanh_local{neighbors}_beta', f'tanh_local{neighbors}_t', f'tanh_local{neighbors}_neighbors'])
        for quantile in (.5, .9, .99):
            extra[f'tanh_local{neighbors}_t_q{quantile}'] = float(np.quantile(t, quantile))
        for threshold in (2., 3., 5.):
            extra[f'tanh_local{neighbors}_t_above{threshold}'] = float(np.mean(t > threshold))
        energy = t*t
        extra[f'tanh_local{neighbors}_effective_fraction'] = float(energy.sum()**2/(np.sum(energy**2)+1e-15)/p)
    available = n-p-1 >= max(15, .1*n) and values[0] > max(values[-1], 1)*1e-10
    ols_beta, ols_t, robust_t = np.zeros((3, p))
    if available:
        inverse = (vectors/values)@vectors.T/n
        influence = inverse@z.T
        ols_beta = influence@y
        remainder = y-z@ols_beta
        variance = np.maximum(np.diag(inverse)*np.sum(remainder**2)/(n-p-1), 1e-20)
        ols_t = np.abs(ols_beta)/np.sqrt(variance)
        leverage = np.sum(z*influence.T, axis=1)+1/n
        robust_var = (influence**2)@(remainder/np.maximum(1-leverage, 1e-8))**2
        robust_t = np.abs(ols_beta)/np.sqrt(np.maximum(robust_var, 1e-20))
    extra['tanh_ols_available'] = float(available)
    columns.extend([np.abs(ols_beta), ols_t, robust_t])
    names.extend(['tanh_ols_beta', 'tanh_ols_t', 'tanh_ols_hc3_t'])
    matrix = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(matrix).all() or not all(np.isfinite(v) for v in extra.values()):
        raise ValueError('Nonfinite saturating-response evidence.')
    return matrix, names, extra
