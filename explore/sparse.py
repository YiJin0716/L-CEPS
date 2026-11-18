"""Numerically stable sparse evidence; canonicalize negligible Lasso coefficients.

All inputs are observed X/y. Residual correction is inspired by debiased
regression, but the regularized inverse and data-selected transform/penalty do
not provide valid p-values automatically. These are learned-model features.
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold

from .features import standardized
from .ols import TRANSFORMS


ZERO_TOLERANCE = 1e-10


def canonical_coefficients(values):
    result = np.asarray(values,dtype=float).copy()
    result[np.abs(result) <= ZERO_TOLERANCE] = 0.
    return result


def sparse_features(x, y):
    x = standardized(x)
    y = standardized(np.asarray(y).ravel())
    n, p = x.shape
    cv = KFold(5, shuffle=True, random_state=417)
    matrices, names, extra = [], [], {}
    for shape, z in zip(TRANSFORMS, (x, x*x, np.exp(-.5*x*x), np.sin(2*x))):
        z = np.asfortranarray(standardized(z))
        max_alpha = max(float(np.max(np.abs(z.T@y))/n), 1e-6)
        alphas = max_alpha*np.geomspace(1., .01, 40)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            fitted = LassoCV(alphas=alphas, cv=cv, n_jobs=1,
                             max_iter=5000, tol=1e-4).fit(z, y)
            mse = fitted.mse_path_.mean(axis=1)
            best = int(np.argmin(mse))
            threshold = mse[best]+fitted.mse_path_[best].std(ddof=1)/np.sqrt(5)
            conservative = int(np.flatnonzero(mse <= threshold)[0])
            sparse = Lasso(alpha=float(fitted.alphas_[conservative]),
                           max_iter=10000, tol=1e-5).fit(z, y)
        # X and y are standardized. At alpha_max the exact solution is zero,
        # but floating-point centering can leave ~1e-17 ghost coefficients.
        # Counting those changes residual degrees of freedom by a whole unit.
        fitted.coef_ = canonical_coefficients(fitted.coef_)
        sparse.coef_ = canonical_coefficients(sparse.coef_)
        b = sparse.coef_
        residual = y-sparse.predict(z)
        active = int(np.count_nonzero(b))
        sigma2 = float(np.sum(residual**2)/max(n-active-1, n*.1))
        # Diagonal and regularized conditional corrections complement each
        # other in weakly and strongly correlated designs, respectively.
        score = z.T@residual/n
        marginal_beta = b+score
        gram = z.T@z/n
        values, vectors = np.linalg.eigh(gram)
        inverse = (vectors/(np.maximum(values, 0)+.1))@vectors.T
        normalizer = np.maximum(np.sum(inverse*gram, axis=1), 1e-10)
        inverse /= normalizer[:, None]
        corrected = b+inverse@score
        variance = np.maximum(np.sum((inverse@gram)*inverse, axis=1)*sigma2/n, 1e-15)
        marginal_t = np.abs(marginal_beta)/np.sqrt(max(sigma2/n, 1e-15))
        corrected_t = np.abs(corrected)/np.sqrt(variance)
        matrices.append(np.column_stack([np.abs(fitted.coef_), np.abs(b),
            np.abs(marginal_beta), marginal_t, np.abs(corrected), corrected_t]))
        names += [shape+'_'+suffix for suffix in ('lasso_beta', 'lasso_1se_beta',
            'sparse_marginal_beta', 'sparse_marginal_t', 'sparse_conditional_beta', 'sparse_conditional_t')]
        prefix = 'sparse_'+shape+'_'
        extra.update({prefix+'cv_r2': float(1-mse[best]),
                      prefix+'1se_cv_r2': float(1-mse[conservative]),
                      prefix+'active_fraction': float(np.count_nonzero(fitted.coef_)/p),
                      prefix+'1se_active_fraction': active/p,
                      prefix+'alpha': float(fitted.alpha_),
                      prefix+'1se_alpha': float(sparse.alpha),
                      prefix+'residual_variance': sigma2,
                      prefix+'convergence_warnings': float(sum(isinstance(w.message, ConvergenceWarning) for w in caught))})
        for kind, t in (('marginal', marginal_t), ('conditional', corrected_t)):
            label = prefix+kind+'_t_'
            extra.update({label+f'q{q:g}': float(np.quantile(t,q)) for q in (.5,.9,.99)})
            extra.update({label+f'above{cutoff:g}': float(np.mean(t > cutoff)) for cutoff in (2.,3.,5.)})
            energy = t*t
            extra[label+'effective_fraction'] = float(energy.sum()**2/(np.sum(energy**2)+1e-15)/p)
    matrix = np.column_stack(matrices).astype(np.float32)
    if not np.isfinite(matrix).all() or not all(np.isfinite(v) for v in extra.values()):
        raise ValueError('Nonfinite sparse features.')
    return matrix, names, extra
