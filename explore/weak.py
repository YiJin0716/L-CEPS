"""Additional observable conditional evidence; leaves the frozen v1 schema intact."""

import numpy as np

from .features import standardized


def weak_conditional_features(x, y):
    """Reduce ridge bias when the design leaves ample residual degrees of freedom.

    Studentization is an input to the learned classifier, not a p-value. In
    underdetermined designs the original regularization floor is retained.
    """
    x = standardized(x)
    y = standardized(np.asarray(y).ravel())
    n, p = x.shape
    relative = 1e-4 if p <= .75 * (n - 1) else .01
    columns, names = [], []
    extra = {'weak_ols_available': float(relative == 1e-4)}
    for transform, z in [('linear', x), ('square', x*x),
                         ('bump', np.exp(-.5*x*x)), ('sin', np.sin(2*x))]:
        z = standardized(z)
        values, vectors = np.linalg.eigh(z.T @ z)
        values = np.maximum(values, 0)
        inverse = 1 / (values + max(n, p) * relative)
        beta = vectors @ (inverse * (vectors.T @ (z.T @ y)))
        variance = (vectors**2) @ (values * inverse**2)
        residual = y - z @ beta
        dof = np.sum(values * inverse)
        scale = np.sqrt(np.sum(residual**2) / max(n - 1 - dof, 1))
        studentized = np.abs(beta) / (scale * np.sqrt(np.maximum(variance, 1e-12)) + 1e-12)
        columns.extend([np.abs(beta), studentized])
        names.extend([transform+'_weak_beta', transform+'_weak_studentized'])
        extra[transform+'_weak_residual_r2'] = float(1 - np.mean(residual**2))
    matrix = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite weak conditional features.')
    return matrix, names, extra
