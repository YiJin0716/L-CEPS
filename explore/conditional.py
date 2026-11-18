"""Feature-level handcrafted evidence, including conditional ridge associations.

These are predictive features, not p-values or asserted confidence bounds.
No true support, covariance label, SNR, or response-family label is accepted.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold,cross_val_predict

from .features import standardized
from .paths import efp_scores,q_from_efp


def conditional_features(x,y):
    """Compute all feature-level X/y evidence once for both base selectors."""
    x=standardized(x);y=standardized(np.asarray(y).ravel())
    n,p=x.shape
    corr=np.abs(x.T@x/n);np.fill_diagonal(corr,0)
    names=['neighbor_max','neighbor_mean','neighbor_energy','neighbors_above_05','neighbors_above_08']
    columns=[corr.max(0),corr.mean(0),(corr**2).sum(0),np.mean(corr>.5,0),np.mean(corr>.8,0)]
    for transform,z in [('linear',x),('square',x*x),('bump',np.exp(-.5*x*x)),('sin',np.sin(2*x))]:
        z=standardized(z)
        gram=z.T@z
        values,vectors=np.linalg.eigh(gram)
        values=np.maximum(values,0)
        rhs=vectors.T@(z.T@y)
        association=np.abs(z.T@y/n)
        names.append(transform+'_marginal');columns.append(association)
        for relative in (.01,.1,1.):
            penalty=max(n,p)*relative
            inverse=1/(values+penalty)
            beta=vectors@(inverse*rhs)
            # Ridge coefficient variance factor; no claim of unbiased inference.
            covariance_diagonal=(vectors**2)@(values*inverse**2)
            residual=y-z@beta
            dof=np.sum(values*inverse)
            residual_scale=np.sqrt(np.sum(residual**2)/max(n-dof,1))
            studentized=np.abs(beta)/(residual_scale*np.sqrt(np.maximum(covariance_diagonal,1e-12))+1e-12)
            names.extend([f'{transform}_beta_{relative}',f'{transform}_studentized_{relative}'])
            columns.extend([np.abs(beta),studentized])
    sine=standardized(np.sin(2*x))
    prediction=cross_val_predict(Ridge(alpha=max(n,p)*.1),sine,y,
                                 cv=KFold(5,shuffle=True,random_state=417))
    extra={'sin_cv_r2':float(1-np.mean((y-prediction)**2))}
    return np.column_stack(columns).astype(np.float32),names,extra


def ranking_features(base,base_names,paths):
    """Append each feature's stability and agreement across delta values."""
    p=paths['p_full']
    columns=[base];names=list(base_names)
    for delta in (0.,1.,2.,4.):
        efp=efp_scores(paths,delta)
        q=q_from_efp(efp)
        rank=np.searchsorted(np.sort(efp),efp,side='right')/p
        columns.append(np.column_stack([np.log(np.maximum(efp,1e-12)),q,rank]))
        names.extend([f'log_efp_{delta}',f'q_{delta}',f'rank_{delta}'])
    summary=np.zeros((p,5))
    if not paths['null']:
        probabilities=paths['stability_paths']
        values=np.column_stack([probabilities.max(0),probabilities.mean(0),
                                np.mean(probabilities>.5,0),np.mean(probabilities>.8,0),
                                np.mean(np.maximum(2*probabilities-1,0)**2,0)])
        summary[paths['preselect_indices']]=values
    columns.append(summary)
    names.extend(['stability_max','stability_mean','stability_above_05','stability_above_08','stability_h2_mean'])
    matrix=np.column_stack(columns).astype(np.float32)
    if not np.isfinite(matrix).all():raise ValueError('Nonfinite ranking features.')
    return matrix,names
