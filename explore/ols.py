"""Unregularized conditional evidence and observable signal-density summaries.

Classical t probabilities are exact only for a fixed, correctly specified
homoskedastic Gaussian linear model. Transform choice and general simulation
mixtures do not meet that condition automatically. These values are inputs
to the learned variable ranker.
"""

import numpy as np
from scipy.stats import t as student_t

from .features import standardized

TRANSFORMS = ('linear', 'square', 'bump', 'sin')


def ols_features(x, y):
    x = standardized(x)
    y = standardized(np.asarray(y).ravel())
    n, p = x.shape
    available = n-p-1 >= max(15, .1*n)
    names = [name+'_'+suffix for name in TRANSFORMS
             for suffix in ('ols_beta', 'ols_t', 'ols_hc3_t', 'ols_logp')]
    matrix = np.zeros((p, len(names)), dtype=np.float32)
    extra = {'ols_available': float(available), 'ols_residual_df': float(max(n-p-1,0))}
    for index, (name,z) in enumerate(zip(TRANSFORMS,[x,x*x,np.exp(-.5*x*x),np.sin(2*x)])):
        z = standardized(z)
        if available:
            values, vectors = np.linalg.eigh(z.T@z)
            full_rank = values[0] > max(values[-1],1)*1e-10
            if full_rank:
                inverse = (vectors/values)@vectors.T
                influence = inverse@z.T
                beta = influence@y
                residual = y-z@beta
                df = n-p-1
                variance = np.maximum(np.diag(inverse), 1e-20)*np.sum(residual**2)/df
                t = np.abs(beta)/np.sqrt(np.maximum(variance,1e-20))
                leverage = np.sum(z*influence.T,axis=1)+1/n
                adjusted_residual = residual/np.maximum(1-leverage,1e-8)
                robust_variance = (influence**2)@(adjusted_residual**2)
                robust_t = np.abs(beta)/np.sqrt(np.maximum(robust_variance,1e-20))
                pvalues = np.maximum(2*student_t.sf(t,df),1e-30)
                matrix[:,4*index:4*(index+1)] = np.column_stack([np.abs(beta),t,robust_t,-np.log(pvalues)])
            else:
                extra['ols_available'] = 0.
        extra[name+'_ols_available'] = float(available and (full_rank if available else False))
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite OLS statistics.')
    return matrix, names, extra


def signal_density_features(matrix, names):
    """Summarize concentration and strength without observing true support size."""
    extra = {}
    for name in TRANSFORMS:
        for suffix in ('studentized_0.1','weak_studentized','ols_t'):
            values = np.asarray(matrix[:,names.index(name+'_'+suffix)],dtype=float)
            label = 'density_'+name+'_'+suffix
            for q in (.5,.9,.99):
                extra[label+f'_q{q:g}'] = float(np.quantile(values,q))
            extra[label+'_max'] = float(values.max())
            for threshold in (2.,3.,5.):
                extra[label+f'_above{threshold:g}'] = float(np.mean(values>threshold))
            energy = values**2
            extra[label+'_effective_fraction'] = float(energy.sum()**2/(np.sum(energy**2)+1e-15)/len(values))
        values = np.asarray(matrix[:,names.index(name+'_weak_beta')],dtype=float)
        energy = values**2
        extra['density_'+name+'_beta_top5_energy'] = float(np.sort(energy)[-5:].sum()/(energy.sum()+1e-15))
        extra['density_'+name+'_beta_effective_fraction'] = float(energy.sum()**2/(np.sum(energy**2)+1e-15)/len(values))
    return extra

