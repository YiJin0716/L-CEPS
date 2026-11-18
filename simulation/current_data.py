"""Paths, common utilities, and training generation for the final model."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import expit,ndtr
PIPELINE=Path(__file__).resolve().parents[1]
ROOT=PIPELINE
DATA=ROOT/'data'

def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def standardized(x):
    x = np.asarray(x, dtype=float)
    return (x-x.mean(0)) / np.maximum(x.std(0), 1e-10)


def broad_generate(spec, seed):
    rng=np.random.default_rng(seed)
    n,p,k=spec['n'],spec['p'],spec['support']
    truth=np.sort(rng.choice(p,k,replace=False))
    coefficients=rng.choice([-1.,1.],k)*rng.uniform(.5,1.5,k)
    for attempt in range(100):
        x=rng.normal(size=(n,p));cov=spec['covariance'];rho=spec['rho']
        if cov=='toeplitz':
            for j in range(1,p):x[:,j]=rho*x[:,j-1]+np.sqrt(1-rho*rho)*x[:,j]
        elif cov=='block':
            shared=rng.normal(size=(n,(p+spec['block_size']-1)//spec['block_size']))
            x=np.sqrt(rho)*shared[:,np.arange(p)//spec['block_size']]+np.sqrt(1-rho)*x
        elif cov=='factor':
            loadings=rng.normal(size=(spec['factors'],p))
            x=(rng.normal(size=(n,spec['factors']))@loadings+x)/np.sqrt(1+(loadings*loadings).sum(0))
        dist=spec['distribution']
        if dist=='t':x/=np.sqrt(rng.chisquare(spec['df'],(n,1))/spec['df'])
        elif dist=='lognormal':x=np.exp(.7*x)
        elif dist=='binary':x=(ndtr(x)>.5).astype(float)
        elif dist=='ordinal':x=np.floor(5*ndtr(x))
        elif dist=='mixture':x+=rng.choice([-1.,1.],(n,1))*rng.normal(size=(1,p))
        elif dist=='uniform':x=ndtr(x)
        elif dist=='laplace':
            u=np.clip(ndtr(x),1e-10,1-1e-10)-.5
            x=-np.sign(u)*np.log1p(-2*np.abs(u))
        elif dist=='contaminated':x*=np.where(rng.random((n,p))<.02,8.,1.)
        if cov=='proxies':
            sources=truth if k else rng.choice(p,p//2,replace=False)
            destinations=rng.choice(np.setdiff1d(np.arange(p),sources),len(sources),replace=False)
            r=spec['proxy_correlation']
            x[:,destinations]=r*standardized(x[:,sources])+np.sqrt(1-r*r)*rng.normal(size=(n,len(sources)))
        z=standardized(x[:,truth]);family=spec['family']
        if not k:signal=np.zeros(n);break
        if family in ('interactions','interaction_main','xor','max_pair'):
            if family=='xor':components=np.sign(z[:,::2])*np.sign(z[:,1::2])
            elif family=='max_pair':components=np.maximum(z[:,::2],z[:,1::2])
            else:components=z[:,::2]*z[:,1::2]
            if np.any(components.std(0)<1e-8):continue
            signal=standardized(components)@coefficients[::2]
            if family=='interaction_main':signal+=.5*(z@coefficients)
        else:
            transforms={'square':lambda u:u*u,'bump':lambda u:np.exp(-.5*u*u),
                'sin':lambda u:np.sin(2*u),'tanh':np.tanh,'threshold':lambda u:(u>.3).astype(float),
                'cosine':lambda u:np.cos(3*u),'triangle':lambda u:np.abs((u+1)%4-2),
                'rational':lambda u:u/(1+u*u)}
            if family=='mixed':
                functions=[lambda u:u,transforms['square'],transforms['sin'],transforms['threshold'],np.tanh]
                components=np.column_stack([functions[j%len(functions)](z[:,j]) for j in range(k)])
            else:components=transforms.get(family,lambda u:u)(z)
            if np.any(components.std(0)<1e-8):continue
            signal=standardized(components)@coefficients
        signal=standardized(signal)
        break
    else:raise RuntimeError('Could not generate nonconstant signal components.')
    if family=='logistic':y=rng.binomial(1,expit(2*signal)).astype(float)
    elif family=='poisson':y=rng.poisson(np.exp(.3+np.clip(.7*signal,-4,4))).astype(float)
    elif family=='xor':y=rng.binomial(1,expit(2*signal)).astype(float)
    else:
        noise=rng.normal(size=n)
        if spec['noise']=='t':noise=rng.standard_t(3,n)
        elif spec['noise']=='outliers':noise*=np.where(rng.random(n)<.05,10.,1.)
        elif spec['noise']=='heteroskedastic' and k:noise*=.2+np.abs(z[:,0])
        noise/=max(noise.std(),1e-10)
        y=signal+noise/np.sqrt(spec['snr']) if k else noise
    if not np.isfinite(x).all() or not np.isfinite(y).all() or y.std()<1e-10:
        raise ValueError('Degenerate generated dataset.')
    return x,y,truth


def generate(spec, seed):
    if spec['suite'] == 'standard':
        if __package__:
            from .training_generators import generate_dataset
        else:
            from training_generators import generate_dataset
        return generate_dataset(np.random.default_rng(seed), spec['parameters'], (.25,1.))
    if spec['suite'] == 'broad':
        return broad_generate(spec, seed)
    raise ValueError('Only the final model training generators are included.')
