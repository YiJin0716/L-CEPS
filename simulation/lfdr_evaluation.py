"""Paired local-FDR comparison on all 4,000 already frozen assessment datasets."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import atexit
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import sqlite3
import subprocess
import sys
import tempfile
import time
import zlib
import numpy as np
from scipy.stats import norm, t as student_t
from scipy.optimize import minimize_scalar
from current_data import ROOT, DATA, sha, write_json
import paper_evaluation as paper
import common_benchmarks as common

PROTOCOL = DATA/'lfdr_comparison_protocol.json'
DATABASE = DATA/'lfdr_comparison.sqlite'
COUNTS = DATA/'lfdr_comparison_counts.npz'
RESULTS = DATA/'lfdr_comparison_results.json'
METHODS = ['the_model', 'adaptz', 'cumulative_ell', 'lasso_mfdr_pointwise', 'lasso_mfdr_cumulative']
RLIB = os.environ.get('IPSS_LFDR_RLIB', '/usr/xtmp/yj255/ipss-lfdr/Rlib')
RSCRIPT = Path(__file__).with_name('lfdr_methods.R')
PLAN = None
RWORKER = None
SCRATCH = None
REFERENCE = None


def source_plans():
    return {'paper': json.loads(paper.PROTOCOL.read_text()),
            'common': json.loads(common.PROTOCOL.read_text())}


def reference_rows():
    return {'paper': {r['id']: r for r in paper.read_rows()},
            'common': {r['id']: r for r in common.read_rows()}}


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError(PROTOCOL)
    plans = source_plans(); rows = reference_rows(); tasks = []; settings = []
    for origin, source in plans.items():
        offset = len(settings)
        settings.extend(dict(origin=origin, source_case=i, **s) for i,s in enumerate(source['settings']))
        assert len(rows[origin]) == len(source['tasks'])
        for task in source['tasks']:
            row = rows[origin][task['id']]
            assert all(row[k]==v for k,v in task.items())
            tasks.append(dict(id=len(tasks), origin=origin, source_id=task['id'],
                case=offset+task['case'], source_case=task['case'], replicate=task['replicate'],
                n=task['n'], p=task['p'], seed=task['seed'], ipss_seed=task['ipss_seed'],
                r_seed=int(task['seed'] % (2**31-2) + 1), input_sha256=row['input_sha256']))
    assert len(tasks)==4000 and len(settings)==40
    packages = subprocess.check_output(['Rscript','--vanilla','-e',
        'cat(jsonlite::toJSON(list(R=R.version.string,ncvreg=as.character(packageVersion("ncvreg")),'
        'ashr=as.character(packageVersion("ashr")),jsonlite=as.character(packageVersion("jsonlite"))),auto_unbox=TRUE))'],
        env=dict(os.environ,R_LIBS_USER=RLIB), text=True)
    frozen_paths = [Path(__file__),RSCRIPT,Path(paper.__file__),Path(common.__file__),
                    ROOT/'simulation/current_data.py',paper.PROTOCOL,common.PROTOCOL,
                    DATA/'paper_evaluation_counts.npz',DATA/'common_benchmark_counts.npz',
                    ROOT/'models/active_model.json',ROOT/'models/retained_mlp_hard_rows.pt',ROOT/'models/final/prior.ubj']
    frozen_paths += list(DATA.glob('paper_evaluation_[0-9]*.sqlite'))+list(DATA.glob('common_benchmark_[0-9]*.sqlite'))
    frozen_paths += [DATA/'paper_ipss_ovarian_rnaseq.npy', DATA/'paper_evaluation_real_predictors.npz', DATA/'common_benchmark_predictors.npz']
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), tasks=tasks,settings=settings,
        methods=METHODS, targets=plans['paper']['targets'], software=json.loads(packages),
        frozen_files={str(p.relative_to(ROOT)):sha(p) for p in frozen_paths},
        reuse='All 4,000 existing X/y/truth draws, identified by exact input hashes. Existing model selections are reused unchanged.',
        marginal_statistic='Pearson correlation t = r sqrt((n-2)/(1-r^2)); z=Phi^{-1}(T_{n-2}(t)). For binary y this equals the pooled two-sample t statistic for X_j across response groups.',
        adaptz='Sun-Cai equation (10), N(0,1) null, Jin-Cai equation (21) null proportion with gamma=.49, 101 frequency points and 101-point trapezoidal integration; pi0 clipped to [0,1]. Gaussian kernel density with R bw.ucv; R density uses 512 points. This is a documented implementation of the paper, not recovered original author code.',
        cumulative_ell='Quasi-Cauchy convolution g(z)=(1-exp(-z^2/2))/(sqrt(2*pi)*z^2), continuous value at zero. Marginal likelihood maximized on [1/p,1]; cumulative posterior-null mean <= target.',
        lasso_mfdr='Author ncvreg package: penalty=lasso, 10-fold CV, lambda.min, gaussian or binomial family, returnX=TRUE. local_mfdr(method=ashr); package defaults otherwise. Evaluate all returned variable scores.',
        lasso_mfdr_rules={'pointwise':'Select score <= alpha, the featurewise threshold used in the paper.',
            'cumulative':'Select the largest complete-tie prefix with mean score <= alpha; explicitly an additional cumulative-rule version, not the paper simulation threshold.'},
        selection='All settings and replicates; no truth or evaluation outcomes supplied to methods; no baseline tuning against test truth; errors stop completion rather than silently removing or replacing tasks.',
        estimand='Every method scored against the same planted generating support, including correlated inactive columns as false selections. Marginal-association nulls in literature are different; correlated designs and nonlinear signals need not satisfy original theory.',
        sources={'sun_cai':'https://doi.org/10.1198/016214507000000545',
            'jin_cai':'https://doi.org/10.1198/016214507000000167',
            'cumulative_ell':'https://arxiv.org/abs/2102.00929v3',
            'miller_breheny':'https://doi.org/10.1002/sim.9678',
            'ncvreg':'https://pbreheny.github.io/ncvreg/reference/local_mfdr.html'},
        summary='Dataset-averaged FDP and TPR; equal weight to all 40 settings. Paired standard errors across the same datasets. FDR and TPR always reported together.')
    write_json(PROTOCOL, plan)
    print('Frozen all 4,000 paired datasets and four comparator curves.',flush=True)


def check_frozen():
    plan=json.loads(PROTOCOL.read_text())
    for relative,digest in plan['frozen_files'].items():
        assert sha(ROOT/relative)==digest,relative
    return plan


def prefix_bits(scores, targets):
    scores=np.asarray(scores,dtype=float); order=np.argsort(scores,kind='stable')
    ordered=scores[order];ends=np.flatnonzero(np.r_[ordered[1:]!=ordered[:-1],True])+1
    means=np.cumsum(ordered)[ends-1]/ends
    out=np.zeros((len(targets),len(scores)),bool)
    for i,target in enumerate(targets):
        valid=ends[means<=target]
        if len(valid):out[i,order[:valid[-1]]]=True
    return out


def independent_ell(z):
    z=np.asarray(z,dtype=float);square=z*z
    ratio=np.divide(-np.expm1(-square/2),square,out=np.full_like(z,.5),where=square>1e-12)
    g=ratio/np.sqrt(2*np.pi);phi=norm.pdf(z)
    objective=lambda w:-np.log((1-w)*phi+w*g).sum()
    fit=minimize_scalar(objective,bounds=(1/len(z),1),method='bounded',options={'xatol':1e-12})
    weight=min([1/len(z),1,fit.x],key=objective)
    return (1-weight)*phi/((1-weight)*phi+weight*g),weight


def close_worker():
    global RWORKER,SCRATCH
    if RWORKER is not None:
        if RWORKER.poll() is None:
            RWORKER.stdin.close();RWORKER.wait(timeout=30)
        RWORKER=None
    if SCRATCH is not None:SCRATCH.cleanup();SCRATCH=None


def initialize():
    global PLAN,REFERENCE,RWORKER,SCRATCH
    PLAN=json.loads(PROTOCOL.read_text());REFERENCE=reference_rows()
    SCRATCH=tempfile.TemporaryDirectory(prefix='ipss-lfdr-',dir='/tmp')
    RWORKER=subprocess.Popen(['Rscript','--vanilla',str(RSCRIPT)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,text=True,bufsize=1,env=dict(os.environ,R_LIBS_USER=RLIB))
    atexit.register(close_worker)


def score_raw(x,y,response,seed):
    assert RWORKER is not None
    path=Path(SCRATCH.name)/'input.bin'
    with path.open('wb') as f:
        np.asarray(x,dtype='<f8').tofile(f);np.asarray(y,dtype='<f8').tofile(f)
    request=dict(input=str(path),n=len(y),p=x.shape[1],response=response,seed=seed)
    RWORKER.stdin.write(json.dumps(request)+'\n');RWORKER.stdin.flush()
    line=RWORKER.stdout.readline()
    if not line:raise RuntimeError('R worker stopped: '+RWORKER.stderr.read())
    value=json.loads(line)
    if not value['ok']:raise RuntimeError(value)
    if isinstance(value['warnings'],str):value['warnings']=[value['warnings']]
    return value


def one(task):
    started=time.time();spec=PLAN['settings'][task['case']]
    engine=paper if task['origin']=='paper' else common
    x,y,truth,meta=engine.generate(spec,task)
    assert hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()==task['input_sha256'],task['id']
    reference=REFERENCE[task['origin']][task['source_id']]
    np.testing.assert_array_equal(truth,reference['truth'])
    raw=score_raw(x,y,spec['response'],task['r_seed'])
    scores=np.asarray(raw['scores']);targets=np.asarray(PLAN['targets'])
    assert scores.shape==(task['p'],3)
    assert np.isfinite(scores).all() and np.all((0<=scores)&(scores<=1))
    xc=x-x.mean(0);yc=y-y.mean();r=xc.T@yc/np.sqrt(np.sum(xc*xc,axis=0)*np.sum(yc*yc))
    r=np.clip(r,-1+1e-15,1-1e-15);t=r*np.sqrt((len(y)-2)/(1-r*r))
    expected=np.sign(t)*norm.isf(student_t.sf(np.abs(t),len(y)-2))
    np.testing.assert_allclose(raw['marginal_z'],expected,rtol=1e-10,atol=1e-10)
    ell,weight=independent_ell(expected)
    np.testing.assert_allclose(scores[:,1],ell,rtol=1e-5,atol=1e-7)
    old=np.unpackbits(reference['selected_bits'][0],axis=-1,count=task['p']).astype(bool)
    bits=np.stack([old,prefix_bits(scores[:,0],targets),prefix_bits(scores[:,1],targets),
                   scores[:,2][None,:]<=targets[:,None],prefix_bits(scores[:,2],targets)])
    mask=np.isin(np.arange(task['p']),truth)
    return dict(**task,truth=truth,selected_bits=np.packbits(bits,axis=-1),tp=bits[:,:,mask].sum(2),
                fp=bits[:,:,~mask].sum(2),scores=scores,diagnostics=raw,total_seconds=time.time()-started)


def run(workers):
    plan=check_frozen()
    with sqlite3.connect(DATABASE) as db:
        db.execute('create table if not exists metadata (key text primary key,value text)')
        db.execute('create table if not exists results (id integer primary key,payload blob)')
        prior=db.execute("select value from metadata where key='protocol_sha256'").fetchone()
        if prior:assert prior[0]==sha(PROTOCOL)
        else:db.execute('insert into metadata values (?,?)',('protocol_sha256',sha(PROTOCOL)));db.commit()
        done={r[0] for r in db.execute('select id from results')}
        pending=sorted([t for t in plan['tasks'] if t['id'] not in done],key=lambda t:-t['p'])
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as executor:
            futures={executor.submit(one,t):t for t in pending}
            try:
                for i,future in enumerate(as_completed(futures),1):
                    row=future.result()
                    db.execute('insert into results values (?,?)',(row['id'],zlib.compress(pickle.dumps(row,protocol=5),3)))
                    if i%10==0:db.commit()
                    if i%50==0:
                        print('Completed',len(done)+i,'of',len(plan['tasks']),flush=True)
                        write_json(DATA/'lfdr_comparison_work.json',dict(status='running',completed=len(done)+i,
                            datasets=len(plan['tasks']),protocol_sha256=sha(PROTOCOL),updated_utc=datetime.now(timezone.utc).isoformat()))
            finally:db.commit()
    print('Completed all local-FDR comparator evaluations.',flush=True)


def read_rows():
    with sqlite3.connect(f'file:{DATABASE}?mode=ro',uri=True) as db:
        assert db.execute("select value from metadata where key='protocol_sha256'").fetchone()[0]==sha(PROTOCOL)
        return [pickle.loads(zlib.decompress(row[0])) for row in db.execute('select payload from results order by id')]


def summarize():
    plan=check_frozen();rows=read_rows()
    assert [r['id'] for r in rows]==list(range(4000))
    support=np.array([len(r['truth']) for r in rows]);cases=np.array([r['case'] for r in rows])
    tp=np.stack([r['tp'] for r in rows],axis=1);fp=np.stack([r['fp'] for r in rows],axis=1)
    np.savez_compressed(COUNTS,tp=tp,fp=fp,support=support,cases=cases,targets=plan['targets'],methods=METHODS,
        n=[r['n'] for r in rows],p=[r['p'] for r in rows],seed=[r['seed'] for r in rows])
    tpr=tp/support[None,:,None];fdr=fp/np.maximum(tp+fp,1)
    def standard_error(values, mask):
        selected_cases=cases[mask]
        variance=sum((selected_cases==case).sum()*values[selected_cases==case].var(0,ddof=1)
                     for case in np.unique(selected_cases))
        return np.sqrt(variance)/mask.sum()
    def metrics(mask):
        result={}
        for m,name in enumerate(METHODS):
            result[name]={metric:a[m,mask].mean(0).tolist() for metric,a in [('tpr',tpr),('fdr',fdr)]}
            for metric,a in [('tpr',tpr),('fdr',fdr)]:
                result[name][metric+'_se']=standard_error(a[m,mask],mask).tolist()
                diff=a[0,mask]-a[m,mask]
                result[name]['model_minus_'+metric]=diff.mean(0).tolist()
                result[name]['model_minus_'+metric+'_se']=standard_error(diff,mask).tolist()
        return result
    settings=[dict(spec=s,datasets=int((cases==i).sum()),methods=metrics(cases==i)) for i,s in enumerate(plan['settings'])]
    warnings=Counter(w for r in rows for w in r['diagnostics']['warnings'])
    write_json(RESULTS,dict(protocol_sha256=sha(PROTOCOL),counts_sha256=sha(COUNTS),datasets=4000,methods=METHODS,
        targets=plan['targets'],all_settings_reported=True,settings=settings,overall=metrics(np.ones(4000,bool)),
        warning_counts=dict(warnings),total_comparator_seconds=sum(r['diagnostics']['seconds'] for r in rows)))
    write_json(DATA/'lfdr_comparison_work.json',dict(status='evaluated',completed=4000,datasets=4000,
        protocol_sha256=sha(PROTOCOL),updated_utc=datetime.now(timezone.utc).isoformat()))
    print('Summarized 40 settings and 4,000 paired datasets.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','summarize'])
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run(args.workers)
    else:summarize()
