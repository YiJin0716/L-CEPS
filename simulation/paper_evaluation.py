"""Fixed small-sample assessment adapted from IPSS (2025), with no result filtering."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,hashlib,json,multiprocessing,pickle,sqlite3,sys,time,zlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
from current_data import DATA,ROOT,standardized,sha,write_json
from release_files import verify_frozen_files, recorded_excluded_seeds
METHODS=['the_model','gb_ipss','lasso_ipss','lasso_cv']
PROTOCOL=DATA/'paper_evaluation_protocol.json'
PLAN=None;MODEL=None;POOLS={}

def pool(name):
    if name not in POOLS:
        if name=='ovarian_rnaseq':POOLS[name]=np.load(DATA/'paper_ipss_ovarian_rnaseq.npy')
        else:
            with np.load(DATA/'paper_evaluation_real_predictors.npz') as f:POOLS[name]=f[name]
    return POOLS[name]

def paper_function(z,rng):
    a=rng.uniform(.5,1.5);b=rng.uniform(-1,1);g=rng.uniform(1,3)
    d1=rng.choice([-1.,1.]);d2=rng.choice([-1.,1.]);tanh=bool(rng.random()<.5)
    value=.5*d1*(1+np.tanh(a*(d2*z-b))) if tanh else d1*np.exp(-g*z*z)
    return value,dict(alpha=a,beta=b,gamma=g,delta1=d1,delta2=d2,tanh=tanh)

def generate(spec,task):
    rng=np.random.default_rng(task['seed']);n,p=task['n'],task['p'];meta={}
    if spec['suite']=='gaussian' or spec['suite']=='simple':
        rho=spec.get('rho',.5);x=rng.normal(size=(n,p))
        for j in range(1,p):x[:,j]=rho*x[:,j-1]+np.sqrt(1-rho*rho)*x[:,j]
        x=standardized(x)
    else:
        source=pool(spec['dataset']);rows=rng.choice(len(source),n,replace=False)
        sampled=source[rows];eligible=np.flatnonzero(sampled.std(0)>1e-10)
        if len(eligible)<p:raise ValueError('Too few varying columns in the sampled real matrix')
        cols=rng.choice(eligible,p,replace=False);x=standardized(sampled[:,cols])
        meta.update(source_rows=rows,source_columns=cols)
    if spec['suite']=='gaussian':k=int(rng.integers(5,16))
    elif spec['suite']=='rnaseq':k=int(rng.integers(10,31))
    else:k=6
    truth=np.sort(rng.choice(p,k,replace=False));z=x[:,truth]
    if spec['suite']=='gaussian':signal=np.exp(-z*z).sum(1)
    elif spec['suite']=='rnaseq':
        count=int(rng.integers(k//2,k+1));order=rng.permutation(k)
        cuts=np.sort(rng.choice(np.arange(1,k),count-1,replace=False))
        groups=np.split(order,cuts);signal=np.zeros(n);parameters=[]
        for indices in groups:
            combined=standardized(z[:,indices].sum(1));part,params=paper_function(combined,rng)
            signal+=part;parameters.append(dict(columns=truth[indices],**params))
        meta['groups']=parameters
    else:
        family=spec['family']
        if family=='linear':signal=z.sum(1)
        elif family=='square':signal=(z*z).sum(1)
        elif family=='sine':signal=np.sin(z).sum(1)
        elif family=='interaction':signal=(z[:,::2]*z[:,1::2]).sum(1)
        else:raise ValueError(family)
    signal=signal-signal.mean()
    if signal.std()<=1e-10:raise ValueError('Degenerate generated signal')
    if spec['response']=='class':
        strength=float(rng.uniform(1,3));prob=expit(strength*signal)
        y=rng.binomial(1,prob).astype(float);meta.update(strength=strength,positive_fraction=float(y.mean()))
    else:
        snr=float(rng.uniform(.5,2)) if spec['suite'] in ('gaussian','rnaseq') else 1.
        if spec['suite'] in ('simple','real'):signal=signal/signal.std()
        noise_sd=float(signal.std()/np.sqrt(snr));y=signal+rng.normal(size=n)*noise_sd;y-=y.mean()
        meta.update(snr=snr,noise_sd=noise_sd)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or y.std()<=1e-10:
        raise ValueError('Nonfinite or constant generated data; no trial is silently replaced')
    return x,y,truth,meta

def collect_seeds(obj,result):
    if isinstance(obj,dict):
        for key,value in obj.items():
            if 'seed' in key and isinstance(value,int):result.add(value)
            else:collect_seeds(value,result)
    elif isinstance(obj,list):
        for value in obj:collect_seeds(value,result)

def freeze():
    if PROTOCOL.exists():raise FileExistsError(PROTOCOL)
    settings=[]
    for suite in ('gaussian','rnaseq'):
        for response in ('reg','class'):
            for ratio in (1.,.8):
                spec=dict(suite=suite,response=response,ratio=ratio)
                if suite=='gaussian':spec['rho']=.5
                else:spec['dataset']='ovarian_rnaseq'
                settings.append(spec)
    for family in ('linear','square','sine','interaction'):
        for ratio in (1.,.8):settings.append(dict(suite='simple',family=family,response='reg',rho=.5,ratio=ratio))
    for dataset in ('isolet','har','coil20'):
        for family in ('linear','square'):
            for ratio in (1.,.8):settings.append(dict(suite='real',dataset=dataset,family=family,response='reg',ratio=ratio))
    excluded={2026111177,2026111188,2026111199}
    for path in list(DATA.glob('*protocol.json'))+[DATA/'current_training.json']:
        collect_seeds(json.loads(path.read_text()),excluded)
    for path in DATA.glob('retained_regression_*_counts.npz'):
        with np.load(path) as f:
            for name in f.files:
                if 'seed' in name:excluded.update(map(int,f[name].ravel()))
    rng=np.random.default_rng(2026111191)
    def new_seed():
        while True:
            value=int(rng.integers(1,2**32-1))
            if value not in excluded:excluded.add(value);return value
    repetitions=100;tasks=[]
    for case,spec in enumerate(settings):
        for rep in range(repetitions):
            p=int(np.rint(np.exp(rng.uniform(np.log(50),np.log(500)))))
            n=p if spec['ratio']==1 else max(50,int(np.floor(.8*p)))
            tasks.append(dict(id=len(tasks),case=case,replicate=rep,n=n,p=p,seed=new_seed(),ipss_seed=new_seed()))
    manifest=json.loads((ROOT/'models/active_model.json').read_text())
    frozen={str(P/name):digest for name,digest in manifest['source_sha256'].items()}
    frozen.update({str(ROOT/'models'/name):digest for name,digest in manifest['weights_and_metadata_sha256'].items()})
    for path in [Path(__file__),P/'simulation/current_data.py',ROOT/'models/active_model.json',DATA/'paper_ipss_ovarian_rnaseq.npy',DATA/'paper_evaluation_real_predictors.npz',DATA/'paper_evaluation_sources.json',DATA/'paper_ipss_source.zip']:
        frozen[str(path)]=sha(path)
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),settings=settings,tasks=tasks,repetitions=repetitions,
        methods=METHODS,targets=np.linspace(0,.5,51).tolist(),frozen_files=frozen,
        dimensions='p=round(exp(U(log(50),log(500)))); n=p or max(50,floor(.8*p)); one frozen draw of dimensions per replicate.',
        selection='All prespecified settings and all repetitions are reported, regardless of model performance. No screening of evaluation settings.',
        adaptation='Small-sample adaptation of IPSS Bioinformatics 2025 Sections 3.2/3.3, not an exact reproduction of its fixed dimensions or published curves.',
        paper_url='https://doi.org/10.1093/bioinformatics/btaf299',paper_preprint='https://arxiv.org/abs/2410.02208v3',
        author_source_commit='d0d3d7705dc64134fdf285901b5a1c02ecec81c3',author_release_commit='b9a4d4e8d7e8a738039ab58ce2b5718680629f5a',
        reconciliation=[
          'Gaussian X standardized as in author code; covariance before standardization has rho=.5. Signal=sum exp(-X_j^2), k uniform integer 5..15.',
          'RNA-seq uses Eq S3.1 and Table S5 random tanh/bump mixture, not the all-bump function hardcoded in both GitHub main and release 1.0.',
          'RNA groups: G uniform integer floor(k/2)..k, as Algorithm S2. Randomly permute true columns, then choose G-1 cut positions uniformly. k uniform integer 10..30.',
          'Center the signal before regression/noise and logistic classification, following the author implementation. Gaussian noise variance=signal variance/SNR, following main text. Algorithm S2 writes a second moment; these coincide after centering.',
          'The source RNA array has 569 rows, consistent with main text; the 596 in Algorithm S2 is not the array dimension.',
          'Real pools: sample rows without replacement, exclude columns constant in these rows, and sample p varying columns without replacement before planting truth.',
          'GB-IPSS uses the retained package default delta 1.25 regression / 1 classification, matching paper Table S1; Lasso-IPSS uses its correlation-based default. Same paths supply the frozen model.',
          'LassoCV uses the frozen linear LassoCV measurement, including numeric 0/1 responses; it has no target FDR.'
        ],
        supplemental='Simple Gaussian predictors use rho=.5; linear, square, sin, or three disjoint pair-products, six randomly chosen relevant columns, centered unit-SD signal and independent N(0,1) noise. Real predictor supplements use the same linear/square response rule.',
        overlap='No refitting. Gaussian/correlation families and several response transforms overlap fitting; new real predictor pools are not fitting data. This is evaluation of power at small n relative to p, not a claim that every mechanism is unseen.',
        software=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__))
    write_json(PROTOCOL,plan)
    write_json(DATA/'paper_evaluation_work.json',dict(status='protocol_frozen',protocol_sha256=sha(PROTOCOL),datasets=len(tasks),settings=len(settings),jobs=[],updated_utc=datetime.now(timezone.utc).isoformat()))
    print('Frozen',len(settings),'settings;',len(tasks),'datasets',flush=True)

def initialize():
    global PLAN,MODEL
    from explore.current import load_model
    torch.set_num_threads(1);PLAN=json.loads(PROTOCOL.read_text());MODEL=load_model()

def one(task):
    from explore.predict import compute_paths
    from explore.observed import observed_records
    from explore.screening import union_qvalues,screened_selection
    from explore.paths import efp_scores,q_from_efp
    t=time.time();spec=PLAN['settings'][task['case']];x,y,truth,meta=generate(spec,task)
    paths=compute_paths(x,y,task['ipss_seed']);path_seconds=time.time()-t
    q=union_qvalues(paths);required=np.flatnonzero(q<=max(PLAN['targets']))
    record=observed_records(x,y,paths)['gb'];matrix=MODEL.matrix(record)
    probability=np.ones(task['p'],dtype=np.float32)
    if len(required):probability[required]=MODEL.estimator.predict_proba(matrix[required])[:,1]
    targets=np.asarray(PLAN['targets']);bits=np.zeros((len(METHODS),len(targets),task['p']),bool)
    for i,ids in enumerate(screened_selection(probability,q,targets)[0]):bits[0,i,ids]=True
    baseline_q={}
    for m,key in [(1,'gb'),(2,'l1')]:
        qq=q_from_efp(efp_scores(paths[key],paths[key]['default_delta']));baseline_q[key]=qq
        bits[m]=qq[None,:]<=targets[:,None]
    column=313+MODEL.original.feature_names.index('linear_lasso_beta');bits[3]=matrix[:,column][None,:]!=0
    true=np.isin(np.arange(task['p']),truth)
    return dict(**task,truth=truth,generation=meta,probability=probability,screening_q=q,baseline_q=baseline_q,
        tp=bits[:,:,true].sum(2),fp=bits[:,:,~true].sum(2),selected_bits=np.packbits(bits,axis=-1),
        input_sha256=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),
        default_deltas={k:float(v['default_delta']) for k,v in paths.items()},
        path_seconds=path_seconds,total_seconds=time.time()-t,
        paths=paths if task['replicate']==0 else None)

def database(shard):return DATA/f'paper_evaluation_{shard}.sqlite'

def run(workers,shard,shards):
    plan=json.loads(PROTOCOL.read_text())
    verify_frozen_files(plan['frozen_files'])
    with sqlite3.connect(database(shard)) as db:
        db.execute('create table if not exists metadata (key text primary key,value text)')
        db.execute('create table if not exists results (id integer primary key,payload blob)')
        previous=db.execute("select value from metadata where key='protocol_sha256'").fetchone()
        if previous:assert previous[0]==sha(PROTOCOL)
        else:db.execute('insert into metadata values (?,?)',('protocol_sha256',sha(PROTOCOL)));db.commit()
        done={r[0] for r in db.execute('select id from results')}
        pending=[t for t in plan['tasks'] if t['id']%shards==shard and t['id'] not in done]
        # Large p first balances the tail of the job, without changing any trial.
        pending.sort(key=lambda t:t['p'],reverse=True)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as executor:
            futures={executor.submit(one,t):t for t in pending}
            for i,future in enumerate(as_completed(futures),1):
                row=future.result();db.execute('insert into results values (?,?)',(row['id'],zlib.compress(pickle.dumps(row,protocol=5),3)))
                if i%10==0:db.commit()
                if i%50==0:print('shard',shard,'completed',len(done)+i,'of',len(done)+len(pending),flush=True)
            db.commit()
    print('Finished shard',shard,flush=True)

def read_rows():
    rows=[]
    for file in sorted(DATA.glob('paper_evaluation_[0-9]*.sqlite')):
        with sqlite3.connect(f'file:{file}?mode=ro',uri=True) as db:
            assert db.execute("select value from metadata where key='protocol_sha256'").fetchone()[0]==sha(PROTOCOL)
            rows.extend(pickle.loads(zlib.decompress(blob)) for blob, in db.execute('select payload from results'))
    return sorted(rows,key=lambda r:r['id'])

def summarize():
    plan=json.loads(PROTOCOL.read_text());rows=read_rows();assert len(rows)==len(plan['tasks'])
    assert len({r['input_sha256'] for r in rows})==len(rows)
    from explore.screening import screened_selection
    for row,task in zip(rows,plan['tasks']):
        assert all(row[k]==v for k,v in task.items())
        bits=np.unpackbits(row['selected_bits'],axis=-1,count=row['p']).astype(bool)
        truth=np.isin(np.arange(row['p']),row['truth'])
        np.testing.assert_array_equal(bits[:,:,truth].sum(2),row['tp']);np.testing.assert_array_equal(bits[:,:,~truth].sum(2),row['fp'])
        for i,ids in enumerate(screened_selection(row['probability'],row['screening_q'],plan['targets'])[0]):
            np.testing.assert_array_equal(ids,np.flatnonzero(bits[0,i]))
    tp=np.stack([r['tp'] for r in rows],axis=1);fp=np.stack([r['fp'] for r in rows],axis=1)
    support=np.array([len(r['truth']) for r in rows]);cases=np.array([r['case'] for r in rows])
    tpr=tp/support[None,:,None];fdr=fp/np.maximum(tp+fp,1);results=[]
    for case,spec in enumerate(plan['settings']):
        mask=cases==case;methods={}
        for m,name in enumerate(METHODS):
            methods[name]={key:array[m,mask].mean(0).tolist() for key,array in [('tpr',tpr),('fdr',fdr)]}
            for key,array in [('tpr',tpr),('fdr',fdr)]:methods[name][key+'_se']=(array[m,mask].std(0,ddof=1)/mask.sum()**.5).tolist()
        results.append(dict(spec=spec,datasets=int(mask.sum()),methods=methods,
            n_range=[int(min(r['n'] for r in rows if r['case']==case)),int(max(r['n'] for r in rows if r['case']==case))],
            p_range=[int(min(r['p'] for r in rows if r['case']==case)),int(max(r['p'] for r in rows if r['case']==case))]))
    count_file=DATA/'paper_evaluation_counts.npz'
    np.savez_compressed(count_file,tp=tp,fp=fp,support=support,cases=cases,targets=plan['targets'],methods=METHODS,
        n=[r['n'] for r in rows],p=[r['p'] for r in rows],seed=[r['seed'] for r in rows],ipss_seed=[r['ipss_seed'] for r in rows])
    write_json(DATA/'paper_evaluation_results.json',dict(protocol_sha256=sha(PROTOCOL),counts_sha256=sha(count_file),
        datasets=len(rows),settings=results,targets=plan['targets'],methods=METHODS,all_settings_reported=True,
        counts_recounted=True,model_selection_reconstructed=True))
    print('Summarized and independently recounted',len(rows),'datasets',flush=True)

def verify():
    from explore.current import select_features
    initialize();rows=read_rows();checked=[]
    verify_frozen_files(PLAN['frozen_files'])
    for row in rows:
        if row['replicate']!=0:continue
        x,y,truth,meta=generate(PLAN['settings'][row['case']],row)
        assert hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()==row['input_sha256']
        result=select_features(x,y,PLAN['targets'],model=MODEL,paths=row['paths'],random_seed=row['ipss_seed'])
        bits=np.unpackbits(row['selected_bits'][0],axis=-1,count=row['p'])
        for i,ids in enumerate(result['selected']):np.testing.assert_array_equal(ids,np.flatnonzero(bits[i]))
        checked.append(row['case'])
    assert len(checked)==len(PLAN['settings'])
    write_json(DATA/'paper_evaluation_verification.json',dict(protocol_sha256=sha(PROTOCOL),
        results_sha256=sha(DATA/'paper_evaluation_results.json'),public_api_cases=checked,
        model_files_unchanged=True,all_dimensions_follow_protocol=all(50<=r['n']<=r['p']<=500 for r in rows)))
    print('Verified',len(checked),'raw-data public API cases',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','summarize','verify'])
    parser.add_argument('--workers',type=int,default=24);parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run(args.workers,args.shard,args.shards)
    elif args.action=='summarize':summarize()
    else:verify()
