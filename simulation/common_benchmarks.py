"""Prespecified additional benchmarks, evaluated with the frozen current model."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import multiprocessing
import pickle
import sqlite3
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.datasets import make_friedman1
import torch
import paper_evaluation as engine
from current_data import DATA, ROOT, standardized, sha, write_json

from release_files import verify_frozen_files, recorded_excluded_seeds

PROTOCOL = DATA/'common_benchmark_protocol.json'
COUNTS = DATA/'common_benchmark_counts.npz'
RESULTS = DATA/'common_benchmark_results.json'
POOLS = {}
METHODS = engine.METHODS


def pool(name):
    if name not in POOLS:
        with np.load(DATA/'common_benchmark_predictors.npz') as z:
            POOLS[name] = z[name]
    return POOLS[name]


def generate(spec, task):
    n, p = task['n'], task['p']
    assert 50 <= n <= p <= 500
    if spec['family'] == 'friedman1':
        state = np.random.RandomState(task['seed'])
        x, y = make_friedman1(n_samples=n, n_features=p, noise=1., random_state=state)
        permutation = state.permutation(p)
        x = x[:, permutation].copy()
        truth = np.flatnonzero(permutation < 5)
        meta = dict(column_permutation=permutation, noise_sd=1.)
    else:
        rng = np.random.default_rng(task['seed'])
        if spec['family'] == 'sparse_linear':
            x = rng.normal(size=(n, p)); rho = .25
            for j in range(1, p):
                x[:, j] = rho*x[:, j-1]+np.sqrt(1-rho*rho)*x[:, j]
            truth = np.sort(rng.choice(p, 10, replace=False))
            coefficient = 4.5/np.sqrt(n)
            y = coefficient*x[:, truth].sum(1)+rng.normal(size=n)
            meta = dict(coefficient=coefficient, noise_sd=1., rho=rho)
        else:
            source = pool(spec['dataset'])
            rows = rng.choice(len(source), n, replace=False)
            sampled = source[rows]
            eligible = np.flatnonzero(sampled.std(0) > 1e-10)
            if len(eligible) < p:
                raise ValueError('Too few varying real columns; do not replace this draw silently.')
            columns = rng.choice(eligible, p, replace=False)
            x = standardized(sampled[:, columns])
            truth = np.sort(rng.choice(p, 6, replace=False))
            z = x[:, truth]
            if spec['family'] == 'linear':
                signal = z.sum(1)
            elif spec['family'] == 'square':
                signal = (z*z).sum(1)
            else:
                raise ValueError(spec['family'])
            if signal.std() <= 1e-10:
                raise ValueError('Constant signal; do not replace this draw silently.')
            signal = standardized(signal)
            y = signal+rng.normal(size=n)
            y -= y.mean()
            meta = dict(source_rows=rows, source_columns=columns, noise_sd=1., snr=1.)
    assert x.shape == (n, p) and y.shape == (n,)
    assert np.isfinite(x).all() and np.isfinite(y).all() and y.std() > 1e-10
    return x, y, truth, meta


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError(PROTOCOL)
    old = json.loads(engine.PROTOCOL.read_text())
    settings = []
    for family in ('sparse_linear', 'friedman1'):
        for ratio in (1., .8):
            settings.append(dict(suite='synthetic', family=family, response='reg', ratio=ratio))
    for dataset in ('mnist', 'basehock'):
        for family in ('linear', 'square'):
            for ratio in (1., .8):
                settings.append(dict(suite='real', dataset=dataset, family=family, response='reg', ratio=ratio))
    excluded = {2026111219, 2026111220}
    for path in list(DATA.glob('*protocol.json'))+[DATA/'current_training.json']:
        engine.collect_seeds(json.loads(path.read_text()), excluded)
    for path in DATA.glob('retained_regression_*_counts.npz'):
        with np.load(path) as z:
            for key in z.files:
                if 'seed' in key:
                    excluded.update(map(int, z[key].ravel()))
    rng = np.random.default_rng(2026111221)
    def new_seed():
        while True:
            seed = int(rng.integers(1, 2**32-1))
            if seed not in excluded:
                excluded.add(seed)
                return seed
    tasks = []
    for case, spec in enumerate(settings):
        for rep in range(100):
            p = int(np.rint(np.exp(rng.uniform(np.log(50), np.log(500)))))
            n = p if spec['ratio'] == 1 else max(50, int(np.floor(.8*p)))
            tasks.append(dict(id=len(tasks), case=case, replicate=rep, n=n, p=p,
                              seed=new_seed(), ipss_seed=new_seed()))
    frozen = dict(old['frozen_files'])
    for path in [Path(__file__), engine.PROTOCOL, COUNTS.with_name('common_benchmark_predictors.npz'),
                 DATA/'common_benchmark_sources.json']:
        frozen[str(path.resolve())] = sha(path)
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), settings=settings, tasks=tasks,
        repetitions=100, methods=METHODS, targets=old['targets'], frozen_files=frozen,
        dimensions=old['dimensions'], selection=old['selection'],
        fitting='No refitting, no changes to the current model or comparators.',
        sparse_linear=dict(rho=.25, support=10, coefficient='4.5/sqrt(n)', noise='independent N(0,1)',
            source='https://web.stanford.edu/group/candes/knockoffs/software/knockoffs/r/knockoff.html',
            adaptation='Correlated Gaussian sparse linear family from the authors model-X vignette; retain rho=.25, amplitude=4.5, noise SD=1, and positive coefficients. Set k=10 instead of 60 for p as small as 50. Use the shared random n,p rule instead of the vignette fixed dimensions. X and y are not sample-standardized or centered by this generator.'),
        friedman1=dict(support=5, noise_sd=1.,
            source='https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html',
            formula='10 sin(pi X1 X2)+20(X3-.5)^2+10 X4+5 X5+N(0,1); independent Xj~Uniform(0,1)',
            adaptation='Use sklearn.datasets.make_friedman1 directly, noise=1. Randomly permute columns after generating y; record the original five signal columns. No standardization before generating the response.'),
        real='MNIST: official 60,000 training images flattened to 784 pixels. BASEHOCK: scikit-feature 1,993 by 4,862 text matrix. Sample n rows without replacement, sample p columns that vary in those rows, standardize selected columns. Plant six relevant columns and generate linear or square response, center/scale signal to SD1, add independent N(0,1), center y. Original labels are unused. These are real-X semi-simulations, not tests on original labels.',
        overlap='Both are standard benchmark families; response families can overlap model development. New random draws and the two new real predictor pools are used. No claim that all mechanisms were unseen.',
        software=old['software'])
    write_json(PROTOCOL, plan)
    write_json(DATA/'common_benchmark_work.json', dict(status='frozen', settings=12, datasets=1200,
               protocol_sha256=sha(PROTOCOL), jobs=[], updated_utc=datetime.now(timezone.utc).isoformat()))
    print('Frozen 12 settings, 1200 datasets.', flush=True)


def initialize():
    from explore.current import load_model
    torch.set_num_threads(1)
    # Reuse precisely the already-audited evaluator; replace only its data provider.
    engine.PLAN = json.loads(PROTOCOL.read_text())
    engine.MODEL = load_model()
    engine.generate = generate


def one(task):
    return engine.one(task)


def database(shard):
    return DATA/f'common_benchmark_{shard}.sqlite'


def run(workers, shard, shards):
    plan = json.loads(PROTOCOL.read_text())
    verify_frozen_files(plan['frozen_files'])
    with sqlite3.connect(database(shard)) as db:
        db.execute('create table if not exists metadata (key text primary key,value text)')
        db.execute('create table if not exists results (id integer primary key,payload blob)')
        existing = db.execute("select value from metadata where key='protocol_sha256'").fetchone()
        if existing:
            assert existing[0] == sha(PROTOCOL)
        else:
            db.execute('insert into metadata values (?,?)', ('protocol_sha256', sha(PROTOCOL)))
            db.commit()
        done = {row[0] for row in db.execute('select id from results')}
        tasks = [t for t in plan['tasks'] if t['id'] % shards == shard and t['id'] not in done]
        tasks.sort(key=lambda t: t['p'], reverse=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize) as executor:
            futures = [executor.submit(one, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result()
                db.execute('insert into results values (?,?)',
                           (row['id'], zlib.compress(pickle.dumps(row, protocol=5), 3)))
                if i % 10 == 0:
                    db.commit()
                if i % 50 == 0:
                    print('shard', shard, 'completed', len(done)+i, 'of', len(done)+len(tasks), flush=True)
            db.commit()
    print('Finished shard', shard, flush=True)


def read_rows():
    rows = []
    for path in sorted(DATA.glob('common_benchmark_[0-9]*.sqlite')):
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
            assert db.execute("select value from metadata where key='protocol_sha256'").fetchone()[0] == sha(PROTOCOL)
            rows.extend(pickle.loads(zlib.decompress(blob)) for blob, in db.execute('select payload from results'))
    return sorted(rows, key=lambda r: r['id'])


def summarize():
    rows = read_rows(); plan = json.loads(PROTOCOL.read_text())
    assert len(rows) == len(plan['tasks']) == 1200
    assert [r['id'] for r in rows] == list(range(1200))
    assert len({r['input_sha256'] for r in rows}) == len(rows)
    for row, task in zip(rows, plan['tasks']):
        assert all(row[k] == v for k, v in task.items())
        mask = np.unpackbits(row['selected_bits'], axis=-1, count=row['p']).astype(bool)
        truth = np.isin(np.arange(row['p']), row['truth'])
        np.testing.assert_array_equal(mask[:, :, truth].sum(2), row['tp'])
        np.testing.assert_array_equal(mask[:, :, ~truth].sum(2), row['fp'])
    tp = np.stack([r['tp'] for r in rows], axis=1)
    fp = np.stack([r['fp'] for r in rows], axis=1)
    support = np.array([len(r['truth']) for r in rows]); cases = np.array([r['case'] for r in rows])
    np.savez_compressed(COUNTS, tp=tp, fp=fp, support=support, cases=cases, targets=plan['targets'], methods=METHODS,
        **{k:np.array([r[k] for r in rows]) for k in ('n', 'p', 'seed', 'ipss_seed')})
    tpr = tp/support[None, :, None]; fdr = fp/np.maximum(tp+fp, 1)
    results = []
    for case, spec in enumerate(plan['settings']):
        take = cases == case; assert take.sum() == 100
        values = {}
        for m, name in enumerate(METHODS):
            values[name] = {}
            for key, arr in [('tpr', tpr), ('fdr', fdr)]:
                values[name][key] = arr[m, take].mean(0).tolist()
                values[name][key+'_se'] = (arr[m, take].std(0, ddof=1)/10).tolist()
        results.append(dict(spec=spec, datasets=100, methods=values))
    write_json(RESULTS, dict(protocol_sha256=sha(PROTOCOL), counts_sha256=sha(COUNTS),
        datasets=1200, settings=results, targets=plan['targets'], methods=METHODS,
        counts_recounted=True, all_settings_reported=True))
    print('Summarized 1200 datasets; all selection masks recounted.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run', 'summarize'])
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    elif args.action == 'run':
        run(args.workers, args.shard, args.shards)
    else:
        summarize()
