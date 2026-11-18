"""Rebuild every new input and independently check its saved selections."""
import hashlib
import json
import numpy as np
import common_benchmarks as benchmark
from release_files import verify_frozen_files, recorded_excluded_seeds
from current_data import DATA, sha, standardized, write_json


def check_formula(spec, task, x, y, truth, metadata):
    """Replay the stated formulas independently of the generator branches."""
    n, p = task['n'], task['p']
    if spec['family'] == 'friedman1':
        rng = np.random.RandomState(task['seed'])
        raw = rng.uniform(size=(n, p))
        expected = (10*np.sin(np.pi*raw[:, 0]*raw[:, 1])
                    +20*(raw[:, 2]-.5)**2+10*raw[:, 3]+5*raw[:, 4]
                    +rng.standard_normal(n))
        permutation = rng.permutation(p)
        np.testing.assert_array_equal(x, raw[:, permutation])
        np.testing.assert_array_equal(y, expected)
        np.testing.assert_array_equal(truth, np.flatnonzero(permutation < 5))
    elif spec['family'] == 'sparse_linear':
        rng = np.random.default_rng(task['seed'])
        expected_x = rng.normal(size=(n, p))
        for j in range(1, p):
            expected_x[:, j] = .25*expected_x[:, j-1]+np.sqrt(1-.25**2)*expected_x[:, j]
        relevant = np.sort(rng.choice(p, 10, replace=False))
        expected_y = (4.5/np.sqrt(n))*expected_x[:, relevant].sum(1)+rng.normal(size=n)
        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_array_equal(y, expected_y)
        np.testing.assert_array_equal(truth, relevant)
    else:
        rng = np.random.default_rng(task['seed'])
        source = benchmark.pool(spec['dataset'])
        rows = rng.choice(len(source), n, replace=False)
        eligible = np.flatnonzero(source[rows].std(0) > 1e-10)
        columns = rng.choice(eligible, p, replace=False)
        np.testing.assert_array_equal(rows, metadata['source_rows'])
        np.testing.assert_array_equal(columns, metadata['source_columns'])
        np.testing.assert_array_equal(x, standardized(source[rows][:, columns]))
        relevant = np.sort(rng.choice(p, 6, replace=False))
        np.testing.assert_array_equal(truth, relevant)
        signal = x[:, relevant].sum(1) if spec['family'] == 'linear' else (x[:, relevant]**2).sum(1)
        expected_y = standardized(signal)+rng.normal(size=n)
        expected_y -= expected_y.mean()
        np.testing.assert_array_equal(y, expected_y)
        assert len(set(rows)) == n and len(set(columns)) == p


def main():
    plan = json.loads(benchmark.PROTOCOL.read_text())
    rows = benchmark.read_rows()
    assert len(rows) == len(plan['tasks']) == 1200
    assert [r['id'] for r in rows] == list(range(1200))
    verify_frozen_files(plan['frozen_files'])
    old_seeds = recorded_excluded_seeds()
    for path in DATA.glob('*protocol.json'):
        if path != benchmark.PROTOCOL:
            benchmark.engine.collect_seeds(json.loads(path.read_text()), old_seeds)
    training = json.loads((DATA/'current_training.json').read_text())
    benchmark.engine.collect_seeds(training, old_seeds)
    training_hashes = {r['input_sha256'] for r in training['fitting_records']}
    for path in DATA.glob('retained_regression_*_counts.npz'):
        with np.load(path) as a:
            for key in a.files:
                if 'seed' in key:
                    old_seeds.update(map(int, a[key].ravel()))
    seeds = [t[key] for t in plan['tasks'] for key in ('seed', 'ipss_seed')]
    assert len(seeds) == len(set(seeds)) and not old_seeds.intersection(seeds)
    assert not training_hashes.intersection(r['input_sha256'] for r in rows)
    targets = np.asarray(plan['targets']); cases = np.zeros(12, dtype=int); hashes = set()
    with np.load(benchmark.COUNTS) as a:
        for index, (row, task) in enumerate(zip(rows, plan['tasks'])):
            assert all(row[k] == v for k, v in task.items())
            spec = plan['settings'][task['case']]; cases[task['case']] += 1
            n, p = task['n'], task['p']
            assert 50 <= n <= p <= 500
            assert n == (p if spec['ratio'] == 1 else max(50, int(.8*p)))
            x, y, truth, metadata = benchmark.generate(spec, task)
            digest = hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
            assert digest == row['input_sha256'] and digest not in hashes
            hashes.add(digest)
            check_formula(spec, task, x, y, truth, metadata)
            np.testing.assert_array_equal(truth, row['truth'])
            assert a['support'][index] == len(truth) and a['cases'][index] == task['case']
            for key in ('n', 'p', 'seed', 'ipss_seed'):
                assert a[key][index] == task[key]
            masks = np.unpackbits(row['selected_bits'], axis=-1, count=p).astype(bool)
            true = np.isin(np.arange(p), truth)
            np.testing.assert_array_equal(masks[:, :, true].sum(2), a['tp'][:, index])
            np.testing.assert_array_equal(masks[:, :, ~true].sum(2), a['fp'][:, index])
            np.testing.assert_array_equal(masks[1], row['baseline_q']['gb'][None, :] <= targets[:, None])
            np.testing.assert_array_equal(masks[2], row['baseline_q']['l1'][None, :] <= targets[:, None])
            assert np.all(masks[3] == masks[3, 0])
            for i, alpha in enumerate(targets):
                candidates = np.flatnonzero(row['screening_q'] <= alpha)
                ordered = sorted(candidates, key=lambda j: (float(row['probability'][j]), int(j)))
                total = 0.; last = 0
                for length, j in enumerate(ordered, 1):
                    value = float(row['probability'][j]); total += value
                    end = length == len(ordered) or value != float(row['probability'][ordered[length]])
                    if end and total/length <= alpha:
                        last = length
                assert set(ordered[:last]) == set(np.flatnonzero(masks[0, i]))
            if (index+1) % 200 == 0:
                print('Audited', index+1, 'new datasets', flush=True)
    assert np.all(cases == 100)
    # A saved IPSS path avoids rerunning its Monte Carlo fits while checking the
    # public raw-data interface on every generator and dimension regime.
    benchmark.initialize()
    from explore.current import select_features
    checked = []
    for row in rows:
        if row['replicate'] != 0:
            continue
        x, y, _, _ = benchmark.generate(plan['settings'][row['case']], row)
        result = select_features(x, y, targets, model=benchmark.engine.MODEL,
                                 paths=row['paths'], random_seed=row['ipss_seed'])
        mask = np.unpackbits(row['selected_bits'][0], axis=-1, count=row['p'])
        for i, ids in enumerate(result['selected']):
            np.testing.assert_array_equal(ids, np.flatnonzero(mask[i]))
        checked.append(row['case'])
        print('Verified public API case', row['case'], flush=True)
    assert checked == list(range(12))
    write_json(DATA/'common_benchmark_audit.json', dict(
        protocol_sha256=sha(benchmark.PROTOCOL), counts_sha256=sha(benchmark.COUNTS),
        results_sha256=sha(benchmark.RESULTS), auditor_sha256=sha(__file__),
        datasets_regenerated=1200, all_input_hashes_match=True,
        all_response_formulas_independently_replayed=True,
        no_duplicate_generated_datasets=True, no_recorded_training_hash_overlap=True,
        no_recorded_training_or_old_assessment_seed_overlap=True,
        all_dimensions_verified=True, per_setting_counts=cases.tolist(),
        all_selection_masks_recounted=True, all_model_selections_independently_reconstructed=True,
        all_baseline_masks_match_saved_qvalues=True, public_api_cases=checked,
        frozen_files_unchanged=True))
    print('New benchmark audit and public API checks passed.', flush=True)


if __name__ == '__main__':
    main()
