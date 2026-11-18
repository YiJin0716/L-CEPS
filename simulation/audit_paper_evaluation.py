"""Audit the saved assessment against its frozen inputs and selection rules."""
import hashlib
import json
import numpy as np
from paper_evaluation import PROTOCOL, DATA, generate, read_rows, collect_seeds
from release_files import verify_frozen_files, recorded_excluded_seeds
from current_data import sha, write_json


def main():
    plan = json.loads(PROTOCOL.read_text())
    rows = read_rows()
    assert len(rows) == len(plan['tasks']) == 2800
    assert [r['id'] for r in rows] == list(range(2800))
    verify_frozen_files(plan['frozen_files'])

    old_seeds = recorded_excluded_seeds()
    for path in DATA.glob('*protocol.json'):
        if path != PROTOCOL:
            collect_seeds(json.loads(path.read_text()), old_seeds)
    training = json.loads((DATA/'current_training.json').read_text())
    collect_seeds(training, old_seeds)
    training_hashes = {r['input_sha256'] for r in training['fitting_records']}
    for path in DATA.glob('retained_regression_*_counts.npz'):
        with np.load(path) as archive:
            for key in archive.files:
                if 'seed' in key:
                    old_seeds.update(map(int, archive[key].ravel()))
    new_seeds = [t[key] for t in plan['tasks'] for key in ('seed', 'ipss_seed')]
    assert len(new_seeds) == len(set(new_seeds))
    assert not old_seeds.intersection(new_seeds)
    assert not training_hashes.intersection(r['input_sha256'] for r in rows)

    targets = np.asarray(plan['targets'])
    cases = np.zeros(28, dtype=int)
    all_hashes = set()
    with np.load(DATA/'paper_evaluation_counts.npz') as archive:
        for index, (row, task) in enumerate(zip(rows, plan['tasks'])):
            assert all(row[key] == value for key, value in task.items())
            spec = plan['settings'][task['case']]
            cases[task['case']] += 1
            n, p = task['n'], task['p']
            assert 50 <= n <= p <= 500
            assert n == (p if spec['ratio'] == 1 else max(50, int(.8*p)))
            x, y, truth, metadata = generate(spec, task)
            digest = hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
            assert digest == row['input_sha256'] and digest not in all_hashes
            all_hashes.add(digest)
            np.testing.assert_array_equal(truth, row['truth'])
            assert x.shape == (n, p) and y.shape == (n,)
            np.testing.assert_allclose(x.mean(0), 0, atol=1e-10)
            np.testing.assert_allclose(x.std(0), 1, atol=1e-10)
            k = len(truth)
            assert archive['support'][index] == k and archive['cases'][index] == task['case']
            for key in ('n', 'p', 'seed', 'ipss_seed'):
                assert archive[key][index] == task[key]
            assert (5 <= k <= 15 if spec['suite'] == 'gaussian' else
                    10 <= k <= 30 if spec['suite'] == 'rnaseq' else k == 6)
            if spec['suite'] == 'rnaseq':
                groups = metadata['groups']
                assert k//2 <= len(groups) <= k
                np.testing.assert_array_equal(sorted(j for g in groups for j in g['columns']), truth)
            if 'source_rows' in metadata:
                assert len(set(metadata['source_rows'])) == n
                assert len(set(metadata['source_columns'])) == p
            assert set(np.unique(y)) == {0., 1.} if spec['response'] == 'class' else abs(y.mean()) < 1e-10

            masks = np.unpackbits(row['selected_bits'], axis=-1, count=p).astype(bool)
            true = np.isin(np.arange(p), truth)
            np.testing.assert_array_equal(masks[:, :, true].sum(2), archive['tp'][:, index])
            np.testing.assert_array_equal(masks[:, :, ~true].sum(2), archive['fp'][:, index])
            np.testing.assert_array_equal(masks[1], row['baseline_q']['gb'][None, :] <= targets[:, None])
            np.testing.assert_array_equal(masks[2], row['baseline_q']['l1'][None, :] <= targets[:, None])
            assert np.all(masks[3] == masks[3, 0])

            # Reconstruct the mathematical rule without calling model selection code.
            for a, alpha in enumerate(targets):
                candidates = np.flatnonzero(row['screening_q'] <= alpha)
                ordered = sorted(candidates, key=lambda j: (float(row['probability'][j]), int(j)))
                total = 0.; last = 0
                for length, j in enumerate(ordered, 1):
                    score = float(row['probability'][j]); total += score
                    is_end = length == len(ordered) or score != float(row['probability'][ordered[length]])
                    if is_end and total/length <= alpha:
                        last = length
                assert set(ordered[:last]) == set(np.flatnonzero(masks[0, a]))
            if (index+1) % 400 == 0:
                print('Audited', index+1, 'datasets', flush=True)
    assert np.all(cases == 100)
    write_json(DATA/'paper_evaluation_audit.json', dict(
        protocol_sha256=sha(PROTOCOL), auditor_sha256=sha(__file__),
        counts_sha256=sha(DATA/'paper_evaluation_counts.npz'),
        datasets_regenerated=2800, all_input_hashes_match=True,
        no_duplicate_generated_datasets=True, no_recorded_training_hash_overlap=True,
        no_recorded_training_or_old_assessment_seed_overlap=True,
        all_dimensions_verified=True, per_setting_counts=cases.tolist(),
        all_selection_masks_recounted=True, all_model_selections_independently_reconstructed=True,
        all_baseline_masks_match_saved_qvalues=True, frozen_files_unchanged=True))
    print('Assessment audit passed.', flush=True)


if __name__ == '__main__':
    main()
