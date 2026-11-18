"""The released IPSS path calculation, independent of delta-learning scripts."""
from inspect import signature
import numpy as np
from joblib import Parallel, delayed
from ipss.helpers import compute_delta, score_based_selection
from ipss.main import selection
from ipss.processing import preprocess_ipss

def compute_stability_paths(x, y, selector, n_jobs=1):
	# Newer ipss versions add a separate preselector argument. Use keywords
	# so both interfaces accept the same settings without shifting arguments.
	extra = {'preselector': None} if 'preselector' in signature(preprocess_ipss).parameters else {}
	prep = preprocess_ipss(x, y, selector=selector, selector_args=None,
		preselect=True, preselector_args=None, B=None, n_alphas=None,
		ipss_function=None, delta=None, standardize_X=None, center_y=None,
		force_regression=False, **extra)

	if prep is None:
		return {'null': True, 'p_full': x.shape[1], 'default_delta': compute_delta(x, selector)}

	selector_function, selector_args = prep['selector_function'], prep['selector_args']
	alphas, n_alphas, b = prep['alphas'], prep['n_alphas'], prep['B']
	subsample_size = prep['X'].shape[0] // 2

	results = Parallel(n_jobs=n_jobs)(
		delayed(selection)(prep['X'], prep['y'], alphas, subsample_size, selector_function, **selector_args)
		for _ in range(b)
	)

	return prepare_stability_paths(np.array(results), prep)


def prepare_stability_paths(results, prep):
	"""Cache only the delta-independent part of ipss.postprocess_ipss.

	Keep IPSS's aggregation order and plateau truncation for matching results.
	The later, delta-dependent integral cutoff stays inside ipss_scores.
	"""
	alphas, n_alphas, b = prep['alphas'], prep['n_alphas'], prep['B']
	p = prep['X'].shape[1]
	if alphas is None:
		# The helper can shift negative importance scores in place.
		results, alphas = score_based_selection(results.copy(), n_alphas)

	z = np.zeros((n_alphas, 2 * b, p))
	for i in range(b):
		z[:, 2 * i:2 * (i + 1), :] = results[i]
	average_selected = np.array([np.mean(np.sum(z[i], axis=1)) for i in range(n_alphas)])
	stability_paths = np.array([z[i].mean(axis=0) for i in range(n_alphas)])

	# This plateau rule does not depend on delta; do it once as well.
	stop_index = n_alphas
	for i in range(2, len(alphas)):
		if np.isclose(stability_paths[i], np.zeros(p)).all():
			continue
		diff = stability_paths[i] - stability_paths[i - 2]
		if np.isclose(diff, np.zeros(p)).all() and stability_paths[i].mean() > 0.01:
			stop_index = i
			break

	return {
		'null': False,
		'stability_paths': stability_paths[:stop_index],
		'average_selected': average_selected[:stop_index],
		'alphas': alphas[:stop_index],
		'p_preselected': p,
		'p_full': prep['p_full'],
		'preselect_indices': prep['preselect_indices'],
		'b': b,
		'ipss_function': prep['ipss_function'],
		'default_delta': prep['delta'],  # ipss's own heuristic, computed as a side effect of delta=None
	}

