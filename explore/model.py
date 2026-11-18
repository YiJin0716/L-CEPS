"""Measurement helpers and the fitted relevant-fraction estimator."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
from .features import path_features

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def global_vector(features, names):
    return np.asarray([features[name] for name in names], dtype=np.float32)

def input_names(record):
    return record['feature_names'] + ['peer_' + s for s in record['feature_names'][33:50]]

def feature_values(record, indices):
    return np.column_stack([record['feature_x'][indices], record['peer_feature_x'][indices, 33:50]])

@dataclass
class SupportPrior:
    estimator: object
    names: list

    def predict(self, records):
        matrix = np.array([global_vector(r['features'], self.names) for r in records])
        return np.clip(self.estimator.predict(matrix), -7., 3.)

def prior_context(record, prior, override=None):
    if override is not None:
        predicted = float(override)
    else:
        cache = record.setdefault('_priors', {})
        if id(prior) not in cache:
            cache[id(prior)] = float(prior.predict([record])[0])
        predicted = cache[id(prior)]
    context = {**record['features'], 'estimated_signal_logit': predicted,
               'estimated_signal_fraction': float(expit(predicted))}
    context.update({'peer_' + k: v for k, v in path_features(record['peer_paths']).items()})
    return context
