"""Construct the current model's 443-entry variable measurements."""
from dataclasses import dataclass
import numpy as np
from .model import input_names,global_vector,prior_context,feature_values

@dataclass
class MeasurementSchema:
    global_names: list
    feature_names: list
    prior: object
@dataclass
class ExpandedF:
    estimator: object
    original: object
    metadata: dict

    def matrix(self, record):
        if input_names(record) != self.original.feature_names:
            raise ValueError('Variable measurement schema changed.')
        p = record['paths']['p_full']
        shared = global_vector(prior_context(record, self.original.prior), self.original.global_names)
        return np.column_stack([np.broadcast_to(shared, (p, len(shared))),
                                feature_values(record, np.arange(p))]).astype(np.float32)

    def probability(self, record):
        return self.estimator.predict_proba(self.matrix(record))[:, 1]

