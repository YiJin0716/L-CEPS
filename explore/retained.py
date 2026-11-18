"""Load the retrained small MLP with the fixed ten-IPSS screening rule."""
import json
from pathlib import Path

from xgboost import XGBRegressor
from .blocks import NeuralScorer
from .expanded_f import MeasurementSchema
from .model import SupportPrior,sha
from .screening import ScreenedMLP

def load_retained_mlp():
    root=Path(__file__).resolve().parents[1]/'models'
    metadata=json.loads((root/'retained_mlp.json').read_text())
    for name,expected in metadata['weights_and_schema_sha256'].items():
        if sha(root/name)!=expected:raise ValueError('Retained model file changed: '+name)
    schema=json.loads((root/'expanded_f.json').read_text())
    prior=XGBRegressor(n_jobs=1);prior.load_model(root/'final/prior.ubj')
    measurements=MeasurementSchema(schema['global_names'],schema['feature_names'],SupportPrior(prior,schema['prior_names']))
    return ScreenedMLP(NeuralScorer(root/metadata['weights_file']),measurements,metadata)
