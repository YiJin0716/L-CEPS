"""Public entry point for the current IPSS-screened small MLP."""
import argparse
import json
from pathlib import Path

import torch
import numpy as np

from .model import sha
from .screening import RULE
from .retained import load_retained_mlp


def load_model():
    """Load g, the small MLP F, and its target-dependent IPSS screening rule."""
    pipeline=Path(__file__).resolve().parents[1]
    root=pipeline/'models'
    # The manifest identifies the single released model in this folder.
    manifest_path=root/'active_model.json'
    if not manifest_path.exists():
        raise FileNotFoundError('No model has been released through this entry point.')
    manifest=json.loads(manifest_path.read_text())
    if (manifest['kind']!='mlp_screened_single_f' or not manifest['verified']
            or not manifest['candidate_gate'] or manifest['screening_rule']!=RULE):
        raise ValueError('The current manifest does not identify the verified screened MLP.')
    for name,expected in manifest['source_sha256'].items():
        if sha(pipeline/name)!=expected:raise ValueError('Model inference source changed: '+name)
    for name,expected in manifest['weights_and_metadata_sha256'].items():
        if sha(root/name)!=expected:raise ValueError('Model weights or metadata changed: '+name)
    return load_retained_mlp()


def select_features(x,y,targets=(.05,.1,.2),*,model=None,random_seed=302,paths=None):
    """Return selected original column IDs for each target, numbered from zero."""
    model=load_model() if model is None else model
    result=model.select(x,y,targets,random_seed=random_seed,paths=paths)
    result['model']='the model'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path,help='NPZ containing finite X and y.')
    parser.add_argument('--targets',type=float,nargs='+',default=[.05,.1,.2])
    parser.add_argument('--seed',type=int,default=302)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    with np.load(args.dataset,allow_pickle=False) as data:
        result=select_features(data['X'],data['y'],args.targets,random_seed=args.seed)
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.write_text(text)
    else:print(text,end='')


if __name__=='__main__':main()
