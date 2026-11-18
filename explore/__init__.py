"""The released IPSS-screened small MLP."""
import torch

def load_model():
    from .current import load_model as load
    return load()

def select_features(*args,**kwargs):
    from .current import select_features as select
    return select(*args,**kwargs)
