"""The current 443-128-64-1 neural scorer and its fixed scaling."""
import torch
from torch import nn
import numpy as np

def network(config):
    if config['architecture']=='mlp':
        layers=[];width=443
        for out in config['hidden']:
            layers.extend([nn.Linear(width,out),nn.GELU(),nn.Dropout(config['dropout'])]);width=out
        return nn.Sequential(*layers,nn.Linear(width,1))
    raise ValueError(config['architecture'])

def transform(matrix,mean,scale):
    matrix=np.asarray(matrix,dtype=np.float32)
    transformed=np.sign(matrix)*np.log1p(np.abs(matrix))
    return np.clip((transformed-mean)/scale,-10,10).astype(np.float32)

class NeuralScorer:
    def __init__(self,path):
        state=torch.load(path,map_location='cpu',weights_only=True)
        self.config=state['config'];self.model=network(self.config)
        self.model.load_state_dict(state['state_dict']);self.model.eval()
        self.mean=state['mean'].numpy();self.scale=state['scale'].numpy()

    def predict_proba(self,matrix):
        values=[]
        with torch.inference_mode():
            for start in range(0,len(matrix),16384):
                x=torch.from_numpy(transform(matrix[start:start+16384],self.mean,self.scale))
                values.append(torch.sigmoid(self.model(x).ravel()).numpy())
        p=np.concatenate(values)
        return np.column_stack([1-p,p])
