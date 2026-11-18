"""Reproduce the retained fitting recipe; never overwrite the released weights."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,gc,json,sys
from pathlib import Path
import torch
from torch import nn
import numpy as np
PIPELINE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PIPELINE))
from current_data import ROOT,DATA,sha
from explore.blocks import network,transform

def fit(output,component,device,threads):
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(threads)
    metadata=json.loads((DATA/'current_training.json').read_text())
    if component=='prior':
        from xgboost import XGBRegressor
        with np.load(DATA/'current_prior_training.npz') as f:x,y,groups=f['x'],f['y'],f['groups']
        _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True);w=1/counts[inverse];w/=w.mean()
        model=XGBRegressor(**metadata['prior_parameters']).fit(x,y,sample_weight=w)
        output.parent.mkdir(parents=True,exist_ok=True);model.save_model(output)
        print('Wrote the refitted fraction estimator:',output);return
    initial=metadata['initial_stage'];final=metadata['final_stage'];config=initial['config']
    with np.load(DATA/'block_training_cache.npz') as f:train={k:f[k] for k in f.files}
    assert sha(DATA/'block_training_cache.npz')==initial['training_cache_sha256']
    schema=json.loads((ROOT/'models/expanded_f.json').read_text())
    qcols=[313+i for i,name in enumerate(schema['feature_names']) if name.startswith('q_') or name.startswith('peer_q_')]
    hard=np.min(train['x'][:,qcols],axis=1)<=.5
    x=torch.from_numpy(transform(train.pop('x'),train['mean'],train['scale'])).to(device);gc.collect()
    labels=torch.from_numpy(train['y']).to(device);torch.backends.cuda.matmul.allow_tf32=False
    def optimize(model,weights,epochs,schedule_epochs,lr,minimum):
        w=torch.from_numpy(weights).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=schedule_epochs,eta_min=minimum)
        for epoch in range(1,epochs+1):
            model.train();order=torch.randperm(len(x),device=device);total=0.
            for start in range(0,len(x),8192):
                ids=order[start:start+8192];optimizer.zero_grad(set_to_none=True)
                loss=(nn.functional.binary_cross_entropy_with_logits(model(x[ids]).ravel(),labels[ids],reduction='none')*w[ids]).mean()
                assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
                total+=float(loss.detach())*len(ids)
            scheduler.step();print('epoch',epoch,'loss',total/len(x),flush=True)
        return model
    torch.manual_seed(initial['seed']);np.random.seed(initial['seed'])
    model=network(config).to(device);fraction=float(np.average(train['y'],weights=train['w']))
    last=[layer for layer in model.modules() if isinstance(layer,nn.Linear)][-1]
    with torch.no_grad():last.bias.fill_(float(np.log(fraction/(1-fraction))))
    model=optimize(model,train['w'],initial['best_epoch'],35,.001,.00005)
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};del model
    # The original continuation recreated the network before loading its initialization.
    torch.manual_seed(2026110555);np.random.seed(2026110555);model=network(config).to(device);model.load_state_dict(state)
    weights=train['w'].copy();weights*=np.where(hard,10.,1.);original=np.zeros(len(weights),bool)
    for i,(s,t) in enumerate(zip(train['offsets'][:-1],train['offsets'][1:])):
        weights[s:t]*=train['w'][s:t].sum()/weights[s:t].sum();original[s:t]=str(train['ids'][i]).startswith('original_train:')
    weights[original]*=.65/weights[original].sum();weights[~original]*=.35/weights[~original].sum();weights/=weights.mean()
    model=optimize(model,weights,final['best']['epoch'],20,.0003,.000015)
    saved=dict(config=config,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        mean=torch.from_numpy(train['mean']),scale=torch.from_numpy(train['scale']))
    output.parent.mkdir(parents=True,exist_ok=True);torch.save(saved,output)
    print('Wrote a reproduction of the fixed recipe:',output,'Released weights were unchanged.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--component',choices=['F','prior'],default='F')
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4);a=p.parse_args();fit(a.output,a.component,a.device,a.threads)
