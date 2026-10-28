"""Proper-rigid multistart pilot; no atlas labels or physical z assumptions."""
import copy
import math
import numpy as np
import torch
from torch import nn

class Rigid2D(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.angle=nn.Parameter(torch.zeros((),device=device))
        self.translation=nn.Parameter(torch.zeros(2,device=device))

    def matrix(self):
        c,s=self.angle.cos(),self.angle.sin()
        return torch.stack((c,-s,s,c)).reshape(2,2)

    def forward(self,x):
        return (x-.5)@self.matrix().T+.5+self.translation

    def inverse(self,x):
        return (x-.5-self.translation)@self.matrix()+.5

def nearest_squared(x,support):
    with torch.no_grad(): indices=torch.cdist(x.detach(),support).argmin(1)
    return (x-support[indices]).square().sum(1).mean()

def profile(model,x):
    # Softmax of predicted log rates avoids overflow and cancels amplitude.
    return torch.softmax(model(x),dim=1)

def fit_rigid(reference,query,ref_model,query_model,ref_split,query_split,settings,*,use_expression):
    """Select poses on validation coordinates/decoded fields, never test expression."""
    device=reference.device
    for model in (ref_model,query_model): model.eval().requires_grad_(False)
    rtrain=reference[ref_split['train']]; qtrain=query[query_split['train']]
    rval=reference[ref_split['val']]; qval=query[query_split['val']]
    with torch.no_grad():
        prtrain=profile(ref_model,rtrain); pqtrain=profile(query_model,qtrain)
        prval=profile(ref_model,rval); pqval=profile(query_model,qval)
    def objective(t,r,q,pr,pq):
        tq=t(q); ir=t.inverse(r)
        geometry=(nearest_squared(tq,rtrain)+nearest_squared(ir,qtrain))/2
        if use_expression:
            expression=(1-nn.functional.cosine_similarity(profile(ref_model,tq),pq).mean()
                        +1-nn.functional.cosine_similarity(profile(query_model,ir),pr).mean())/2
            loss=expression+settings['geometry_weight']*geometry
        else:
            expression=geometry.new_zeros(())
            loss=geometry
        return loss,expression,geometry
    candidates=[]
    for angle in np.linspace(-180,180,settings['angle_starts'],endpoint=False):
        t=Rigid2D(device)
        with torch.no_grad():
            t.angle.fill_(math.radians(float(angle)))
            t.translation.copy_(rtrain.mean(0)-.5-(qtrain.mean(0)-.5)@t.matrix().T)
            loss,expression,geometry=objective(t,rval,qval,prval,pqval)
        if torch.isfinite(loss):
            candidates.append(dict(angle=float(angle),objective=float(loss),
                expression=float(expression),geometry=float(geometry),state=copy.deepcopy(t.state_dict())))
    if not candidates: raise RuntimeError('No finite coarse rigid pose')
    grid=[{k:v for k,v in c.items() if k!='state'} for c in candidates]
    best=float('inf'); selected=None; history=[]
    for candidate in sorted(candidates,key=lambda c:c['objective'])[:settings['refine_starts']]:
        torch.manual_seed(settings['seed'])
        t=Rigid2D(device); t.load_state_dict(candidate['state'])
        opt=torch.optim.Adam(t.parameters(),lr=settings['lr'])
        for step in range(settings['steps']+1):
            if step%20==0 or step==settings['steps']:
                with torch.no_grad():
                    loss,expression,geometry=objective(t,rval,qval,prval,pqval)
                history.append(dict(start_angle=candidate['angle'],step=step,objective=float(loss),
                                    expression=float(expression),geometry=float(geometry)))
                if torch.isfinite(loss) and float(loss)<best:
                    best=float(loss); selected=copy.deepcopy(t.state_dict())
            if step==settings['steps']: break
            ri=torch.randint(len(rtrain),(min(settings['batch_size'],len(rtrain)),),device=device)
            qi=torch.randint(len(qtrain),(min(settings['batch_size'],len(qtrain)),),device=device)
            loss,_,_=objective(t,rtrain[ri],qtrain[qi],prtrain[ri],pqtrain[qi])
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite rigid objective')
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(t.parameters(),5.); opt.step()
    if selected is None: raise RuntimeError('No finite rigid checkpoint')
    t=Rigid2D(device); t.load_state_dict(selected); t.requires_grad_(False)
    with torch.no_grad(): mapped=t(query).cpu()
    return dict(mapped=mapped,state={k:v.cpu() for k,v in t.state_dict().items()},
        angle_degrees=float(t.angle)*180/math.pi, determinant=float(torch.linalg.det(t.matrix())),
        validation_objective=best,grid=grid,history=history)
