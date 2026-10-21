"""D-a scene-held-out ridge probes. No navigation parameters are involved.

Predeclared lambda grid: .01,.1,1,10,100,1000,10000 for sum-squared-error
ridge. Inner CV maximizes mean five-region cosine; ties choose larger lambda.
Outer folds: seeded shuffled unique scenes split into two groups; inner folds
use only training scenes split into three groups. Every scaler is fitted anew
on the corresponding training fold. Targets are centered for an intercept.
"""
import numpy as np

LAMBDAS=(.01,.1,1.,10.,100.,1000.,10000.)


def scene_folds(scenes,count,seed=20261019):
    unique=sorted(set(map(str,scenes)))
    if len(unique)<count:raise ValueError('Insufficient scenes for disjoint folds')
    shuffled=np.asarray(unique)[np.random.default_rng(seed).permutation(len(unique))]
    return [sorted(x.tolist()) for x in np.array_split(shuffled,count)]


def _problem(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if x.ndim!=2 or y.ndim<2 or len(x)!=len(y) or not len(x):
        raise ValueError('Expected nonempty aligned [samples,features] and targets')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite probe data')
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>0,scale,1.)
    normalized=(x-mean)/scale
    target_shape=y.shape[1:];flat=y.reshape(len(y),-1);y_mean=flat.mean(0)
    centered=flat-y_mean
    # The primal form is reused for every candidate lambda, avoiding repeated
    # feature normalization and Gram construction across inner CV candidates.
    return dict(x_mean=mean,x_scale=scale,y_mean=y_mean,target_shape=target_shape,
                gram=normalized.T@normalized,rhs=normalized.T@centered,n_train=len(x))


def _solve(problem,regularization,device='cpu'):
    if not np.isfinite(regularization) or regularization<=0:raise ValueError('Positive finite ridge lambda required')
    gram=problem['gram'].copy()
    gram.flat[::len(gram)+1]+=regularization
    # Positive ridge shift makes the Gram SPD: Cholesky avoids a generic LU
    # solve with thousands of right-hand sides, on CPU or requested device.
    import torch
    with torch.no_grad():
        g=torch.as_tensor(gram,device=device,dtype=torch.float64)
        rhs=torch.as_tensor(problem['rhs'],device=device,dtype=torch.float64)
        weights=torch.cholesky_solve(rhs,torch.linalg.cholesky(g)).cpu().numpy()
    if not np.isfinite(weights).all():raise ValueError('Nonfinite ridge solution')
    return {**{key:value for key,value in problem.items() if key not in ('gram','rhs')},
            'weights':weights,'lambda':regularization}


def fit_ridge(x,y,regularization,device='cpu'):
    return _solve(_problem(x,y),regularization,device)


def predict(fitted,x):
    x=np.asarray(x,dtype=np.float64)
    values=((x-fitted['x_mean'])/fitted['x_scale'])@fitted['weights']+fitted['y_mean']
    return values.reshape(len(x),*fitted['target_shape'])


def normalize(values):
    values=np.asarray(values,dtype=np.float64)
    return values/np.maximum(np.linalg.norm(values,axis=-1,keepdims=True),1e-12)


def region_cosines(prediction,target):
    if prediction.shape!=target.shape or prediction.ndim!=3 or prediction.shape[1]!=5:
        raise ValueError('Expected aligned [samples,5,descriptor]')
    return (normalize(prediction)*normalize(target)).sum(-1)


def retrieval_metrics(prediction,episode_targets,source):
    """Rank among every frame; exact ties get average rank and fractional top1."""
    if prediction.ndim!=2 or prediction.shape[0]!=5 or episode_targets.shape[1:]!=prediction.shape:
        raise ValueError('Expected prediction [5,D], all episode targets [frames,5,D]')
    if not 0<=source<len(episode_targets):raise ValueError('Source outside episode')
    similarities=(normalize(episode_targets)*normalize(prediction)[None]).sum(-1).mean(-1)
    score=similarities[source]
    greater=int((similarities>score).sum());tied=int((similarities==score).sum())
    rank=1+greater+(tied-1)/2
    return dict(cosine=float(score),rank=rank,top1=1/tied if greater==0 else 0.,
                mrr=1/rank,n_candidate_frames=len(episode_targets),ties=tied)


def choose_lambda(x,y,scenes,lambdas=LAMBDAS,device='cpu',prediction_chunk_size=64):
    if prediction_chunk_size<1:raise ValueError('Positive prediction chunk size required')
    scenes=np.asarray(scenes,dtype=str)
    if len(scenes)!=len(x):raise ValueError('Scene/sample mismatch')
    folds=scene_folds(scenes,3)
    totals={float(value):[0.,0] for value in lambdas};evidence=[]
    for validation in folds:
        is_validation=np.isin(scenes,validation);is_train=~is_validation
        problem=_problem(x[is_train],y[is_train])
        evidence.append(dict(train_scenes=sorted(set(scenes[is_train])),validation_scenes=validation,
                             n_train=int(is_train.sum()),n_validation=int(is_validation.sum()),scores={}))
        for regularization in lambdas:
            fitted=_solve(problem,regularization,device)
            indices=np.flatnonzero(is_validation);fold_sum=0.
            for start in range(0,len(indices),prediction_chunk_size):
                part=indices[start:start+prediction_chunk_size]
                values=region_cosines(predict(fitted,x[part]),y[part]).mean(-1)
                fold_sum+=float(values.sum())
            totals[regularization][0]+=fold_sum;totals[regularization][1]+=len(indices)
            evidence[-1]['scores'][str(regularization)]=fold_sum/len(indices)
    scores={value:total/count for value,(total,count) in totals.items()}
    selected=max(scores,key=lambda value:(scores[value],value))
    return dict(**{'lambda':selected},scores={str(k):v for k,v in scores.items()},folds=evidence,
                selection_metric='mean five-region cosine over inner held-out samples',
                tie_rule='larger lambda',ridge_objective='sum squared error + lambda * squared weights')


def fit_outer_folds(x,y,scenes,lambdas=LAMBDAS,device='cpu'):
    """Return held-out predictions and fitted models for later D-b transforms."""
    scenes=np.asarray(scenes,dtype=str)
    predictions=np.empty_like(y,dtype=np.float64);models=[]
    for test_scenes in scene_folds(scenes,2):
        test=np.isin(scenes,test_scenes);train=~test
        selection=choose_lambda(x[train],y[train],scenes[train],lambdas,device)
        fitted=fit_ridge(x[train],y[train],selection['lambda'],device)
        predictions[test]=predict(fitted,x[test])
        models.append(dict(model=fitted,selection=selection,test_scenes=test_scenes,
                           train_scenes=sorted(set(scenes[train])),n_train=int(train.sum()),n_test=int(test.sum())))
    return predictions,models
