"""Fit ablations using train only, select on validation, report old test as diagnostic.

Video labels supervise max-pooled person windows (MIL). No instance labels are
invented and external news clips are never included in training/threshold tuning.
"""
from pathlib import Path
import sys,argparse,json,copy,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,confusion_matrix,precision_score,recall_score,accuracy_score
from src.behavior_features import BEHAVIOR_VERSION,DESCRIPTOR_NAMES,DESCRIPTOR_DIM

ROOT=Path(__file__).resolve().parents[1]

def metrics(y,scores,threshold):
    pred=np.asarray(scores)>=threshold
    return dict(precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),
        f1=float(f1_score(y,pred,zero_division=0)),accuracy=float(accuracy_score(y,pred)),
        roc_auc=float(roc_auc_score(y,scores)),average_precision=float(average_precision_score(y,scores)),
        confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',default='data/manifest.csv');parser.add_argument('--epochs',type=int,default=160);a=parser.parse_args()
    torch.set_num_threads(2);np.random.seed(42);torch.manual_seed(42)
    m=pd.read_csv(ROOT/a.manifest)
    if m.video_id.duplicated().any() or m.sha256.duplicated().any():raise ValueError('Duplicate source across splits')
    bags=[];coverage=[]
    for r in m.to_dict('records'):
        cache=ROOT/'data/behavior_features'/f"{r['video_id']}_{r['sha256'][:12]}_{BEHAVIOR_VERSION}_w6_fps2.npz"
        z=np.load(cache);x=z['x'].astype(np.float32)
        if x.ndim!=2 or x.shape[1]!=DESCRIPTOR_DIM or not np.isfinite(x).all():raise ValueError('Invalid features '+str(cache))
        bags.append(x);info=json.loads(str(z['info']))
        coverage.append(dict(video_id=r['video_id'],split=r['split'],label=r['label'],instances=len(x),**{k:info[k] for k in ['sampled_frames','frames_with_pose']}))
    train=np.flatnonzero((m.split=='train').to_numpy());val=np.flatnonzero((m.split=='val').to_numpy());test=np.flatnonzero((m.split=='test').to_numpy());labels=m.label.to_numpy()
    fit=[i for i in train if len(bags[i])]
    if set(labels[fit])!={0,1}:raise ValueError('Not enough observed training videos from both classes')
    if set(labels[val])!={0,1} or set(labels[test])!={0,1}:raise ValueError('Validation/test need both labels')
    pose_tokens=['joint_','elbow','torso','wrist_speed','wrist_motion','head_','pose_coverage','body_aspect']
    pose_indices=[i for i,n in enumerate(DESCRIPTOR_NAMES) if any(s in n for s in pose_tokens)]
    hoi_indices=[i for i in range(DESCRIPTOR_DIM) if i not in pose_indices]
    variants={'pose_only':pose_indices,'interaction_only':hoi_indices,'pose_and_interaction':list(range(DESCRIPTOR_DIM))}
    out=ROOT/'artifacts/behavior_v2';out.mkdir(exist_ok=True)
    results={};bundles={};all_predictions={}
    for name,indices in variants.items():
        torch.manual_seed(42)
        xfit=np.concatenate([bags[i][:,indices] for i in fit])
        weights=np.concatenate([np.full(len(bags[i]),1/len(bags[i])) for i in fit])
        scaler=StandardScaler().fit(xfit,sample_weight=weights)
        tensors=[torch.tensor(np.clip(scaler.transform(x[:,indices]),-5,5),dtype=torch.float32) if len(x) else None for x in bags]
        model=torch.nn.Linear(len(indices),1);optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
        best_loss=float('inf');best_state=None;best_epoch=0;patience=0
        positive_weight=sum(labels[fit]==0)/max(1,sum(labels[fit]==1))
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(positive_weight)))
        def predict(ids):
            model.eval()
            with torch.no_grad():return np.array([float(torch.sigmoid(model(tensors[i]).max())) if tensors[i] is not None else 0. for i in ids])
        for epoch in range(a.epochs):
            model.train();optimizer.zero_grad()
            # Equal video weighting; positive bag label does not label every instance.
            losses=[criterion(model(tensors[i]).max().reshape(1),torch.tensor([float(labels[i])])) for i in fit]
            loss=torch.stack(losses).mean();loss.backward();optimizer.step()
            scores=np.clip(predict(val),1e-6,1-1e-6)
            vloss=float(-np.mean(labels[val]*np.log(scores)+(1-labels[val])*np.log(1-scores)))
            if vloss<best_loss-1e-5:
                best_loss=vloss;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1;patience=0
            else:patience+=1
            if patience>=25:break
        model.load_state_dict(best_state);v=predict(val)
        threshold=float(max(np.linspace(.1,.9,81),key=lambda t:(f1_score(labels[val],v>=t,zero_division=0),-abs(t-.5))))
        results[name]=dict(validation=metrics(labels[val],v,threshold),threshold=threshold,epoch=best_epoch,
            input_dimensions=len(indices),validation_loss=best_loss)
        bundles[name]=dict(version=BEHAVIOR_VERSION,input_dim=DESCRIPTOR_DIM,indices=indices,
            mean=scaler.mean_.astype(float).tolist(),scale=scaler.scale_.astype(float).tolist(),
            state=best_state,threshold=threshold,name=name,label_level='video',pool='max',
            instance_calibration='none',approved_for_real_world=False)
        all_predictions[name]=predict(test)
        print(name,'validation',results[name],flush=True)
    selected=max(results,key=lambda n:(results[n]['validation']['f1'],results[n]['validation']['roc_auc'],-results[n]['input_dimensions']))
    torch.save(bundles[selected],out/'model.pt')
    for name,scores in all_predictions.items():results[name]['diagnostic_test']=metrics(labels[test],scores,results[name]['threshold'])
    predictions=m.loc[test,['video_id','label']].copy()
    for name,scores in all_predictions.items():predictions[name]=scores
    predictions.to_csv(out/'diagnostic_test_predictions.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    baseline=json.loads((ROOT/'artifacts/metrics.json').read_text())
    result=dict(selected_model=selected,variants=results,original_baseline=baseline['test'][baseline['selected_model']],
        original_baseline_validation=baseline['validation'][baseline['selected_model']],
        deployment_recommendation='retain baseline; new MIL is experimental and has not surpassed baseline validation F1',
        feature_dimension=DESCRIPTOR_DIM,frame_feature_dimension=DESCRIPTOR_DIM//3,
        training_observed_videos=len(fit),training_empty_videos=len(train)-len(fit),
        note='Old test has already been inspected; all v2 test comparisons are exploratory diagnostics, not independent final accuracy. No person-level ground truth or calibration. External news clips excluded.',
        original_model_overwritten=False,approved_for_real_world=False)
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'feature_schema.json').write_text(json.dumps(DESCRIPTOR_NAMES,indent=2))
    print('Selected',selected);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
