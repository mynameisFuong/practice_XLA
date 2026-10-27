from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score,average_precision_score,accuracy_score,
    precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay,roc_curve)
from src.features import extract_video,aggregate,VERSION

ROOT=Path(__file__).resolve().parents[1]

def evaluate(y,p,threshold):
    pred=np.asarray(p)>=threshold
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
                recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
                roc_auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),
                confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='data/manifest.csv')
    args=parser.parse_args()
    t0=time.perf_counter()
    manifest=pd.read_csv(ROOT/args.manifest)
    if manifest.video_id.duplicated().any():
        raise ValueError('Video overlap across splits')
    if manifest.sha256.duplicated().any():
        raise ValueError('Exact duplicate video content')
    for split in ['train','val','test']:
        if set(manifest.loc[manifest.split==split,'label'])!={0,1}:
            raise ValueError(f'{split} must contain both labels')
    cache=ROOT/'data/features'; cache.mkdir(exist_ok=True)
    clips=[]; metadata=[]
    for r in manifest.to_dict('records'):
        target=cache/f"{r['video_id']}_{str(r['sha256'])[:12]}_{VERSION}.npz"
        if target.exists():
            z=np.load(target); x=z['x']; records=json.loads(str(z['records']))
        else:
            x,records,info=extract_video(ROOT/r['local_path'])
            np.savez_compressed(target,x=x,records=json.dumps(records),info=json.dumps(info))
        clips.append(x); metadata.append(records)
        print('Features',r['video_id'],x.shape,flush=True)
    # Video-level descriptor avoids assigning positive labels to innocent windows.
    # Each video has one training example: mean, std and p90 of clip descriptors.
    X=np.stack([np.r_[x.mean(0),x.std(0),np.quantile(x,.9,axis=0)] for x in clips])
    y=manifest.label.to_numpy()
    tr=(manifest.split=='train').to_numpy(); va=(manifest.split=='val').to_numpy(); te=(manifest.split=='test').to_numpy()
    models=dict(dummy=DummyClassifier(strategy='prior'),
                logistic=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=3000,random_state=42)),
                extra_trees=ExtraTreesClassifier(n_estimators=300,max_depth=6,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1))
    out=ROOT/'artifacts';out.mkdir(exist_ok=True)
    validation={}; predictions={}
    # Choose threshold and model exclusively using validation videos.
    for name,model in models.items():
        model.fit(X[tr],y[tr])
        p=model.predict_proba(X[va])[:,1]
        thresholds=np.linspace(.1,.9,81)
        threshold=float(max(thresholds,key=lambda th:(f1_score(y[va],p>=th,zero_division=0),-abs(th-.5))))
        validation[name]=dict(threshold=threshold,**evaluate(y[va],p,threshold))
        predictions[name]=model.predict_proba(X[te])[:,1]
    # A dummy comparator is never deployed. Ties favor the simpler logistic model.
    selected=max(['logistic','extra_trees'],key=lambda n:(validation[n]['f1'],validation[n]['roc_auc'],n=='logistic'))
    threshold=validation[selected]['threshold']
    test={n:evaluate(y[te],p,validation[n]['threshold']) for n,p in predictions.items()}
    bundle=dict(model=models[selected],name=selected,threshold=threshold,version=VERSION,
                max_windows=12,aggregation='mean_std_p90',feature_dimensions=int(X.shape[1]))
    joblib.dump(bundle,out/'model.joblib')
    results=dict(selected_model=selected,threshold=threshold,feature_dimensions=int(X.shape[1]),
                 validation=validation,test=test,split_counts=manifest.groupby(['split','label']).size().to_dict(),
                 training_seconds=round(time.perf_counter()-t0,3),
                 limitations=['Video-level stealing vs normal only; no motorcycle-specific ground truth',
                              'Small size-filtered subset; camera-disjointness not verified',
                              'Window scores are exploratory; no temporal localization evaluation'])
    results['split_counts']={f'{k[0]}_label{k[1]}':int(v) for k,v in results['split_counts'].items()}
    (out/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    frame=manifest.loc[te,['video_id','label','local_path']].copy()
    for name,p in predictions.items(): frame[name+'_score']=p
    frame['predicted']=(predictions[selected]>=threshold).astype(int)
    frame.to_csv(out/'test_predictions.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    ConfusionMatrixDisplay(np.array(test[selected]['confusion_matrix']),display_labels=['Normal','Stealing']).plot(ax=axes[0],colorbar=False)
    for name,p in predictions.items():
        fpr,tpr,_=roc_curve(y[te],p); axes[1].plot(fpr,tpr,label=f"{name} AUC={test[name]['roc_auc']:.3f}")
    axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set(xlabel='False positive rate',ylabel='True positive rate');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'evaluation.png',dpi=160);plt.close(fig)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
