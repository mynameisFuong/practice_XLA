from pathlib import Path
import time
import joblib
import numpy as np
from .features import extract_video, VERSION
from .events import make_events

ROOT=Path(__file__).resolve().parents[1]

def predict_video(path, model_path=None, dense=False):
    start=time.perf_counter()
    # Load only the trusted model supplied with the project.
    bundle=joblib.load(model_path or ROOT/'artifacts/model.joblib')
    if bundle['version'] != VERSION:
        raise ValueError('Model và bộ trích xuất đặc trưng khác phiên bản')
    # Video score always uses the exact same 12-window representation as training.
    x,records,info=extract_video(path,max_windows=bundle['max_windows'])
    representation=np.r_[x.mean(0),x.std(0),np.quantile(x,.9,axis=0)]
    video_score=float(bundle['model'].predict_proba(representation.reshape(1,-1))[0,1])
    if dense:
        x,records,info=extract_video(path,max_windows=0)
    # Dense timeline uses trailing contexts of up to 12 clips (up to 48 seconds).
    # Same descriptor as training; temporal sampling differs, so localization
    # remains exploratory and is never reported as measured theft detection.
    contexts=[x[max(0,i-11):i+1] for i in range(len(x))] if dense else [x]*len(x)
    local=np.stack([np.r_[c.mean(0),c.std(0),np.quantile(c,.9,axis=0)] for c in contexts])
    scores=bundle['model'].predict_proba(local)[:,1]
    records=[dict(r,score=float(s)) for r,s in zip(records,scores)]
    usable=sum(r['usable'] for r in records)
    quality_ok=usable/len(records)>=.5
    return dict(video=Path(path).name,model=bundle['name'],threshold=bundle['threshold'],
                video_score=video_score,quality_ok=quality_ok,
                decision=('insufficient_quality' if not quality_ok else 'review' if video_score>=bundle['threshold'] else 'low_score'),
                info=info,windows=records,events=make_events(records,bundle['threshold']) if dense else [],
                mode='dense' if dense else 'sampled',
                warning='Điểm nghi vấn chưa hiệu chuẩn; mốc thời gian chỉ để xem xét, chưa được kiểm định định vị hành vi.',
                elapsed_seconds=round(time.perf_counter()-start,3))
