"""External diagnostic only; user parking clips are never used for fitting."""
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from src.visual_backbone import ResNetPersonEncoder,person_crop
from src.visual_model import VisualBehaviorScorer
from src.predict import predict_video
from src.live import stream_video
from scripts.extract_behavior import extract

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dense',action='store_true');a=parser.parse_args()
    torch.set_num_threads(2);encoder=ResNetPersonEncoder();model=VisualBehaviorScorer();rows=[]
    sources=json.loads((ROOT/'data/external_test/user_parking_sources.json').read_text())
    for source in sources:
        path=ROOT/source['local_path'];pose,records,info=extract(path)
        cap=cv2.VideoCapture(str(path));crops=[]
        try:
            for record in records:
                cap.set(cv2.CAP_PROP_POS_FRAMES,round(record['timestamp']*info['fps']));ok,frame=cap.read()
                if not ok:raise ValueError('Cannot decode feature-aligned source frame')
                scale=min(1,960/frame.shape[1]);frame=cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale)))
                crops.append(person_crop(frame,record['bbox']))
        finally:cap.release()
        features=np.concatenate([encoder.encode(crops),pose],axis=1);scores=model.predict(features);baseline=predict_video(path)
        row=dict(id=source['id'],title=source['title'],url=source['url'],baseline_score=baseline['video_score'],
                 resnet_score=float(scores.max()) if len(scores) else None,threshold=model.threshold,instances=len(features),
                 label='presumed_normal',used_for_training=False,sampling='6 sampled windows; not event-level accuracy')
        if a.dense:
            output=ROOT/'runtime/annotated'/f"{source['id']}_resnet_three_levels.mp4"
            for update in stream_video(path,output,behavior=True,risk_backend='resnet_mil',capture_evidence=False):
                if update['kind']=='complete':
                    r=update['report'];row['risk_person_frames']={level:sum(f['risk_counts'][level] for f in r['frames']) for level in ['safe','suspicious','danger','unknown']}
                    row['risk_policy']=r['risk_policy']
        rows.append(row)
        (ROOT/'artifacts/resnet_v3/external_evaluation.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(row,flush=True)
