"""Add a frozen ResNet descriptor to the SAME person windows as pose/HOI."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pandas as pd
import torch
from src.visual_backbone import ResNetPersonEncoder,person_crop,ENCODER_VERSION,ENCODER_DIM
from src.behavior_features import BEHAVIOR_VERSION,DESCRIPTOR_DIM

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    torch.set_num_threads(2)
    encoder=ResNetPersonEncoder()
    folder=ROOT/'data/visual_features';folder.mkdir(exist_ok=True)
    out=ROOT/'artifacts/resnet_v3';out.mkdir(exist_ok=True)
    (out/'encoder_source.json').write_text(json.dumps(encoder.model_source,indent=2))
    t0=time.perf_counter()
    for i,row in enumerate(pd.read_csv(ROOT/'data/manifest.csv').to_dict('records')):
        key=f"{row['video_id']}_{row['sha256'][:12]}"
        target=folder/f'{key}_{ENCODER_VERSION}.npz'
        if target.exists():print('Cached',row['video_id'],flush=True);continue
        cache=ROOT/'data/behavior_features'/f'{key}_{BEHAVIOR_VERSION}_w6_fps2.npz'
        z=np.load(cache);records=json.loads(str(z['records']));info=json.loads(str(z['info']))
        cap=cv2.VideoCapture(str(ROOT/row['local_path']));embeddings=[]
        try:
            for record in records:
                cap.set(cv2.CAP_PROP_POS_FRAMES,round(record['timestamp']*info['fps']))
                ok,frame=cap.read()
                if not ok:raise ValueError('Failed to decode the feature-aligned person frame')
                scale=min(1,960/frame.shape[1])
                frame=cv2.resize(frame,(max(2,round(frame.shape[1]*scale)),max(2,round(frame.shape[0]*scale))))
                embeddings.append(person_crop(frame,record['bbox']))
            visual=encoder.encode(embeddings)
        finally:cap.release()
        features=np.concatenate([visual,z['x']],axis=1) if len(visual) else np.empty((0,ENCODER_DIM+DESCRIPTOR_DIM),np.float32)
        np.savez_compressed(target,x=features,records=json.dumps(records),version=ENCODER_VERSION,visual_dim=ENCODER_DIM)
        print(f"{i+1}/63 {row['video_id']} persons={len(visual)} elapsed={time.perf_counter()-t0:.1f}s",flush=True)
