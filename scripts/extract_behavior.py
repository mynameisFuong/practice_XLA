"""Extract person-window descriptors, preserving original video splits."""
from pathlib import Path
import sys,argparse,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pandas as pd
from src.features import video_info
from src.live import PeopleMotorcycleDetector
from src.behavior_features import (PoseEstimator,InteractionFeatures,BEHAVIOR_VERSION,DESCRIPTOR_DIM,DESCRIPTOR_NAMES)

ROOT=Path(__file__).resolve().parents[1]

def extract(path,max_windows=6,fps=2):
    info=video_info(path)
    if info['fps']<fps:raise ValueError('Source FPS too low')
    dense=np.arange(0,info['duration'],4.)
    starts=dense if len(dense)<=max_windows else np.linspace(0,max(0,info['duration']-4),max_windows)
    detector=PeopleMotorcycleDetector();pose=PoseEstimator(ROOT/'artifacts/yolov8n-pose.pt')
    cap=cv2.VideoCapture(str(path));features=[];records=[];frames=0;pose_frames=0
    try:
        for start in starts:
            engine=InteractionFeatures();latest={}
            indices=sorted(set(min(info['frames']-1,round(t*info['fps'])) for t in np.arange(start,min(start+4,info['duration']),1/fps)))
            if not indices:continue
            cap.set(cv2.CAP_PROP_POS_FRAMES,indices[0]);wanted=set(indices)
            for index in range(indices[0],indices[-1]+1):
                if not cap.grab():break
                if index not in wanted:continue
                ok,frame=cap.retrieve()
                if not ok:continue
                scale=min(1,960/frame.shape[1]);frame=cv2.resize(frame,(max(2,round(frame.shape[1]*scale)),max(2,round(frame.shape[0]*scale))))
                detections=pose.attach(frame,detector.detect(frame))
                observations,cut=engine.update(frame,detections,index/info['fps']);frames+=1
                pose_frames+=any(o['pose_coverage']>.25 for o in observations)
                if cut:latest.clear()
                for o in observations:
                    if o['observations']>=4 and o['observed_seconds']>=1.4:
                        latest[o['track_id']]=o
            for o in latest.values():
                features.append(o['feature'])
                records.append({k:v for k,v in o.items() if k not in ['feature','keypoints']})
    finally:cap.release()
    x=np.stack(features) if features else np.empty((0,DESCRIPTOR_DIM),np.float32)
    return x,records,dict(**info,sampled_frames=frames,frames_with_pose=int(pose_frames))

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='data/manifest.csv');p.add_argument('--max-windows',type=int,default=6);p.add_argument('--fps',type=float,default=2);a=p.parse_args()
    manifest=pd.read_csv(ROOT/a.manifest);folder=ROOT/'data/behavior_features';folder.mkdir(exist_ok=True)
    t=time.perf_counter()
    for i,r in enumerate(manifest.to_dict('records')):
        cache=folder/f"{r['video_id']}_{r['sha256'][:12]}_{BEHAVIOR_VERSION}_w{a.max_windows}_fps{a.fps:g}.npz"
        if cache.exists():print('Cached',i+1,r['video_id'],flush=True);continue
        x,records,info=extract(ROOT/r['local_path'],a.max_windows,a.fps)
        np.savez_compressed(cache,x=x,records=json.dumps(records),info=json.dumps(info),names=np.array(DESCRIPTOR_NAMES),version=BEHAVIOR_VERSION)
        print(f"{i+1}/{len(manifest)} {r['video_id']} instances={len(x)} pose_frames={info['frames_with_pose']}/{info['sampled_frames']} elapsed={time.perf_counter()-t:.1f}s",flush=True)

if __name__=='__main__':main()
