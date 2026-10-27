from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.live import stream_video

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--out',default='runtime/annotated/result.mp4')
    p.add_argument('--fps',type=int,default=8)
    p.add_argument('--behavior',action='store_true');p.add_argument('--mil-alarm',action='store_true');a=p.parse_args()
    for update in stream_video(a.video,a.out,target_fps=a.fps,behavior=a.behavior,use_behavior_alarm=a.mil_alarm):
        if update['kind']=='complete':
            r=update['report'];print({k:v for k,v in r.items() if k not in ['windows','frames']})
