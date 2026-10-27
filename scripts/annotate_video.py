from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.live import stream_video

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--out',default='runtime/annotated/result.mp4')
    p.add_argument('--fps',type=int,default=8)
    p.add_argument('--detector-weights',help='Đường dẫn checkpoint YOLO detect, lớp 0=person và 3=motorcycle')
    p.add_argument('--detector-profile',choices=['fast','small_people'],default='fast')
    p.add_argument('--preprocessing',choices=['off','auto','low_light'],default='off')
    p.add_argument('--denoise',action='store_true');p.add_argument('--sharpen',action='store_true')
    p.add_argument('--behavior',action='store_true');p.add_argument('--mil-alarm',action='store_true')
    p.add_argument('--no-evidence',action='store_true');p.add_argument('--camera',default='HAM_B1_CAM01')
    p.add_argument('--backend',choices=['scene','pose_mil','resnet_mil']);a=p.parse_args()
    for update in stream_video(a.video,a.out,target_fps=a.fps,behavior=a.behavior,use_behavior_alarm=a.mil_alarm,
                               camera_id=a.camera,capture_evidence=not a.no_evidence,risk_backend=a.backend,
                               detector_weights=a.detector_weights,preprocessing=a.preprocessing,denoise=a.denoise,sharpen=a.sharpen,
                               detector_profile=a.detector_profile):
        if update['kind']=='complete':
            r=update['report'];print({k:v for k,v in r.items() if k not in ['windows','frames']})
