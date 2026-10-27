"""Keep user-supplied external videos out of training and threshold tuning."""
from pathlib import Path
import sys,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.extract_behavior import extract
from src.behavior_model import BehaviorScorer
from src.predict import predict_video
from src.live import stream_video

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dense',action='store_true');a=parser.parse_args()
    sources=json.loads((ROOT/'data/external_test/user_parking_sources.json').read_text())
    scorer=BehaviorScorer();rows=[]
    for source in sources:
        path=ROOT/source['local_path'];baseline=predict_video(path)
        x,observations,info=extract(path)
        scores=scorer.predict(x);score=float(scores.max()) if len(scores) else None
        row=dict(source=source,baseline_score=baseline['video_score'],baseline_threshold=baseline['threshold'],
            behavior_score=score,behavior_threshold=scorer.threshold,instances=len(x),coverage=info,
            labeling='Presumed normal from user; source/scene reviewed, no independently verified authorization labels',
            sampling='6 sampled windows, not a complete event-level audit',used_for_training=False)
        if a.dense:
            output=ROOT/'runtime/annotated'/f"{source['id']}_behavior.mp4"
            for update in stream_video(path,output,behavior=True,use_behavior_alarm=True,capture_evidence=False):
                if update['kind']=='complete':
                    report=update['report']
                    row.update(dense_review_frames=report['review_frames'],dense_processed_frames=report['processed_frames'],
                               dense_peak_behavior_score=report['peak_behavior_score'],annotated_video=str(output))
        rows.append(row)
        out=ROOT/'artifacts/behavior_v2/external_evaluation.json'
        out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(source['id'],'baseline',row['baseline_score'],'MIL',score,'instances',len(x),'review_frames',row.get('dense_review_frames'),flush=True)

if __name__=='__main__':main()
