"""Compare filter configurations on the same file; no accuracy without labels."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.live import stream_video

def main():
    p=argparse.ArgumentParser()
    p.add_argument('video');p.add_argument('--out-dir',default='runtime/filter_comparison')
    p.add_argument('--detector-weights')
    p.add_argument('--modes',nargs='+',choices=['off','auto','low_light'],default=['off','auto'])
    p.add_argument('--denoise',action='store_true');p.add_argument('--sharpen',action='store_true')
    args=p.parse_args();folder=Path(args.out_dir);folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for mode in args.modes:
        # The reference always has all filters off.
        for update in stream_video(args.video,folder/f'{Path(args.video).stem}_{mode}.mp4',
                detector_weights=args.detector_weights,behavior=True,preprocessing=mode,
                denoise=args.denoise and mode!='off',sharpen=args.sharpen and mode!='off',
                capture_evidence=False,risk_backend='scene'):
            if update['kind']!='complete':continue
            r=update['report'];frames=r['frames'];people=[person for f in frames for person in f['people']]
            summary={key:r[key] for key in ['video','detector','detector_metadata','preprocessing','risk_policy',
                'processed_frames','elapsed_seconds','review_frames','annotated_video']}
            summary.update(person_observations=len(people),
                risk_counts={level:sum(p['risk_level']==level for p in people) for level in ['safe','suspicious','danger','unknown']},
                crouching_observations=sum(bool(p.get('evidence',{}).get('crouching_proxy')) for p in people),
                enhanced_frames=sum(bool(f['preprocessing']['applied']) for f in frames))
            results.append(summary)
            print(mode,summary['risk_counts'],flush=True)
    report=dict(results=results,warning='Unlabeled comparison. Counts are person-frame observations, not unique people, events or accuracy.')
    (folder/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
