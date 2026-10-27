from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.predict import predict_video

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('video');parser.add_argument('--dense',action='store_true')
    parser.add_argument('--out',default='artifacts/prediction.json')
    args=parser.parse_args()
    result=predict_video(args.video,dense=args.dense)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='windows'},ensure_ascii=False,indent=2))
