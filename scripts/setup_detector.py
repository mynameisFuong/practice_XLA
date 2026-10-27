"""Download official pretrained YOLOv8n COCO weights for person/bike boxes."""
from pathlib import Path
import hashlib,json
import requests

ROOT=Path(__file__).resolve().parents[1]
URL='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt'

if __name__=='__main__':
    for filename,metadata,purpose in [('yolov8n.pt','detector_source.json','COCO object detection'),
                                       ('yolov8n-pose.pt','pose_source.json','COCO 17-keypoint pose')]:
        target=ROOT/'artifacts'/filename;target.parent.mkdir(exist_ok=True)
        url=URL.replace('yolov8n.pt',filename)
        if not target.exists():
            with requests.get(url,stream=True,timeout=(30,120)) as r:
                r.raise_for_status()
                temp=target.with_suffix('.pt.part')
                with temp.open('wb') as f:
                    for chunk in r.iter_content(1024*1024):f.write(chunk)
                temp.replace(target)
        (ROOT/'artifacts'/metadata).write_text(json.dumps(dict(source=url,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            purpose=purpose+' only; not trained to identify theft',
            upstream='https://docs.ultralytics.com/models/yolov8/'),indent=2))
        print('Model ready:',target)
