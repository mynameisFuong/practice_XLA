"""Frozen ResNet50 appearance encoder, inspired by the reviewed Dilaniu notebook.

Uses official Torchvision ImageNet V2 weights. This is not a pretrained theft
classifier. Features describe a person crop in context, never facial identity.
"""
from pathlib import Path
import hashlib,json,os
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ENCODER_VERSION='resnet50-imagenet-v2-person-context-v1'
ENCODER_DIM=2048

def person_crop(frame,bbox,padding=.2):
    h,w=frame.shape[:2];x1,y1,x2,y2=map(float,bbox)
    if not np.isfinite([x1,y1,x2,y2]).all() or x2<=x1 or y2<=y1:raise ValueError('Invalid person box')
    px=(x2-x1)*padding;py=(y2-y1)*padding
    left=max(0,int(np.floor(x1-px)));right=min(w,int(np.ceil(x2+px)))
    top=max(0,int(np.floor(y1-py)));bottom=min(h,int(np.ceil(y2+py)))
    if right<=left or bottom<=top:raise ValueError('Person box outside image')
    return frame[top:bottom,left:right]

class ResNetPersonEncoder:
    def __init__(self):
        import torch
        import requests
        from torchvision.models import resnet50,ResNet50_Weights
        weights=ResNet50_Weights.IMAGENET1K_V2
        checkpoint=ROOT/'runtime/models/resnet50-11ad3fa6.pth'
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        if not checkpoint.exists():
            temp=checkpoint.with_suffix('.part')
            with requests.get(weights.url,stream=True,timeout=(30,120)) as response:
                response.raise_for_status()
                with temp.open('wb') as f:
                    for chunk in response.iter_content(1024*1024):f.write(chunk)
            digest=hashlib.sha256(temp.read_bytes()).hexdigest()
            if not digest.startswith('11ad3fa6'):raise ValueError('Official ResNet checkpoint hash mismatch')
            os.replace(temp,checkpoint)
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if not digest.startswith('11ad3fa6'):raise ValueError('Cached ResNet checkpoint hash mismatch')
        self.torch=torch;model=resnet50(weights=None)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        self.model=torch.nn.Sequential(*list(model.children())[:-1]).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.preprocess=weights.transforms()
        self.model_source=dict(version=ENCODER_VERSION,url=weights.url,weights='IMAGENET1K_V2',
            sha256=digest,
            purpose='Frozen person-context embeddings; head must be trained separately; not face recognition')

    def encode(self,crops,batch_size=16):
        from PIL import Image
        if not crops:return np.empty((0,ENCODER_DIM),np.float32)
        output=[]
        for start in range(0,len(crops),batch_size):
            tensor=self.torch.stack([self.preprocess(Image.fromarray(cv2.cvtColor(c,cv2.COLOR_BGR2RGB))) for c in crops[start:start+batch_size]])
            with self.torch.inference_mode():features=self.model(tensor).flatten(1)
            output.append(features.numpy())
        return np.concatenate(output).astype(np.float32)
