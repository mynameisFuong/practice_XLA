"""Head trained in this project over frozen ResNet embeddings + optional pose."""
from pathlib import Path
import numpy as np
from .visual_backbone import ENCODER_VERSION,ENCODER_DIM
from .behavior_features import DESCRIPTOR_DIM

ROOT=Path(__file__).resolve().parents[1]

class VisualBehaviorScorer:
    def __init__(self,path=None):
        import torch
        self.torch=torch
        self.bundle=torch.load(path or ROOT/'artifacts/resnet_v3/model.pt',map_location='cpu',weights_only=True)
        self.input_dim=ENCODER_DIM+DESCRIPTOR_DIM
        if self.bundle['version']!=ENCODER_VERSION or self.bundle['input_dim']!=self.input_dim:raise ValueError('Visual feature version mismatch')
        self.indices=np.array(self.bundle['indices'],int)
        self.mean=np.array(self.bundle['mean'],np.float32);self.scale=np.array(self.bundle['scale'],np.float32)
        self.model=torch.nn.Linear(len(self.indices),1);self.model.load_state_dict(self.bundle['state']);self.model.eval()
        self.threshold=float(self.bundle['threshold'])

    def predict(self,features):
        x=np.asarray(features,np.float32).reshape(-1,self.input_dim)
        if not len(x):return np.array([],np.float32)
        z=np.clip((x[:,self.indices]-self.mean)/self.scale,-5,5)
        with self.torch.inference_mode():return self.torch.sigmoid(self.model(self.torch.from_numpy(z))).numpy().ravel()
