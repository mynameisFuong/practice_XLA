"""Weakly supervised max-MIL. Instance scores have no person-level calibration."""
from pathlib import Path
import numpy as np
from .behavior_features import BEHAVIOR_VERSION,DESCRIPTOR_DIM

ROOT=Path(__file__).resolve().parents[1]

class BehaviorScorer:
    def __init__(self,path=None):
        import torch
        self.torch=torch
        self.bundle=torch.load(path or ROOT/'artifacts/behavior_v2/model.pt',map_location='cpu',weights_only=True)
        if self.bundle['version']!=BEHAVIOR_VERSION or self.bundle['input_dim']!=DESCRIPTOR_DIM:raise ValueError('Behavior feature version mismatch')
        self.indices=np.asarray(self.bundle['indices'],int)
        self.mean=np.asarray(self.bundle['mean'],np.float32);self.scale=np.asarray(self.bundle['scale'],np.float32)
        self.model=torch.nn.Linear(len(self.indices),1);self.model.load_state_dict(self.bundle['state']);self.model.eval()
        self.threshold=float(self.bundle['threshold'])

    def predict(self,features):
        x=np.asarray(features,np.float32).reshape(-1,DESCRIPTOR_DIM)
        if not len(x):return np.array([],np.float32)
        z=np.clip((x[:,self.indices]-self.mean)/self.scale,-5,5)
        with self.torch.no_grad():return self.torch.sigmoid(self.model(self.torch.from_numpy(z))).numpy().ravel()
