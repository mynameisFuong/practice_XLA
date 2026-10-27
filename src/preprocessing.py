"""Conservative, optional enhancement for detection/pose, without geometry changes.

Original frames must remain the source for learned feature heads and evidence.
Filters cannot recover missing detail, remove occlusion or fix severe blur.
"""
import cv2
import numpy as np

MODES={'off','auto','low_light'}

def enhance_frame(frame,mode='off',denoise=False,sharpen=False):
    if mode not in MODES:raise ValueError('Unknown preprocessing mode: '+str(mode))
    if frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3:
        raise ValueError('Expected uint8 BGR frame')
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    brightness=float(gray.mean());contrast=float(gray.std())
    out=frame.copy();applied=[]
    if mode=='low_light' or (mode=='auto' and (brightness<65 or contrast<30)):
        lab=cv2.cvtColor(out,cv2.COLOR_BGR2LAB)
        lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0])
        out=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR);applied.append('clahe_luminance')
        if brightness<65:
            gamma=.75
            lut=np.round(255*(np.arange(256)/255.)**gamma).astype(np.uint8)
            out=cv2.LUT(out,lut);applied.append('gamma_0.75')
    if denoise:
        out=cv2.bilateralFilter(out,5,25,25);applied.append('bilateral')
    if sharpen:
        out=cv2.addWeighted(out,1.25,cv2.GaussianBlur(out,(0,0),1.),-.25,0)
        applied.append('mild_unsharp')
    return out,dict(mode=mode,denoise=bool(denoise),sharpen=bool(sharpen),
                    source_brightness=brightness,source_contrast=contrast,applied=applied)
