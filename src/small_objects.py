"""Overlapping crops and class-wise duplicate suppression in source coordinates."""
import cv2
import numpy as np

def detection_regions(frame):
    h,w=frame.shape[:2]
    yield frame,0,0,1280
    cw=max(1,int(np.ceil(w*.65)));ch=max(1,int(np.ceil(h*.65)))
    for y in sorted({0,h-ch}):
        for x in sorted({0,w-cw}):
            yield frame[y:y+ch,x:x+cw],x,y,640

def merge_candidates(candidates,shape,threshold=.15,iou=.45):
    """Rows: x1,y1,x2,y2,confidence,class_id. Never merge different classes."""
    a=np.asarray(candidates,dtype=np.float32).reshape(-1,6).copy()
    if not len(a):return a
    h,w=shape[:2]
    a[:,[0,2]]=np.clip(a[:,[0,2]],0,w);a[:,[1,3]]=np.clip(a[:,[1,3]],0,h)
    a=a[np.isfinite(a).all(1)&(a[:,2]>a[:,0])&(a[:,3]>a[:,1])&(a[:,4]>=threshold)]
    keep=[]
    for cls in np.unique(a[:,5]):
        indices=np.flatnonzero(a[:,5]==cls)
        boxes=a[indices,:4].copy();boxes[:,2:]-=boxes[:,:2]
        selected=cv2.dnn.NMSBoxes(boxes.tolist(),a[indices,4].tolist(),threshold,iou)
        keep.extend(indices[np.asarray(selected,dtype=int).reshape(-1)].tolist())
    return a[sorted(keep,key=lambda i:-a[i,4])]
