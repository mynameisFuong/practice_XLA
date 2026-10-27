"""Shared preprocessing and temporal descriptors for training and inference."""
from pathlib import Path
import cv2
import numpy as np

VERSION = 'hog-hsv-lbp-flow-v1'
WIDTH, HEIGHT, SAMPLE_FPS, WINDOW = 128, 96, 4, 4.0

def preprocess(frame):
    if frame is None or frame.size == 0:
        raise ValueError('Khung hình trống')
    h, w = frame.shape[:2]
    scale = min(WIDTH / w, HEIGHT / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    # Crop padding before feature extraction; masks exclude letterbox from histograms.
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    x, y = (WIDTH-nw)//2, (HEIGHT-nh)//2
    canvas[y:y+nh, x:x+nw] = resized
    mask = np.zeros((HEIGHT, WIDTH), np.uint8); mask[y:y+nh,x:x+nw] = 255
    gray_raw = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    active = gray_raw[mask > 0]
    blur = float(cv2.Laplacian(gray_raw[y:y+nh,x:x+nw], cv2.CV_64F).var())
    quality = dict(brightness=float(active.mean()), blur=blur,
                   dark_fraction=float((active < 20).mean()),
                   saturated_fraction=float((active > 245).mean()))
    clean = cv2.GaussianBlur(gray_raw, (3,3), 0.5)
    # CLAHE only within content, so padding does not drive contrast statistics.
    gray = np.zeros_like(clean)
    gray[y:y+nh,x:x+nw] = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(clean[y:y+nh,x:x+nw])
    return canvas, gray, mask, quality

def frame_descriptor(bgr, gray, mask):
    # Spatial gradient histograms in a 4x3 grid; HOG style cell normalization.
    gx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    gy = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    mag, angle = cv2.cartToPolar(gx,gy,angleInDegrees=True)
    mag[mask == 0] = 0
    histograms = []
    for yy in range(0,HEIGHT,32):
        for xx in range(0,WIDTH,32):
            bins = ((angle[yy:yy+32,xx:xx+32] % 180)/20).astype(int)
            hist = np.bincount(bins.ravel(), weights=mag[yy:yy+32,xx:xx+32].ravel(),minlength=9)
            histograms.extend(hist/(np.linalg.norm(hist)+1e-6))
    hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    color = []
    for ch, bins, extent in [(0,12,180),(1,8,256),(2,8,256)]:
        hist = cv2.calcHist([hsv],[ch],mask,[bins],[0,extent]).ravel()
        color.extend(hist/(hist.sum()+1e-6))
    center = gray[1:-1,1:-1]
    lbp = np.zeros_like(center)
    for i,(dy,dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
        lbp |= ((gray[1+dy:HEIGHT-1+dy,1+dx:WIDTH-1+dx]>=center).astype(np.uint8)<<i)
    valid = mask[1:-1,1:-1] > 0
    texture = np.bincount((lbp[valid]//16),minlength=16).astype(float)
    texture /= texture.sum()+1e-6
    # 108 gradient + 28 color + 16 texture = 152 per frame.
    return np.asarray([*histograms,*color,*texture],np.float32)

def describe_window(frames):
    appearance, motions, qualities = [], [], []
    previous = None
    for frame in frames:
        bgr, gray, mask, quality = preprocess(frame)
        appearance.append(frame_descriptor(bgr,gray,mask)); qualities.append(quality)
        if previous is not None:
            flow = cv2.calcOpticalFlowFarneback(previous,gray,None,0.5,2,15,2,5,1.1,0)
            # Subtract median flow to reduce small global camera translations.
            flow -= np.median(flow[mask>0],axis=0)
            mag, ang = cv2.cartToPolar(flow[...,0],flow[...,1])
            selected = mag[mask>0]
            direction = np.histogram(ang[mask>0],bins=8,range=(0,2*np.pi),weights=selected)[0]
            direction /= direction.sum()+1e-6
            diff = cv2.absdiff(previous,gray)[mask>0]/255.0
            motions.append([float(selected.mean()),float(np.quantile(selected,.9)),
                            float((selected>1).mean()),float(diff.mean()),*direction.tolist()])
        previous = gray
    if not appearance:
        raise ValueError('Không đọc được khung hình nào')
    a = np.asarray(appearance)
    m = np.asarray(motions) if motions else np.zeros((1,12))
    # 152*2 + 12*2 = 328 dimensions. Quality is a gate, not a predictive shortcut.
    feature = np.r_[a.mean(0),a.std(0),m.mean(0),m.std(0)].astype(np.float32)
    quality = {k:float(np.mean([q[k] for q in qualities])) for k in qualities[0]}
    quality['valid_frames'] = len(frames)
    quality['usable'] = len(frames)>=4 and quality['brightness']>=12 and quality['blur']>=8
    return feature,quality

def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Không mở được video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS); count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width,height = cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    if not np.isfinite(fps) or fps<=0 or count<=0:
        raise ValueError('FPS hoặc thời lượng không hợp lệ; hãy chuyển sang MP4 CFR trước')
    return dict(fps=float(fps),frames=int(count),duration=float(count/fps),width=int(width),height=int(height))

def extract_video(path, max_windows=12):
    """Uniformly spaced 4 s clips; max_windows=0 means dense nonoverlapping clips.

    OpenCV frame-index timing assumes constant frame rate (CFR). VFR must first
    be converted with ffmpeg. No interpolation of missing frames is invented.
    """
    info = video_info(path)
    duration = info['duration']
    dense = np.arange(0,duration,WINDOW)
    starts = dense if not max_windows or len(dense)<=max_windows else np.linspace(0,max(0,duration-WINDOW),max_windows)
    cap = cv2.VideoCapture(str(path))
    features, records = [], []
    try:
        for start in starts:
            frames=[]; indices=[]
            for t in np.arange(start,min(start+WINDOW,duration),1/SAMPLE_FPS):
                index = min(int(round(t*info['fps'])),info['frames']-1)
                if not indices or indices[-1]!=index:indices.append(index)
            # Seek once per clip, decode intervening frames sequentially. Seeking
            # once per sampled frame repeatedly decodes GOPs and is much slower.
            if indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES,indices[0])
                wanted=set(indices)
                for index in range(indices[0],indices[-1]+1):
                    if not cap.grab():break
                    if index in wanted:
                        ok,frame=cap.retrieve()
                        if ok:frames.append(frame)
            if frames:
                feat,q=describe_window(frames)
                q['low_source_fps'] = info['fps'] < SAMPLE_FPS
                q['usable'] = bool(q['usable'] and not q['low_source_fps'])
                records.append(dict(start=float(start),end=float(min(start+WINDOW,duration)),**q))
                features.append(feat)
    finally:
        cap.release()
    if not features:
        raise ValueError('Video lỗi hoặc không có khung hình giải mã được')
    return np.stack(features),records,info

def aggregate(scores):
    return float(np.quantile(scores,.9))
