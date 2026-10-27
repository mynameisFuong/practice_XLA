"""Causal file-stream demo: detect people/bikes, track, score past context.

The existing classifier is scene-level. No person-level theft probability is
invented. Red is only a heuristic review cue: usable scene score above threshold
AND a tracked person staying near a detected motorcycle for at least 2 seconds.
"""
from collections import deque
from pathlib import Path
import json,time,tempfile
import cv2
import joblib
import numpy as np
from .features import describe_window,video_info,VERSION,WINDOW,SAMPLE_FPS
from .media import encode_h264
from .behavior_features import PoseEstimator,InteractionFeatures,draw_pose

ROOT=Path(__file__).resolve().parents[1]

class CausalSceneScorer:
    def __init__(self,bundle=None):
        self.bundle=bundle or joblib.load(ROOT/'artifacts/model.joblib')
        if self.bundle['version']!=VERSION:raise ValueError('Sai phiên bản đặc trưng')
        self.context=deque(maxlen=12)
        self.frames=[];self.window_start=0.0;self.next_sample=0.0
        self.score=None;self.quality=None;self.records=[]
        self.score_updated_at=None

    def _close_window(self,end,updated_at):
        if not self.frames:return
        feat,quality=describe_window(self.frames)
        self.context.append(feat)
        x=np.stack(self.context)
        representation=np.r_[x.mean(0),x.std(0),np.quantile(x,.9,axis=0)]
        self.score=float(self.bundle['model'].predict_proba(representation.reshape(1,-1))[0,1])
        self.quality=quality;self.score_updated_at=float(updated_at)
        self.records.append(dict(start=self.window_start,end=float(end),
                                 available_at=float(updated_at),score=self.score,**quality))
        self.frames=[]

    def update(self,frame,timestamp):
        # Close only observed windows, before adding a future frame to the next.
        while timestamp>=self.window_start+WINDOW:
            end=self.window_start+WINDOW
            self._close_window(end,timestamp)
            self.window_start=end
        if timestamp+1e-6>=self.next_sample:
            self.frames.append(frame.copy())
            while self.next_sample<=timestamp+1e-6:self.next_sample+=1/SAMPLE_FPS
        return self.score

    def finish(self,duration):
        # Tail available only after end-of-stream, never backfilled onto old frames.
        self._close_window(duration,duration)

class PeopleMotorcycleDetector:
    def __init__(self,weights=None):
        from ultralytics import YOLO
        weights=Path(weights or ROOT/'artifacts/yolov8n.pt')
        if not weights.exists():
            raise FileNotFoundError('Thiếu detector. Chạy: python scripts/setup_detector.py')
        self.model=YOLO(str(weights))

    def detect(self,frame):
        results=self.model.track(frame,persist=True,classes=[0,3],conf=.25,iou=.45,
                                 tracker='bytetrack.yaml',device='cpu',imgsz=640,verbose=False)
        boxes=results[0].boxes
        if boxes is None:return []
        ids=boxes.id.int().cpu().tolist() if boxes.id is not None else [None]*len(boxes)
        return [dict(bbox=[float(v) for v in xy],class_id=int(cls),confidence=float(conf),track_id=tid)
                for xy,cls,conf,tid in zip(boxes.xyxy.cpu().tolist(),boxes.cls.cpu().tolist(),boxes.conf.cpu().tolist(),ids)]

def near_motorcycle(person,bike):
    """Image-space proximity cue; no authorization or semantic action inference."""
    x1,y1,x2,y2=person;bx1,by1,bx2,by2=bike
    dx=max(bx1-x2,x1-bx2,0);dy=max(by1-y2,y1-by2,0)
    return float(np.hypot(dx,dy))<=.15*max(bx2-bx1,by2-by1)

class ReviewRules:
    def __init__(self,threshold,min_dwell=2.0,max_gap=.75):
        self.threshold=threshold;self.min_dwell=min_dwell;self.max_gap=max_gap;self.state={}

    def update(self,detections,timestamp,score,usable):
        bikes=[d['bbox'] for d in detections if d['class_id']==3]
        people=[];seen=set()
        for detection in detections:
            if detection['class_id']!=0:continue
            row=dict(detection);tid=row['track_id']
            near=any(near_motorcycle(row['bbox'],bike) for bike in bikes)
            dwell=0.0
            if tid is not None:
                seen.add(tid);previous=self.state.get(tid)
                if near:
                    since=previous['since'] if previous and previous['near'] and timestamp-previous['last']<=self.max_gap else timestamp
                    dwell=timestamp-since
                else:since=timestamp
                self.state[tid]=dict(since=since,last=timestamp,near=near)
            row.update(near_motorcycle=near,dwell_seconds=round(dwell,2),
                       scene_score=score,review=bool(usable and score is not None and score>=self.threshold
                                                    and near and dwell>=self.min_dwell))
            people.append(row)
        for tid in list(self.state):
            if tid not in seen and timestamp-self.state[tid]['last']>self.max_gap:del self.state[tid]
        return people

def draw_overlay(frame,people,bikes,score,usable,timestamp,threshold):
    out=frame.copy();h,w=out.shape[:2]
    # Motorcycles inform the proximity rule; only people are outlined to keep
    # crowded parking footage readable.
    for person in people:
        x1,y1,x2,y2=map(int,person['bbox'])
        color=(40,60,240) if person['review'] else (70,210,70)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        tid=person['track_id']
        label=f"{'REVIEW' if person['review'] else 'PERSON'} #{tid if tid is not None else '?'} det={person['confidence']:.2f}"
        detail=f"scene={score:.2f}" if score is not None else 'scene=warming up'
        detail+=f" near-bike={person['dwell_seconds']:.1f}s"
        lines=[label,detail]
        if 'behavior_score' in person:
            lines.append('MIL experimental='+('n/a' if person['behavior_score'] is None else f"{person['behavior_score']:.2f}"))
        label_width=max(cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,.43,1)[0][0] for line in lines)+8
        label_height=17*len(lines)+3
        lx=max(2,min(x1,w-label_width-2));ly=max(62,min(y1-label_height,h-label_height-2))
        cv2.rectangle(out,(lx,ly),(min(w-1,lx+label_width),ly+label_height),(20,20,20),-1)
        for j,line in enumerate(lines):
            cv2.putText(out,line,(lx+4,ly+14+j*17),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)
    cv2.rectangle(out,(0,0),(w,min(h,58)),(25,25,25),-1)
    status='WAITING FOR 4s CONTEXT' if score is None else f'SCENE RISK {score:.3f} / threshold {threshold:.2f}'
    cv2.putText(out,f'{timestamp:06.2f}s | {status}',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,'Red = review cue, not confirmed theft'+(' | LOW QUALITY' if score is not None and not usable else ''),
                (10,45),cv2.FONT_HERSHEY_SIMPLEX,.43,(100,210,255),1,cv2.LINE_AA)
    return out

def stream_video(path,output_path,detector=None,target_fps=8,pace=False,behavior=False,use_behavior_alarm=False):
    """Yield annotated frames as they are processed; then a playable H.264 file.

    This simulates streaming a CFR file, not an RTSP service. Actual wall-clock
    FPS is reported; on slow hardware playback falls behind source time.
    """
    if target_fps<SAMPLE_FPS:raise ValueError('FPS xử lý cần ít nhất 4')
    if use_behavior_alarm and not behavior:raise ValueError('Cần bật đặc trưng hành vi trước khi dùng MIL')
    info=video_info(path);fps=min(float(target_fps),info['fps'],4. if behavior else float(target_fps))
    if fps<SAMPLE_FPS:raise ValueError('FPS video nguồn dưới 4, không đủ cho bộ chấm điểm')
    detector=detector or PeopleMotorcycleDetector()
    scorer=CausalSceneScorer();rules=ReviewRules(scorer.bundle['threshold'])
    estimator=PoseEstimator(ROOT/'artifacts/yolov8n-pose.pt') if behavior else None
    behavior_engine=InteractionFeatures() if behavior else None
    behavior_model=None;latest_behavior={};next_behavior=0.;peak_behavior=0.;streaks={}
    if behavior and (ROOT/'artifacts/behavior_v2/model.pt').exists():
        from .behavior_model import BehaviorScorer
        behavior_model=BehaviorScorer()
    if use_behavior_alarm and behavior_model is None:raise ValueError('Chưa có mô hình hành vi; chạy scripts/train_behavior.py')
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();cap=cv2.VideoCapture(str(path))
    scale=min(1,960/info['width'])
    width=max(2,int(info['width']*scale)//2*2);height=max(2,int(info['height']*scale)//2*2)
    rows=[];processed=0;peak=0.0;next_index=0.0;writer=None
    with tempfile.TemporaryDirectory(prefix='overlay_',dir=output_path.parent) as temp:
        raw=Path(temp)/'raw.mp4'
        try:
            writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
            if not writer.isOpened():raise RuntimeError('Không mở được bộ ghi video')
            for index in range(info['frames']):
                if not cap.grab():raise RuntimeError(f'Giải mã lỗi tại frame {index}; hãy chuyển video sang CFR')
                if index+1e-6<round(next_index):continue
                next_index+=info['fps']/fps
                ok,frame=cap.retrieve()
                if not ok:raise RuntimeError(f'Không đọc được frame {index}')
                timestamp=index/info['fps']
                frame=cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
                score=scorer.update(frame,timestamp)
                usable=bool(scorer.quality and scorer.quality['usable'])
                detections=detector.detect(frame)
                pose_observations=[]
                if behavior:
                    detections=estimator.attach(frame,detections)
                    if timestamp+1e-6>=next_behavior:
                        observations,cut=behavior_engine.update(frame,detections,timestamp)
                        while next_behavior<=timestamp+1e-6:next_behavior+=.5
                        if cut:latest_behavior.clear();streaks.clear()
                        for observation in observations:
                            bscore=None
                            if (behavior_model is not None and observation['observations']>=4
                                    and observation['observed_seconds']>=1.4 and observation['pose_coverage']>=.25):
                                bscore=float(behavior_model.predict(observation['feature'][None])[0]);peak_behavior=max(peak_behavior,bscore)
                            observation['behavior_score']=bscore
                            tid=observation['track_id'];latest_behavior[tid]=observation
                            positive=bool(bscore is not None and bscore>=behavior_model.threshold and observation['near_bike'] and usable)
                            streaks[tid]=(streaks.get(tid,0)+1) if positive else 0
                    # Draw only currently observed joints, never invent occluded points.
                    pose_observations=[dict(keypoints=d.get('keypoints')) for d in detections if d['class_id']==0]
                people=rules.update(detections,timestamp,score,usable)
                if behavior:
                    for person in people:
                        evidence=latest_behavior.get(person['track_id'])
                        if evidence and timestamp-evidence['timestamp']<=.8:
                            person.update(behavior_score=evidence['behavior_score'],
                                behavior_observed_at=evidence['timestamp'],
                                evidence={k:v for k,v in evidence.items() if k not in ['feature','keypoints','bbox','track_id','behavior_score']})
                        else:person.update(behavior_score=None,evidence={})
                        if use_behavior_alarm:person['review']=bool(person['behavior_score'] is not None and streaks.get(person['track_id'],0)>=2 and usable)
                bikes=[d for d in detections if d['class_id']==3]
                overlay_source=draw_pose(frame.copy(),pose_observations) if behavior else frame
                annotated=draw_overlay(overlay_source,people,bikes,score,usable,timestamp,rules.threshold)
                writer.write(annotated);processed+=1
                if score is not None:peak=max(peak,score)
                wall=time.perf_counter()-start
                if pace and timestamp>wall:time.sleep(min(timestamp-wall,1/fps))
                elapsed=time.perf_counter()-start
                summary=dict(timestamp=timestamp,scene_score=score,score_available_at=scorer.score_updated_at,
                             quality_ok=usable,people=people,motorcycles=len(bikes),review_count=sum(p['review'] for p in people))
                rows.append(summary)
                yield dict(kind='frame',frame=annotated,progress=min(1,(index+1)/info['frames']),
                           processed_fps=processed/max(elapsed,1e-6),elapsed_seconds=elapsed,**summary)
            scorer.finish(info['duration'])
        finally:
            cap.release()
            if writer is not None:writer.release()
        yield dict(kind='encoding',progress=1.0)
        encode_h264(raw,output_path)
    report=dict(video=Path(path).name,mode='causal_stream',detector='YOLOv8n COCO person/motorcycle + ByteTrack',
                behavior_features=behavior,behavior_model=(behavior_model.bundle['name'] if behavior_model else None),
                alarm_mode='experimental_mil' if use_behavior_alarm else 'baseline_scene_rule',
                alarm_threshold=behavior_model.threshold if use_behavior_alarm else rules.threshold,
                peak_behavior_score=peak_behavior if behavior_model else None,
                threshold=rules.threshold,target_fps=fps,processed_frames=processed,
                elapsed_seconds=round(time.perf_counter()-start,3),peak_scene_score=peak,
                review_frames=sum(r['review_count']>0 for r in rows),windows=scorer.records,frames=rows,
                annotated_video=str(output_path),
                warning='Scene risk is not a person-level theft probability; red boxes are heuristic review candidates.')
    output_path.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    yield dict(kind='complete',report=report,path=str(output_path))
