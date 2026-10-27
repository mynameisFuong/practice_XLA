"""Causal person/bike tracking with optional pose/ResNet heads and risk policy.

Scores are exploratory, not calibrated theft probabilities. Three display
levels use temporal interaction rules; no lock-breaking action model is claimed.
"""
from collections import deque
from pathlib import Path
import json,time,tempfile,importlib,hashlib
import cv2
import joblib
import numpy as np
from .features import describe_window,video_info,VERSION,WINDOW,SAMPLE_FPS
from .media import encode_h264
from . import behavior_features as behavior_module
if getattr(behavior_module,'INTERACTION_EVIDENCE_VERSION',0)<4:importlib.reload(behavior_module)
PoseEstimator=behavior_module.PoseEstimator
InteractionFeatures=behavior_module.InteractionFeatures
draw_pose=behavior_module.draw_pose
from . import evidence as evidence_module
if getattr(evidence_module,'EVIDENCE_API_VERSION',0)<2:importlib.reload(evidence_module)
EvidenceRecorder=evidence_module.EvidenceRecorder
from . import risk_levels as risk_module
if getattr(risk_module,'RISK_POLICY_VERSION','')!='three-level-observable-v3':importlib.reload(risk_module)
RiskLevels=risk_module.RiskLevels
LABELS=risk_module.LABELS
COLORS=risk_module.COLORS
OVERLAY_LABELS=risk_module.OVERLAY_LABELS
from .preprocessing import enhance_frame,MODES

ROOT=Path(__file__).resolve().parents[1]
STREAM_API_VERSION=5

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
    def __init__(self,weights=None,profile='fast'):
        from ultralytics import YOLO
        weights=Path(weights or ROOT/'artifacts/yolov8n.pt')
        if not weights.exists():
            raise FileNotFoundError('Thiếu detector. Chạy: python scripts/setup_detector.py')
        self.model=YOLO(str(weights))
        if profile not in {'fast','small_people'}:raise ValueError('Unknown detector profile')
        self.profile=profile;self.tile_tracker=None
        if profile=='small_people':
            from types import SimpleNamespace
            from ultralytics.trackers.byte_tracker import BYTETracker
            self.tile_tracker=BYTETracker(SimpleNamespace(track_high_thresh=.2,track_low_thresh=.1,
                new_track_thresh=.2,track_buffer=30,match_thresh=.8,fuse_score=True))
        names=self.model.names
        if self.model.task!='detect' or names.get(0)!='person' or names.get(3)!='motorcycle':
            raise ValueError('Detector cần tác vụ detect với lớp 0=person và 3=motorcycle; không tương thích checkpoint này.')
        self.metadata=dict(weights=str(weights.resolve()),sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                           task=self.model.task,names=names,tracker='ByteTrack',profile=profile,
                           confidence_threshold=.15 if profile=='small_people' else .25)

    def detect(self,frame):
        if self.profile=='small_people':return self._detect_small(frame)
        results=self.model.track(frame,persist=True,classes=[0,3],conf=.25,iou=.45,
                                 tracker='bytetrack.yaml',device='cpu',imgsz=640,verbose=False)
        boxes=results[0].boxes
        if boxes is None:return []
        ids=boxes.id.int().cpu().tolist() if boxes.id is not None else [None]*len(boxes)
        return [dict(bbox=[float(v) for v in xy],class_id=int(cls),confidence=float(conf),track_id=tid)
                for xy,cls,conf,tid in zip(boxes.xyxy.cpu().tolist(),boxes.cls.cpu().tolist(),boxes.conf.cpu().tolist(),ids)]

    def _detect_small(self,frame):
        from .small_objects import detection_regions,merge_candidates
        from ultralytics.engine.results import Boxes
        candidates=[]
        for region,x,y,size in detection_regions(frame):
            result=self.model.predict(region,classes=[0,3],conf=.15,iou=.45,device='cpu',imgsz=size,verbose=False)[0]
            if result.boxes is None:continue
            boxes=result.boxes.data.cpu().numpy().copy()
            boxes[:,[0,2]]+=x;boxes[:,[1,3]]+=y
            candidates.extend(boxes.tolist())
        merged=merge_candidates(candidates,frame.shape)
        tracks=self.tile_tracker.update(Boxes(merged,frame.shape[:2]),frame)
        tracked={int(row[-1]):row for row in tracks}
        # Show weak unmatched detections with ID=None rather than invent an ID
        # or accumulate vehicle interaction time for unconfirmed candidates.
        rows=[]
        for i,box in enumerate(merged):
            track=tracked.get(i)
            rows.append(dict(bbox=[float(v) for v in (track[:4] if track is not None else box[:4])],
                class_id=int(box[5]),confidence=float(box[4]),track_id=int(track[4]) if track is not None else None))
        return rows

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
        level=person.get('risk_level','danger' if person['review'] else 'unknown')
        color=COLORS[level]
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        tid=person['track_id']
        label=f"{OVERLAY_LABELS[level]} #{tid if tid is not None else '?'} det={person['confidence']:.2f}"
        detail=f"scene={score:.2f}" if score is not None else 'scene=warming up'
        detail+=f" near-bike={person['dwell_seconds']:.1f}s"
        lines=[label,detail]
        if 'behavior_score' in person:
            lines.append('MIL experimental='+('n/a' if person['behavior_score'] is None else f"{person['behavior_score']:.2f}"))
        if person.get('visual_score') is not None:lines.append(f"ResNet MIL={person['visual_score']:.2f}")
        if person.get('evidence',{}).get('crouching_proxy'):lines.append('Crouching proxy: verify activity')
        label_width=max(cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,.43,1)[0][0] for line in lines)+8
        label_height=17*len(lines)+3
        lx=max(2,min(x1,w-label_width-2));ly=max(62,min(y1-label_height,h-label_height-2))
        cv2.rectangle(out,(lx,ly),(min(w-1,lx+label_width),ly+label_height),(20,20,20),-1)
        for j,line in enumerate(lines):
            cv2.putText(out,line,(lx+4,ly+14+j*17),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)
    cv2.rectangle(out,(0,0),(w,min(h,58)),(25,25,25),-1)
    status='WAITING FOR 4s CONTEXT' if score is None else f'SCENE RISK {score:.3f} / threshold {threshold:.2f}'
    cv2.putText(out,f'{timestamp:06.2f}s | {status}',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    legend=('Green: no cues | Yellow: suspicious | Red: high risk, verify' if people
            else 'NO PERSON DETECTED | Behavior not assessed')
    cv2.putText(out,legend+(' | LOW QUALITY' if score is not None and not usable else ''),
                (10,45),cv2.FONT_HERSHEY_SIMPLEX,.43,(100,210,255),1,cv2.LINE_AA)
    return out

def stream_video(path,output_path,detector=None,target_fps=8,pace=False,behavior=False,use_behavior_alarm=False,
                 camera_id='HAM_B1_CAM01',capture_evidence=True,evidence_root=None,evidence_db=None,risk_backend=None,
                 detector_weights=None,preprocessing='off',denoise=False,sharpen=False,detector_profile='fast'):
    """Yield annotated frames as they are processed; then a playable H.264 file.

    This simulates streaming a CFR file, not an RTSP service. Actual wall-clock
    FPS is reported; on slow hardware playback falls behind source time.
    """
    if target_fps<SAMPLE_FPS:raise ValueError('FPS xử lý cần ít nhất 4')
    if preprocessing not in MODES:raise ValueError('Unknown preprocessing mode')
    backend=risk_backend or ('pose_mil' if use_behavior_alarm else 'scene')
    if backend not in {'scene','pose_mil','resnet_mil'}:raise ValueError('Unknown risk backend')
    if backend!='scene' and not behavior:raise ValueError('Cần bật đặc trưng hành vi trước khi dùng MIL')
    info=video_info(path);fps=min(float(target_fps),info['fps'],4. if behavior else float(target_fps))
    if fps<SAMPLE_FPS:raise ValueError('FPS video nguồn dưới 4, không đủ cho bộ chấm điểm')
    if detector is not None and detector_weights is not None:
        raise ValueError('Chỉ truyền detector hoặc detector_weights, không truyền cả hai.')
    detector=detector or (PeopleMotorcycleDetector(detector_weights) if detector_profile=='fast'
                         else PeopleMotorcycleDetector(detector_weights,profile=detector_profile))
    scorer=CausalSceneScorer();rules=ReviewRules(scorer.bundle['threshold'])
    estimator=PoseEstimator(ROOT/'artifacts/yolov8n-pose.pt') if behavior else None
    behavior_engine=InteractionFeatures() if behavior else None
    behavior_model=None;latest_behavior={};next_behavior=0.;peak_behavior=0.;peak_visual=0.
    visual_model=None;visual_encoder=None
    if behavior and (ROOT/'artifacts/behavior_v2/model.pt').exists():
        from .behavior_model import BehaviorScorer
        behavior_model=BehaviorScorer()
    if backend=='pose_mil' and behavior_model is None:raise ValueError('Chưa có mô hình hành vi; chạy scripts/train_behavior.py')
    if backend=='resnet_mil':
        from .visual_model import VisualBehaviorScorer
        from .visual_backbone import ResNetPersonEncoder,person_crop
        visual_model=VisualBehaviorScorer();visual_encoder=ResNetPersonEncoder()
    alarm_mode={'scene':'baseline_scene_rule','pose_mil':'experimental_mil','resnet_mil':'resnet_mil'}[backend]
    alarm_threshold=visual_model.threshold if visual_model else behavior_model.threshold if backend=='pose_mil' else rules.threshold
    risk_levels=RiskLevels(alarm_threshold,require_pose=behavior)
    recorder=EvidenceRecorder(path,camera_id,alarm_mode,alarm_threshold,root=evidence_root,db_path=evidence_db) if capture_evidence else None
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
                source_frame=frame
                timestamp=index/info['fps']
                frame=cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
                score=scorer.update(frame,timestamp)
                usable=bool(scorer.quality and scorer.quality['usable'])
                inference_frame,filter_info=enhance_frame(frame,preprocessing,denoise,sharpen)
                detections=detector.detect(inference_frame)
                pose_observations=[]
                if behavior:
                    detections=estimator.attach(inference_frame,detections)
                    if timestamp+1e-6>=next_behavior:
                        observations,cut=behavior_engine.update(frame,detections,timestamp)
                        while next_behavior<=timestamp+1e-6:next_behavior+=.5
                        if cut:latest_behavior.clear();risk_levels.states.clear()
                        eligible=[i for i,o in enumerate(observations) if o['observations']>=4 and o['observed_seconds']>=1.4]
                        visual_vectors={}
                        if visual_encoder and eligible:
                            embeddings=visual_encoder.encode([person_crop(frame,observations[i]['bbox']) for i in eligible])
                            visual_vectors=dict(zip(eligible,embeddings))
                        for observation_index,observation in enumerate(observations):
                            bscore=None
                            if (behavior_model is not None and observation['observations']>=4
                                    and observation['observed_seconds']>=1.4 and observation['pose_coverage']>=.25):
                                bscore=float(behavior_model.predict(observation['feature'][None])[0]);peak_behavior=max(peak_behavior,bscore)
                            observation['behavior_score']=bscore
                            vscore=None
                            if observation_index in visual_vectors:
                                vscore=float(visual_model.predict(np.r_[visual_vectors[observation_index],observation['feature']][None])[0])
                                peak_visual=max(peak_visual,vscore)
                            observation['visual_score']=vscore
                            tid=observation['track_id'];latest_behavior[tid]=observation
                    # Draw only currently observed joints, never invent occluded points.
                    pose_observations=[dict(keypoints=d.get('keypoints')) for d in detections if d['class_id']==0]
                people=rules.update(detections,timestamp,score,usable)
                if behavior:
                    for person in people:
                        evidence=latest_behavior.get(person['track_id'])
                        if evidence and timestamp-evidence['timestamp']<=.8:
                            person.update(behavior_score=evidence['behavior_score'],
                                visual_score=evidence.get('visual_score'),
                                behavior_observed_at=evidence['timestamp'],
                                evidence={k:v for k,v in evidence.items() if k not in ['feature','keypoints','bbox','track_id','behavior_score','visual_score']})
                        else:person.update(behavior_score=None,visual_score=None,evidence={})
                risk_levels.assess(people,timestamp,usable,score_key={'scene':'scene_score','pose_mil':'behavior_score','resnet_mil':'visual_score'}[backend])
                for person in people:
                    person['preprocessing']=filter_info
                bikes=[d for d in detections if d['class_id']==3]
                overlay_source=draw_pose(frame.copy(),pose_observations) if behavior else frame
                annotated=draw_overlay(overlay_source,people,bikes,score,usable,timestamp,rules.threshold)
                captures=recorder.update(source_frame,annotated,people,timestamp,index,info['fps'],score,scorer.score_updated_at) if recorder else []
                writer.write(annotated);processed+=1
                if score is not None:peak=max(peak,score)
                wall=time.perf_counter()-start
                if pace and timestamp>wall:time.sleep(min(timestamp-wall,1/fps))
                elapsed=time.perf_counter()-start
                summary=dict(timestamp=timestamp,scene_score=score,score_available_at=scorer.score_updated_at,
                             quality_ok=usable,preprocessing=filter_info,people=people,motorcycles=len(bikes),review_count=sum(p['review'] for p in people),
                             risk_counts={level:sum(p['risk_level']==level for p in people) for level in LABELS},
                             new_evidence=captures)
                rows.append(summary)
                yield dict(kind='frame',frame=annotated,progress=min(1,(index+1)/info['frames']),
                           processed_fps=processed/max(elapsed,1e-6),elapsed_seconds=elapsed,**summary)
            scorer.finish(info['duration'])
        finally:
            cap.release()
            if writer is not None:writer.release()
            if recorder is not None:recorder.close()
        yield dict(kind='encoding',progress=1.0)
        encode_h264(raw,output_path)
    detector_metadata=getattr(detector,'metadata',None)
    report=dict(video=Path(path).name,mode='causal_stream',
                detector=(Path(detector_metadata['weights']).name+' + ByteTrack' if detector_metadata else type(detector).__name__),
                detector_metadata=detector_metadata,
                preprocessing=dict(mode=preprocessing,denoise=bool(denoise),sharpen=bool(sharpen),
                                   applied_to='detector_and_pose_only',learned_features='original_frames',evidence='original_frames'),
                behavior_features=behavior,behavior_model=(behavior_model.bundle['name'] if behavior_model else None),
                alarm_mode=alarm_mode,alarm_threshold=alarm_threshold,
                risk_policy=risk_levels.policy,selected_risk_backend=backend,
                visual_model=visual_model.bundle['name'] if visual_model else None,
                peak_visual_score=peak_visual if visual_model else None,
                camera_id=camera_id,evidence=recorder.records if recorder else [],
                peak_behavior_score=peak_behavior if behavior_model else None,
                threshold=rules.threshold,target_fps=fps,processed_frames=processed,
                elapsed_seconds=round(time.perf_counter()-start,3),peak_scene_score=peak,
                review_frames=sum(r['review_count']>0 for r in rows),windows=scorer.records,frames=rows,
                annotated_video=str(output_path),
                warning='Scene risk is not a person-level theft probability; red boxes are heuristic review candidates.')
    output_path.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    yield dict(kind='complete',report=report,path=str(output_path))
