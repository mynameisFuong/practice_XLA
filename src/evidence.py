"""Persist the actual triggering frame, grouped by tracked-person episodes.

PNG preserves decoded source pixels. Markup is stored separately. Hashes help
detect file changes; these images document a model alert awaiting human review.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,uuid
import cv2
from .storage import register_evidence

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE_API_VERSION=2
RISK_PRIORITY={'unknown':0,'safe':0,'suspicious':1,'danger':2}

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def atomic_json(path,data):
    path=Path(path);temp=path.with_suffix('.json.part')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)

class EvidenceRecorder:
    def __init__(self,source,camera_id='HAM_B1_CAM01',alarm_mode='baseline_scene_rule',threshold=.5,
                 root=None,db_path=None,release_gap=3.,cooldown=10.):
        self.source=Path(source).resolve();self.camera_id=camera_id;self.alarm_mode=alarm_mode;self.threshold=threshold
        self.root=Path(root or ROOT/'runtime/evidence').resolve();self.db_path=db_path
        self.run_id=uuid.uuid4().hex;self.records=[];self.active={};self.last_capture={}
        self.release_gap=release_gap;self.cooldown=cooldown;self.source_hash=None

    def update(self,original,annotated,people,timestamp,frame_index,source_fps,scene_score,score_available_at):
        candidates=[p for p in people if p.get('review') and p.get('track_id') is not None]
        fresh=[]
        for person in candidates:
            tid=person['track_id'];active=self.active.get(tid)
            priority=RISK_PRIORITY.get(person.get('risk_level','danger'),2)
            score=person.get('behavior_score') if self.alarm_mode=='experimental_mil' else scene_score
            if self.alarm_mode=='resnet_mil':score=person.get('visual_score')
            if score is None:continue
            if active and timestamp-active['last']<=self.release_gap:
                active['last']=timestamp
                active['record']['last_review_video_time_seconds']=float(timestamp)
                active['record']['peak_alarm_score']=max(active['record']['peak_alarm_score'],float(score))
                if priority<=active['priority']:continue
                # Preserve a NEW image at yellow -> red even inside cooldown.
                fresh.append(person);continue
            if timestamp-self.last_capture.get(tid,float('-inf'))<self.cooldown:continue
            fresh.append(person)
        if not fresh:return []
        self.root.mkdir(parents=True,exist_ok=True)
        evidence_id=uuid.uuid4().hex
        folder=self.root/self.run_id/evidence_id;folder.mkdir(parents=True)
        original_path=folder/'original.png';annotated_path=folder/'annotated.jpg';metadata_path=folder/'metadata.json'
        if not cv2.imwrite(str(original_path),original):raise OSError('Không lưu được ảnh gốc bằng chứng')
        if not cv2.imwrite(str(annotated_path),annotated,[cv2.IMWRITE_JPEG_QUALITY,95]):raise OSError('Không lưu được ảnh đánh dấu bằng chứng')
        if self.source_hash is None:self.source_hash=sha256(self.source)
        oh,ow=original.shape[:2];rh,rw=annotated.shape[:2]
        marked=[]
        for person in fresh:
            item={k:v for k,v in person.items() if k!='keypoints'}
            x1,y1,x2,y2=person['bbox']
            item['bbox_source']=[max(0,min(ow,x1*ow/rw)),max(0,min(oh,y1*oh/rh)),
                                 max(0,min(ow,x2*ow/rw)),max(0,min(oh,y2*oh/rh))]
            marked.append(item)
        scores=[float(p['visual_score'] if self.alarm_mode=='resnet_mil' else p['behavior_score'] if self.alarm_mode=='experimental_mil' else scene_score) for p in fresh]
        risk_level=max((p.get('risk_level','danger') for p in fresh),key=lambda level:RISK_PRIORITY.get(level,0))
        record=dict(id=evidence_id,run_id=self.run_id,captured_at_utc=datetime.now(timezone.utc).isoformat(),
            camera_id=self.camera_id,source_video=self.source.name,source_path=str(self.source),source_sha256=self.source_hash,
            video_time_seconds=float(timestamp),source_frame_index=int(frame_index),source_frame_index_base=0,
            source_fps=float(source_fps),source_size=[ow,oh],annotated_size=[rw,rh],
            track_ids=[p['track_id'] for p in fresh],people=marked,alarm_mode=self.alarm_mode,
            risk_level_at_capture=risk_level,trigger_basis='observable_three_level_policy' if 'risk_level' in fresh[0] else 'legacy_review_flag',
            alarm_score=max(scores),peak_alarm_score=max(scores),threshold=float(self.threshold),
            scene_score=scene_score,scene_score_available_at=score_available_at,
            last_review_video_time_seconds=float(timestamp),
            original_path=str(original_path),annotated_path=str(annotated_path),metadata_path=str(metadata_path),
            original_sha256=sha256(original_path),annotated_sha256=sha256(annotated_path),
            status_at_capture='pending',warning='Ghi nhận dấu hiệu do mô hình đánh dấu, cần người xác minh; chưa phải kết luận trộm.')
        # Make the files usable even if the process exits just after DB commit.
        atomic_json(metadata_path,record)
        record['event_id']=register_evidence(record,self.db_path)
        atomic_json(metadata_path,record)
        self.records.append(record)
        for person in fresh:
            tid=person['track_id'];self.active[tid]=dict(last=timestamp,record=record,priority=RISK_PRIORITY.get(person.get('risk_level','danger'),2));self.last_capture[tid]=timestamp
        return [record]

    def close(self):
        # Also run on cancellation: a saved snapshot survives stopping the video.
        for record in self.records:atomic_json(record['metadata_path'],record)
