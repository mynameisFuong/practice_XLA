from pathlib import Path
import hashlib,json
import cv2
import numpy as np
from src.evidence import EvidenceRecorder
from src.storage import connect,list_evidence,update_event

def candidate(tid=7,review=True):
    return dict(track_id=tid,review=review,bbox=[10.,5.,40.,40.],confidence=.9,behavior_score=.8,dwell_seconds=2.)

def recorder(tmp_path,**kwargs):
    source=tmp_path/'source.mp4';source.write_bytes(b'test source identifier')
    return EvidenceRecorder(source,root=tmp_path/'evidence',db_path=tmp_path/'events.sqlite3',**kwargs)

def test_trigger_saves_lossless_source_separate_markup_and_pending_event(tmp_path):
    original=np.random.default_rng(42).integers(0,256,(100,200,3),dtype=np.uint8)
    annotated=np.zeros((50,100,3),np.uint8)
    r=recorder(tmp_path,camera_id='CAM_TEST')
    records=r.update(original,annotated,[candidate()],1.5,12,8,.9,1.)
    r.close();assert len(records)==1
    meta=json.loads(Path(records[0]['metadata_path']).read_text())
    np.testing.assert_array_equal(cv2.imread(meta['original_path']),original)
    assert meta['source_frame_index']==12 and meta['video_time_seconds']==1.5
    assert meta['camera_id']=='CAM_TEST' and meta['track_ids']==[7]
    assert meta['people'][0]['bbox_source']==[20,10,80,80]
    assert meta['original_sha256']==hashlib.sha256(Path(meta['original_path']).read_bytes()).hexdigest()
    assert len(list_evidence(meta['event_id'],tmp_path/'events.sqlite3'))==1
    with connect(tmp_path/'events.sqlite3') as db:
        assert db.execute('SELECT status FROM events WHERE id=?',(meta['event_id'],)).fetchone()[0]=='pending'
    update_event(meta['event_id'],'false_alarm','Chủ xe lấy xe','Bảo vệ',tmp_path/'events.sqlite3')
    assert Path(meta['original_path']).exists()

def test_no_candidate_does_not_create_snapshot(tmp_path):
    r=recorder(tmp_path);frame=np.zeros((100,200,3),np.uint8)
    assert not r.update(frame,frame,[candidate(review=False)],0,0,8,.9,0)
    assert not r.root.exists()

def test_episode_grouping_peak_update_and_cooldown(tmp_path):
    r=recorder(tmp_path);frame=np.zeros((100,200,3),np.uint8)
    for t in [0,.5,1,1.5,2]:r.update(frame,frame,[candidate()],t,round(t*8),8,.6+t*.1,t)
    assert len(r.records)==1
    # A fresh episode inside cooldown is suppressed, then allowed after 10s.
    assert not r.update(frame,frame,[candidate()],7,56,8,.9,7)
    assert len(r.update(frame,frame,[candidate()],11,88,8,.9,11))==1
    r.close();assert len(r.records)==2
    meta=json.loads(Path(r.records[0]['metadata_path']).read_text())
    assert meta['alarm_score']==.6 and meta['peak_alarm_score']==.8
    assert meta['last_review_video_time_seconds']==2

def test_simultaneous_people_share_one_frame_with_both_ids(tmp_path):
    r=recorder(tmp_path);frame=np.zeros((100,200,3),np.uint8)
    records=r.update(frame,frame,[candidate(1),candidate(2)],4,32,8,.8,4)
    r.close()
    assert len(records)==1 and records[0]['track_ids']==[1,2]
    assert len(list(r.root.rglob('original.png')))==1

def test_stream_captures_triggering_frame_and_survives_cancellation(tmp_path,monkeypatch):
    import src.live as live
    source=tmp_path/'moving.avi';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),8,(160,120))
    for i in range(16):
        frame=np.full((120,160,3),i*12,np.uint8);writer.write(frame)
    writer.release()
    class NoDetections:
        def detect(self,frame):return []
    # Exercise persistence only; fake alerts are test fixtures, not accuracy data.
    class FixtureRules:
        def __init__(self,threshold):self.threshold=threshold
        def update(self,detections,timestamp,score,usable):
            return [candidate()] if timestamp>=1 else []
    class FixtureScorer:
        bundle={'threshold':.5};quality={'usable':True};score_updated_at=0;records=[]
        def update(self,frame,timestamp):self.score_updated_at=timestamp;return .8
        def finish(self,duration):pass
    class FixtureRiskLevels:
        def __init__(self,threshold,require_pose=False):self.policy={};self.states={}
        def assess(self,people,timestamp,usable,score_key='scene_score'):
            for p in people:p.update(risk_level='danger',risk_label='Test fixture',risk_reasons=['persistence test'])
            return people
    monkeypatch.setattr(live,'ReviewRules',FixtureRules);monkeypatch.setattr(live,'CausalSceneScorer',FixtureScorer)
    monkeypatch.setattr(live,'RiskLevels',FixtureRiskLevels)
    stream=live.stream_video(source,tmp_path/'out.mp4',detector=NoDetections(),evidence_root=tmp_path/'captures',evidence_db=tmp_path/'db.sqlite3')
    for update in stream:
        if update.get('new_evidence'):
            saved=update['new_evidence'][0];break
    stream.close()
    meta=json.loads(Path(saved['metadata_path']).read_text())
    assert meta['source_frame_index']==8 and meta['video_time_seconds']==1
    original=cv2.imread(meta['original_path'])
    assert original.mean()>80  # first frame was black: prove it wasn't saved instead
    assert not (tmp_path/'out.mp4').exists()
    assert list_evidence(meta['event_id'],tmp_path/'db.sqlite3')

def test_yellow_to_red_keeps_a_second_frame_inside_cooldown(tmp_path):
    r=recorder(tmp_path);frame=np.zeros((100,200,3),np.uint8)
    yellow=dict(candidate(),risk_level='suspicious',risk_reasons=['test yellow'])
    red=dict(candidate(),risk_level='danger',risk_reasons=['test red'])
    first=r.update(frame,frame,[yellow],4,32,8,.6,4)[0]
    second=r.update(frame,frame,[red],6,48,8,.9,6)[0]
    r.close()
    assert first['risk_level_at_capture']=='suspicious'
    assert second['risk_level_at_capture']=='danger'
    assert first['source_frame_index']==32 and second['source_frame_index']==48
    assert len(r.records)==2
