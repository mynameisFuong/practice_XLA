from pathlib import Path
import cv2
import numpy as np
import pytest
from src.features import VERSION
from src.live import CausalSceneScorer,ReviewRules,stream_video
from src.media import browser_video

class ConstantModel:
    def predict_proba(self,x):return np.tile([.2,.8],(len(x),1))

def test_scene_score_waits_for_past_context():
    scorer=CausalSceneScorer(dict(version=VERSION,model=ConstantModel(),threshold=.5))
    frame=np.zeros((96,128,3),np.uint8)
    for t in np.arange(0,4,.125):assert scorer.update(frame,t) is None
    assert scorer.update(frame,4)==pytest.approx(.8)
    assert scorer.records[0]['available_at']==4
    assert scorer.records[0]['end']==4
    assert scorer.records[0]['valid_frames']==16
    assert len(scorer.frames)==1  # frame at 4s belongs to the next window

def test_red_box_needs_scene_score_quality_and_continuous_nearness():
    person=dict(class_id=0,bbox=[20,10,50,90],confidence=.9,track_id=1)
    bike=dict(class_id=3,bbox=[40,50,110,95],confidence=.8,track_id=2)
    rules=ReviewRules(.5)
    for t in [0,.5,1,1.5]:assert not rules.update([person,bike],t,.8,True)[0]['review']
    assert rules.update([person,bike],2,.8,True)[0]['review']
    assert not rules.update([person,bike],2.1,None,True)[0]['review']
    assert not rules.update([person,bike],2.2,.8,False)[0]['review']
    assert not rules.update([person,bike],2.3,.2,True)[0]['review']
    assert not rules.update([person],2.4,.8,True)[0]['review']
    assert not rules.update([person,bike],4,.8,True)[0]['review']

def test_untracked_person_never_accumulates_someone_elses_dwell():
    p=dict(class_id=0,bbox=[20,10,50,90],confidence=.9,track_id=None)
    b=dict(class_id=3,bbox=[40,50,110,95],confidence=.8,track_id=2)
    rule=ReviewRules(.5)
    for t in [0,1,2,3]:assert not rule.update([p,b],t,.9,True)[0]['review']

def test_h264_playback_and_short_stream_tail(tmp_path):
    source=tmp_path/'source.avi';w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),8,(160,120))
    for i in range(8):
        frame=np.zeros((120,160,3),np.uint8);cv2.rectangle(frame,(i*5,20),(i*5+30,80),(150,180,200),-1);w.write(frame)
    w.release()
    playable=browser_video(source,tmp_path/'playback')
    assert browser_video(source,tmp_path/'playback')==playable
    cap=cv2.VideoCapture(str(playable));fourcc=int(cap.get(cv2.CAP_PROP_FOURCC));ok,frame=cap.read();cap.release()
    assert ok and ''.join(chr((fourcc>>(8*i))&255) for i in range(4)) in {'avc1','h264'}
    class NoPeople:
        def detect(self,frame):return []
    updates=list(stream_video(source,tmp_path/'annotated.mp4',detector=NoPeople()))
    frame_updates=[u for u in updates if u['kind']=='frame']
    assert len(frame_updates)==8
    assert all(u['scene_score'] is None for u in frame_updates)
    assert updates[-1]['kind']=='complete'
    assert updates[-1]['report']['windows'][0]['available_at']==pytest.approx(1)
    assert Path(updates[-1]['path']).exists()
