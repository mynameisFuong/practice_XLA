import numpy as np
import pytest
from src.behavior_features import InteractionFeatures,FRAME_NAMES,DESCRIPTOR_DIM

def person(pose=None,shift=0):
    d=dict(class_id=0,track_id=1,bbox=[20+shift,10,60+shift,100],confidence=.9)
    if pose is not None:d['keypoints']=pose.tolist()
    return d

def bike(track_id=2,shift=0):return dict(class_id=3,track_id=track_id,bbox=[50+shift,50,130+shift,110],confidence=.8)

def test_occluded_pose_is_masked_not_a_wrist_at_origin():
    engine=InteractionFeatures();frame=np.full((120,160,3),100,np.uint8)
    rows,_=engine.update(frame,[person(),bike()],0)
    values=dict(zip(FRAME_NAMES,rows[0]['feature'][:len(FRAME_NAMES)]))
    assert rows[0]['feature'].shape==(DESCRIPTOR_DIM,)
    assert np.isfinite(rows[0]['feature']).all()
    assert values['pose_coverage']==0 and values['left_wrist_valid']==0
    assert values['left_wrist_upper_bike']==0 and values['left_wrist_motion_valid']==0

def test_motion_is_not_invented_when_pose_anchor_is_missing():
    engine=InteractionFeatures();frame=np.zeros((120,160,3),np.uint8)
    pose=np.zeros((17,3));pose[9]=[55,65,.9]
    engine.update(frame,[person(pose),bike()],0)
    pose[9,0]+=10
    rows,_=engine.update(frame,[person(pose),bike()],.5)
    values=dict(zip(FRAME_NAMES,rows[0]['feature'][-len(FRAME_NAMES):]))
    assert values['left_wrist_valid']==1
    assert values['left_wrist_motion_valid']==0
    assert values['left_wrist_speed']==0

def test_pair_dwell_resets_on_bike_switch_or_observation_gap():
    engine=InteractionFeatures();frame=np.zeros((120,160,3),np.uint8)
    engine.update(frame,[person(),bike()],0)
    rows,_=engine.update(frame,[person(),bike()],.5)
    last=rows[0]['feature'][-len(FRAME_NAMES):]
    assert last[FRAME_NAMES.index('near_seconds')]==pytest.approx(.5/4)
    rows,_=engine.update(frame,[person(),bike(track_id=3)],1)
    assert rows[0]['feature'][-len(FRAME_NAMES):][FRAME_NAMES.index('near_seconds')]==0
    rows,_=engine.update(frame,[person(),bike(track_id=3)],3)
    assert rows[0]['observations']==1 and rows[0]['observed_seconds']==0

def test_global_camera_translation_does_not_become_bike_movement():
    engine=InteractionFeatures();frame=np.zeros((120,160,3),np.uint8)
    class KnownCameraMotion:
        def update(self,frame,people,t):return np.array([[1.,0.,10.],[0.,1.,0.]]),True,False
    engine.camera=KnownCameraMotion()
    engine.update(frame,[person(),bike()],0)
    rows,_=engine.update(frame,[person(shift=10),bike(shift=10)],.5)
    values=dict(zip(FRAME_NAMES,rows[0]['feature'][-len(FRAME_NAMES):]))
    assert values['bike_motion_valid']==1 and values['person_motion_valid']==1
    assert values['bike_speed']==pytest.approx(0) and values['person_speed']==pytest.approx(0)
