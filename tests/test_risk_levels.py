from src.risk_levels import RiskLevels

def observation(t,score=.9,pose=True):
    return dict(track_id=1,confidence=.9,scene_score=score,near_motorcycle=True,dwell_seconds=5,
        evidence=dict(timestamp=t,pose_coverage=.9 if pose else 0,observations=8,
            lower_body_valid=True,left_wrist_visible=True,right_wrist_visible=True,
            wrist_near_bike=True,torso_lean=.6,hand_motion=.6,hand_motion_valid=True,
            hand_region_seconds=3,head_valid=False))

def test_score_alone_only_reaches_yellow_without_action_observations():
    assess=RiskLevels(.5)
    for t in [0,.5,1,1.5,2,2.5,3]:
        p=observation(t,pose=False);assess.assess([p],t,True)
        assert p['risk_level']!='danger'
    assert p['risk_level']=='suspicious'

def test_sustained_local_cues_produce_green_yellow_red():
    assess=RiskLevels(.5);levels=[]
    for t in [0,.5,1,1.5,2]:
        p=observation(t);assess.assess([p],t,True);levels.append(p['risk_level'])
    assert levels[0]=='safe' and levels[2]=='suspicious' and levels[-1]=='danger'

def test_mask_or_head_movement_alone_does_not_raise_a_level():
    assess=RiskLevels(.5)
    for i,t in enumerate([0,.5,1,1.5,2,2.5]):
        p=observation(t,score=.2);p['mask_detected']=True;p['near_motorcycle']=False
        p['evidence'].update(head_valid=True,head_offset_x=(-1)**i)
        assess.assess([p],t,True)
        assert p['risk_level']=='safe'

def test_missing_quality_clears_high_risk_instead_of_claiming_safety():
    assess=RiskLevels(.5)
    for t in [0,.5,1,1.5,2]:
        p=observation(t);assess.assess([p],t,True)
    assert p['risk_level']=='danger'
    p=observation(2.5);assess.assess([p],2.5,False)
    assert p['risk_level']=='unknown' and not p['review'] and not p['assessment_ready']

def test_one_bad_frame_does_not_flash_red_and_downgrade_is_delayed():
    assess=RiskLevels(.5)
    p=observation(0);assess.assess([p],0,True)
    p=observation(.5,.1);p['near_motorcycle']=False;assess.assess([p],.5,True)
    assert p['risk_level']=='safe'
    for t in [1,1.5,2,2.5,3]:
        p=observation(t);assess.assess([p],t,True)
    assert p['risk_level']=='danger'
    for t in [3.5,4,4.5,5,5.5]:
        p=observation(t,.1);p['near_motorcycle']=False;assess.assess([p],t,True)
        if t==3.5:assert p['risk_level']=='danger'
    assert p['risk_level']=='safe'

def test_pose_mode_reports_missing_or_stale_pose_as_unknown():
    assess=RiskLevels(.5,require_pose=True)
    p=observation(0,.1,pose=False);assess.assess([p],0,True)
    assert p['risk_level']=='unknown'
    p=observation(0,.1);assess.assess([p],2,True)
    assert p['risk_level']=='unknown'

def test_crouching_or_helmet_alone_cannot_trigger_alert():
    assess=RiskLevels(.5,require_pose=True)
    for t in [0,.5,1,1.5,2,2.5]:
        p=observation(t,.1)
        p['helmet_detected']=True;p['mask_detected']=True
        p['evidence'].update(lower_body_valid=True,crouching_proxy=True,torso_lean=0,
                             wrist_near_bike=False,hand_motion=0,hand_motion_valid=False)
        assess.assess([p],t,True)
        assert p['risk_level']=='safe'

def test_crouching_with_sustained_interaction_can_replace_lean_but_needs_score_for_red():
    for score,expected in [(.1,'suspicious'),(.9,'danger')]:
        assess=RiskLevels(.5,require_pose=True)
        for t in [0,.5,1,1.5,2]:
            p=observation(t,score)
            p['evidence'].update(torso_lean=0,lower_body_valid=True,crouching_proxy=True)
            assess.assess([p],t,True)
        assert p['risk_level']==expected

def test_half_visible_pose_with_hidden_legs_is_not_green():
    assess=RiskLevels(.5,require_pose=True)
    p=observation(16.76,.34957)
    p.update(dwell_seconds=2.76)
    p['evidence'].update(pose_coverage=.5,lower_body_valid=False,right_wrist_visible=False,
                         wrist_near_bike=False,torso_lean=.154,hand_motion=.132)
    assess.assess([p],16.76,True)
    assert p['risk_level']=='unknown' and not p['assessment_ready']
    assert any('hai chân' in reason for reason in p['assessment_diagnostics'])
    assert any('dưới ngưỡng' in reason for reason in p['assessment_diagnostics'])

def test_occluded_legs_do_not_block_sustained_visible_hand_interaction():
    assess=RiskLevels(.5,require_pose=True)
    for t in [0,.5,1,1.5,2]:
        p=observation(t)
        p['evidence'].update(lower_body_valid=False,right_wrist_visible=False)
        assess.assess([p],t,True)
    assert p['risk_level']=='danger'
