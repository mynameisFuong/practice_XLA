"""Temporal display policy, NOT a trained three-class theft recognizer.

Appearance (including a mask) never raises a level. Red needs sustained local
interaction cues AND a high selected-model score. Insufficient data is explicit.
"""
from collections import deque

RISK_POLICY_VERSION='three-level-observable-v3'
LABELS={'safe':'An toàn · chưa ghi nhận dấu hiệu','suspicious':'Nghi vấn',
        'danger':'Nguy hiểm · cần xác minh','unknown':'Chưa đủ dữ liệu'}
COLORS={'safe':(70,210,70),'suspicious':(0,220,255),'danger':(40,50,240),'unknown':(170,170,170)}
OVERLAY_LABELS={'safe':'SAFE','suspicious':'SUSPICIOUS','danger':'HIGH RISK','unknown':'OBSERVING'}
RANK={'unknown':-1,'safe':0,'suspicious':1,'danger':2}

class RiskLevels:
    def __init__(self,yellow_threshold=.5,red_threshold=None,require_pose=False):
        self.yellow=float(yellow_threshold)
        self.red=float(red_threshold if red_threshold is not None else min(.98,max(.8,self.yellow+.2)))
        if not 0<=self.yellow<self.red<=1:raise ValueError('Risk thresholds must satisfy 0 <= yellow < red <= 1')
        self.states={}
        self.require_pose=bool(require_pose)

    @property
    def policy(self):
        return dict(version=RISK_POLICY_VERSION,yellow_score_threshold=self.yellow,red_score_threshold=self.red,
            yellow_hold_seconds=1.,red_hold_seconds=2.,downgrade_hold_seconds=2.,
            require_pose=self.require_pose,
            green_requires_complete_action_visibility=self.require_pose,
            red_requires='near vehicle >=4s, reliable pose, wrist near vehicle >=2s, leaning or crouching, relative wrist movement, high score',
            calibrated=False,trained_three_class_model=False,mask_used=False,helmet_used=False,lock_breaking_detector_available=False)

    def assess(self,people,timestamp,usable,score_key='scene_score'):
        seen=set()
        for person in people:
            tid=person.get('track_id');seen.add(tid)
            score=person.get(score_key)
            evidence=person.get('evidence') or {}
            fresh=bool(evidence and 0<=timestamp-evidence.get('timestamp',-100)<=.8)
            pose_ok=bool(fresh and evidence.get('pose_coverage',0)>=.5 and evidence.get('observations',0)>=4)
            complete_pose=bool(pose_ok and evidence.get('lower_body_valid',False)
                               and evidence.get('left_wrist_visible',False) and evidence.get('right_wrist_visible',False))
            blockers=[]
            if score is None:blockers.append('Chưa có điểm từ nhánh đang chọn')
            elif score<self.yellow:blockers.append('Điểm dưới ngưỡng vàng')
            if not usable:blockers.append('Chất lượng nguồn chưa đạt')
            if tid is None:blockers.append('Chưa có ID theo dõi')
            if person.get('confidence',0)<.35:blockers.append('Độ tin cậy phát hiện người thấp')
            if self.require_pose:
                if not pose_ok:blockers.append('Tư thế chưa đủ quan sát/tin cậy hoặc đã cũ')
                if not evidence.get('lower_body_valid',False):blockers.append('Không đủ điểm hai chân để đánh giá ngồi thấp')
                if not (evidence.get('left_wrist_visible',False) and evidence.get('right_wrist_visible',False)):
                    blockers.append('Không quan sát đủ hai cổ tay')
            if not person.get('near_motorcycle',False):blockers.append('Chưa ghi nhận gần xe máy')
            if person.get('dwell_seconds',0)<3:blockers.append('Thời gian gần xe chưa đủ 3 giây cho nhánh tương tác')
            if not evidence.get('wrist_near_bike',False):blockers.append('Chưa quan sát được tay gần vùng xe')
            person['assessment_diagnostics']=blockers
            state=self.states.get(tid)
            if state is None or timestamp-state['last']>1.:
                state=dict(last=timestamp,level='safe',yellow_since=None,red_since=None,down_since=None,head=deque(),head_t=None,reasons=[])
            state['last']=timestamp
            ready=bool(tid is not None and usable and score is not None and person.get('confidence',0)>=.35
                       and (not self.require_pose or pose_ok))
            if not ready:
                state.update(level='unknown',yellow_since=None,red_since=None,down_since=None,reasons=['Thiếu điểm, chất lượng, theo dõi hoặc tư thế tin cậy'])
                state['head'].clear()
                self._apply(person,state,False)
                if tid is not None:self.states[tid]=state
                continue
            if state['level']=='unknown':state['level']='safe'
            near=bool(person.get('near_motorcycle',False))
            dwell=float(person.get('dwell_seconds',0))
            hand=bool(pose_ok and evidence.get('wrist_near_bike',False))
            lean=bool(pose_ok and evidence.get('torso_lean',0)>=.35)
            crouch=bool(pose_ok and evidence.get('lower_body_valid',False) and evidence.get('crouching_proxy',False))
            moving_hand=bool(pose_ok and evidence.get('hand_motion_valid',False) and evidence.get('hand_motion',0)>=.3)
            e_t=evidence.get('timestamp')
            if pose_ok and e_t!=state['head_t']:
                state['head_t']=e_t
                if evidence.get('head_valid',False):state['head'].append((e_t,float(evidence.get('head_offset_x',0))))
                else:state['head'].clear()
            while state['head'] and timestamp-state['head'][0][0]>6:state['head'].popleft()
            head_values=[v for _,v in state['head']]
            deltas=[b-a for a,b in zip(head_values,head_values[1:]) if abs(b-a)>=.12]
            reversals=sum(a*b<0 for a,b in zip(deltas,deltas[1:]))
            head_motion=bool(len(head_values)>=4 and max(head_values)-min(head_values)>=.5 and reversals>=2)
            repeated=bool(pose_ok and (evidence.get('approach_count',0)>=2 or evidence.get('bike_switch_count',0)>=2))
            cues=[]
            if near and dwell>=2 and score>=self.yellow:cues.append('Điểm mô hình cao khi ở gần xe kéo dài')
            if near and dwell>=3 and hand and (lean or moving_hand):cues.append('Tay gần vùng xe kết hợp tư thế/chuyển động')
            if near and dwell>=3 and crouch and hand and moving_hand and evidence.get('hand_region_seconds',0)>=2:
                cues.append('Tư thế co gối/ngồi thấp kèm thao tác tay gần xe kéo dài')
            if near and head_motion and repeated:cues.append('Biến động hướng đầu ước lượng kèm tiếp cận nhiều lần')
            red_signal=bool(pose_ok and near and dwell>=4 and hand and (lean or crouch) and moving_hand
                            and evidence.get('hand_region_seconds',0)>=2 and score>=self.red)
            if red_signal:cues.append('Chuỗi thao tác tay gần xe kéo dài, cúi/ngồi thấp và điểm cao')
            yellow_signal=bool(cues)
            state['yellow_since']=(state['yellow_since'] if state['yellow_since'] is not None else timestamp) if yellow_signal else None
            state['red_since']=(state['red_since'] if state['red_since'] is not None else timestamp) if red_signal else None
            if state['red_since'] is not None and timestamp-state['red_since']>=2:target='danger'
            elif state['yellow_since'] is not None and timestamp-state['yellow_since']>=1:target='suspicious'
            else:target='safe'
            # Partial pose can still justify observed interaction alerts, but
            # absence of cues behind occlusion must not become a green result.
            if self.require_pose and not complete_pose and target=='safe':
                target='unknown'
                if state['level'] in {'safe','unknown'}:
                    state.update(level='unknown',reasons=['Bị che/thiếu khớp chân hoặc cổ tay; chưa đủ quan sát để hiển thị xanh'])
            if RANK[target]>=RANK[state['level']]:
                state.update(level=target,down_since=None,reasons=cues if target!='safe' else ['Chưa ghi nhận chuỗi dấu hiệu cần xem xét'])
            else:
                if state['down_since'] is None:state['down_since']=timestamp
                if timestamp-state['down_since']>=2:
                    state.update(level=target,down_since=None,reasons=cues if target!='safe' else ['Dấu hiệu đã giảm sau thời gian theo dõi'])
            person['head_motion_proxy']=head_motion
            if not head_motion:person['assessment_diagnostics'].append('Chuyển động đầu 2D chưa đạt điều kiện; không phải nhận dạng hướng nhìn')
            if state['level']=='unknown':
                state['reasons']=['Bị che/thiếu khớp chân hoặc cổ tay; chưa đủ quan sát để hiển thị xanh']
            self._apply(person,state,True)
            self.states[tid]=state
        self.states={tid:s for tid,s in self.states.items() if tid in seen or timestamp-s['last']<=1}
        return people

    @staticmethod
    def _apply(person,state,ready):
        level=state['level']
        person.update(risk_level=level,risk_label=LABELS[level],risk_reasons=list(state['reasons']),
                      assessment_ready=bool(ready and level!='unknown'),review=level in {'suspicious','danger'})
