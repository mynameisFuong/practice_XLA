"""Observable pose and person-motorcycle interactions, without identity cues.

Missing joints/motion are explicitly masked. Nothing here labels a person as a
thief. Four-second person-track descriptors feed a weakly supervised MIL head.
"""
from collections import deque
import cv2
import numpy as np

BEHAVIOR_VERSION='pose-hoi-v1'
JOINTS=list(range(5,17))
FRAME_NAMES=([f'joint_{j}_{axis}' for j in JOINTS for axis in ['x','y']]
             +[f'joint_{j}_valid' for j in JOINTS]
             +['left_elbow_cos','right_elbow_cos','torso_lean',
                'left_wrist_bike_distance','right_wrist_bike_distance',
                'left_wrist_upper_bike','right_wrist_upper_bike','left_wrist_valid','right_wrist_valid',
                'left_wrist_speed','right_wrist_speed','left_wrist_motion_valid','right_wrist_motion_valid',
                'person_speed','direction_change','person_motion_valid',
                'bike_speed','bike_motion_valid','co_motion',
                'near_bike','bike_visible','near_seconds','hand_region_seconds','approach_count',
                'bike_switch_count','track_seconds','pose_coverage','head_offset_x','head_offset_y','head_valid',
                'body_aspect'])
DESCRIPTOR_NAMES=[f'{stat}_{name}' for stat in ['mean','std','last'] for name in FRAME_NAMES]
FRAME_DIM=len(FRAME_NAMES)
DESCRIPTOR_DIM=len(DESCRIPTOR_NAMES)

def box_iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    return intersection/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)

def center(box):return np.array([(box[0]+box[2])/2,(box[1]+box[3])/2],float)

def point_box_distance(point,box):
    return float(np.hypot(max(box[0]-point[0],point[0]-box[2],0),max(box[1]-point[1],point[1]-box[3],0)))

def box_distance(a,b):
    return float(np.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0)))

def transformed(point,matrix):return matrix[:,:2]@np.asarray(point)+matrix[:,2]

class PoseEstimator:
    def __init__(self,weights):
        from ultralytics import YOLO
        self.model=YOLO(str(weights))

    def attach(self,frame,detections):
        people=[d for d in detections if d['class_id']==0]
        if not people:return detections
        result=self.model.predict(frame,device='cpu',imgsz=640,conf=.25,iou=.45,verbose=False)[0]
        if result.keypoints is None or result.boxes is None:return detections
        boxes=result.boxes.xyxy.cpu().numpy()
        poses=result.keypoints.data.cpu().numpy()
        pairs=sorted([(box_iou(p['bbox'],box),i,j) for i,p in enumerate(people) for j,box in enumerate(boxes)],reverse=True)
        used_people=set();used_poses=set()
        for overlap,i,j in pairs:
            if overlap<.3:break
            if i in used_people or j in used_poses:continue
            people[i]['keypoints']=poses[j].tolist();used_people.add(i);used_poses.add(j)
        return detections

class CameraMotion:
    """RANSAC affine background motion; no vehicle-motion evidence on failure."""
    def __init__(self):self.gray=None;self.timestamp=None

    def update(self,frame,people,timestamp):
        h,w=frame.shape[:2];gray=cv2.cvtColor(cv2.resize(frame,(320,240)),cv2.COLOR_BGR2GRAY)
        previous=self.gray;dt=timestamp-self.timestamp if self.timestamp is not None else 0
        self.gray=gray;self.timestamp=timestamp
        identity=np.array([[1.,0.,0.],[0.,1.,0.]])
        if previous is None or not 0<dt<=1.25:return identity,False,False
        cut=float(np.mean(cv2.absdiff(previous,gray)))>45
        if cut:return identity,False,True
        mask=np.full(gray.shape,255,np.uint8)
        for person in people:
            x1,y1,x2,y2=person['bbox']
            cv2.rectangle(mask,(int(x1/w*320),int(y1/h*240)),(int(x2/w*320),int(y2/h*240)),0,-1)
        points=cv2.goodFeaturesToTrack(previous,150,.02,7,mask=mask)
        if points is None or len(points)<8:return identity,False,False
        follow,status,_=cv2.calcOpticalFlowPyrLK(previous,gray,points,None)
        if follow is None:return identity,False,False
        valid=status.ravel()==1
        if valid.sum()<8:return identity,False,False
        matrix,inliers=cv2.estimateAffinePartial2D(points[valid],follow[valid],method=cv2.RANSAC,ransacReprojThreshold=2)
        if matrix is None or inliers is None or inliers.mean()<.55:return identity,False,False
        scale=float(np.hypot(matrix[0,0],matrix[0,1]))
        if not .8<scale<1.25:return identity,False,True
        # Affine is estimated on a 320x240 image; undo anisotropic scaling.
        S=np.diag([w/320,h/240,1.]);M=np.vstack([matrix,[0,0,1]])
        return (S@M@np.linalg.inv(S))[:2],True,False

class InteractionFeatures:
    def __init__(self):
        self.people={};self.bikes={};self.camera=CameraMotion();self.last_timestamp=None

    def update(self,frame,detections,timestamp):
        people=[d for d in detections if d['class_id']==0]
        bikes=[d for d in detections if d['class_id']==3]
        matrix,motion_ok,cut=self.camera.update(frame,people,timestamp)
        if cut or (self.last_timestamp is not None and timestamp-self.last_timestamp>1.25):
            self.people.clear();self.bikes.clear()
        self.last_timestamp=timestamp
        bike_motion={}
        for bike in bikes:
            tid=bike.get('track_id');prior=self.bikes.get(tid) if tid is not None else None
            diag=max(1,float(np.linalg.norm(np.subtract(bike['bbox'][2:],bike['bbox'][:2]))))
            velocity=np.zeros(2);valid=False
            if prior and motion_ok and 0<timestamp-prior['t']<=1.25:
                velocity=(center(bike['bbox'])-transformed(prior['center'],matrix))/diag/(timestamp-prior['t']);valid=True
            if tid is not None:self.bikes[tid]=dict(center=center(bike['bbox']),t=timestamp)
            bike_motion[id(bike)]=(velocity,valid)
        output=[]
        for person in people:
            tid=person.get('track_id')
            if tid is None:continue  # never fuse unrelated untracked detections
            prior=self.people.get(tid)
            if prior and timestamp-prior['t']>1.25:prior=None
            bbox=np.array(person['bbox'],float);height=max(1,bbox[3]-bbox[1]);width=max(1,bbox[2]-bbox[0])
            pose=np.asarray(person.get('keypoints',np.zeros((17,3))),float)
            if pose.shape!=(17,3):pose=np.zeros((17,3))
            visible=(pose[:,2]>=.4)&np.isfinite(pose).all(1)
            anchor_valid=bool(visible[[5,6,11,12]].all())
            pelvis=pose[[11,12],:2].mean(0) if visible[[11,12]].all() else center(bbox)
            body_scale=max(height*.25,1)
            if visible[[5,6,11,12]].all():body_scale=max(1,np.linalg.norm(pose[[5,6],:2].mean(0)-pelvis))
            canonical=np.clip((pose[:,:2]-pelvis)/body_scale,-4,4);canonical[~visible]=0
            values={name:0.0 for name in FRAME_NAMES}
            for j in JOINTS:
                values[f'joint_{j}_x']=float(canonical[j,0]);values[f'joint_{j}_y']=float(canonical[j,1]);values[f'joint_{j}_valid']=float(visible[j])
            for side,(a,b,c) in [('left',(5,7,9)),('right',(6,8,10))]:
                if visible[[a,b,c]].all():
                    u=pose[a,:2]-pose[b,:2];v=pose[c,:2]-pose[b,:2]
                    values[side+'_elbow_cos']=float(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-6))
            if visible[[5,6,11,12]].all():
                shoulders=pose[[5,6],:2].mean(0)
                values['torso_lean']=float(abs(shoulders[0]-pelvis[0])/body_scale)
            if visible[0] and visible[[5,6]].all():
                offset=(pose[0,:2]-pose[[5,6],:2].mean(0))/body_scale
                values.update(head_offset_x=float(offset[0]),head_offset_y=float(offset[1]),head_valid=1.)
            bike=min(bikes,key=lambda b:box_distance(bbox,b['bbox']),default=None)
            diag=max(1,float(np.linalg.norm(np.subtract(bike['bbox'][2:],bike['bbox'][:2])))) if bike else 1
            near=bool(bike and box_distance(bbox,bike['bbox'])<=.15*diag)
            bike_key=bike.get('track_id') if bike else None
            # Stable pair identity is necessary to accumulate pair-specific time.
            stable_pair=bool(near and bike_key is not None and prior and prior['near'] and prior['bike']==bike_key)
            since=prior['near_since'] if stable_pair else timestamp
            contact=False
            for side,j in [('left',9),('right',10)]:
                values[side+'_wrist_valid']=float(visible[j])
                if bike and visible[j]:
                    values[side+'_wrist_bike_distance']=min(3,point_box_distance(pose[j,:2],bike['bbox'])/diag)
                    bb=bike['bbox'];upper=[bb[0],bb[1],bb[2],bb[1]+.55*(bb[3]-bb[1])]
                    hand_near=bool(near and point_box_distance(pose[j,:2],upper)/diag<.08)
                    values[side+'_wrist_upper_bike']=float(hand_near);contact|=hand_near
                if prior and anchor_valid and prior['anchor_valid'] and visible[j] and prior['visible'][j] and 0<timestamp-prior['t']<=1.25:
                    values[side+'_wrist_speed']=min(5,float(np.linalg.norm(canonical[j]-prior['canonical'][j])/(timestamp-prior['t'])))
                    values[side+'_wrist_motion_valid']=1.
            contact_since=prior['contact_since'] if stable_pair and prior['contact'] and contact else timestamp
            velocity=np.zeros(2);pv=False
            if prior and motion_ok and 0<timestamp-prior['t']<=1.25:
                velocity=(center(bbox)-transformed(prior['center'],matrix))/height/(timestamp-prior['t']);pv=True
                values['person_speed']=min(5,float(np.linalg.norm(velocity)))
                if prior['motion_valid'] and np.linalg.norm(velocity)>.03 and np.linalg.norm(prior['velocity'])>.03:
                    values['direction_change']=float(1-np.dot(velocity,prior['velocity'])/(np.linalg.norm(velocity)*np.linalg.norm(prior['velocity'])+1e-6))
            values['person_motion_valid']=float(pv)
            if bike:
                bv,valid=bike_motion[id(bike)];values['bike_motion_valid']=float(valid)
                values['bike_speed']=min(5,float(np.linalg.norm(bv))) if valid else 0.
                if valid and pv and np.linalg.norm(bv)>.03 and np.linalg.norm(velocity)>.03:
                    values['co_motion']=float(np.dot(velocity,bv)/(np.linalg.norm(velocity)*np.linalg.norm(bv)+1e-6))
            approaches=prior['approaches'] if prior else 0
            switches=prior['switches'] if prior else 0
            if near and (not prior or not prior['near']):approaches+=1
            if near and prior and prior['bike'] is not None and bike_key is not None and prior['bike']!=bike_key:switches+=1
            born=prior['born'] if prior else timestamp
            values.update(near_bike=float(near),bike_visible=float(bike is not None),
                near_seconds=min(4,timestamp-since)/4 if near and bike_key is not None else 0.,
                hand_region_seconds=min(4,timestamp-contact_since)/4 if contact and bike_key is not None else 0.,
                approach_count=min(approaches,5)/5,bike_switch_count=min(switches,5)/5,
                track_seconds=min(4,timestamp-born)/4,pose_coverage=float(visible[JOINTS].mean()),body_aspect=min(4,width/height))
            vector=np.array([values[k] for k in FRAME_NAMES],np.float32)
            history=prior['history'] if prior else deque()
            history.append((timestamp,vector))
            while history and timestamp-history[0][0]>4:history.popleft()
            self.people[tid]=dict(t=timestamp,born=born,near=near,bike=bike_key,near_since=since,
                contact=contact,contact_since=contact_since,visible=visible,canonical=canonical,
                anchor_valid=anchor_valid,
                center=center(bbox),velocity=velocity,motion_valid=pv,approaches=approaches,switches=switches,history=history)
            samples=np.stack([x[1] for x in history])
            descriptor=np.r_[samples.mean(0),samples.std(0),samples[-1]].astype(np.float32)
            output.append(dict(track_id=tid,bbox=person['bbox'],timestamp=float(timestamp),
                observed_seconds=float(timestamp-history[0][0]),observations=len(history),
                pose_coverage=values['pose_coverage'],near_bike=near,
                wrist_near_bike=contact,torso_lean=values['torso_lean'],
                hand_motion=max(values['left_wrist_speed'],values['right_wrist_speed']),
                bike_speed=values['bike_speed'],camera_motion_valid=motion_ok,
                keypoints=person.get('keypoints'),feature=descriptor))
        self.people={k:v for k,v in self.people.items() if timestamp-v['t']<=1.25}
        self.bikes={k:v for k,v in self.bikes.items() if timestamp-v['t']<=1.25}
        return output,cut

def draw_pose(frame,observations):
    edges=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for row in observations:
        pose=np.asarray(row.get('keypoints'))
        if pose.shape!=(17,3):continue
        for a,b in edges:
            if min(pose[a,2],pose[b,2])>=.4:
                cv2.line(frame,tuple(pose[a,:2].astype(int)),tuple(pose[b,:2].astype(int)),(50,210,255),2)
        for j in JOINTS:
            if pose[j,2]>=.4:cv2.circle(frame,tuple(pose[j,:2].astype(int)),3,(20,180,255),-1)
    return frame
