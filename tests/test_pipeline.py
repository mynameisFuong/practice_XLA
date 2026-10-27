from pathlib import Path
import cv2
import numpy as np
import pytest
from src.features import preprocess,describe_window,extract_video,video_info
from src.events import make_events
from src.storage import create_event,update_event,connect

def test_letterbox_aspect_and_mask():
    image=np.full((100,200,3),100,np.uint8)
    bgr,gray,mask,q=preprocess(image)
    assert bgr.shape==(96,128,3)
    assert np.count_nonzero(mask)==128*64
    assert q['brightness']==100

def test_descriptor_and_short_corrupt_video(tmp_path):
    p=tmp_path/'short.avi'
    writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),8,(80,60))
    for i in range(8):
        frame=np.zeros((60,80,3),np.uint8);cv2.rectangle(frame,(i*3,15),(i*3+20,40),(180,180,180),-1);writer.write(frame)
    writer.release()
    x,records,info=extract_video(p)
    assert x.shape==(1,328) and np.isfinite(x).all()
    assert records[0]['end']==pytest.approx(1)
    broken=tmp_path/'broken.mp4';broken.write_text('broken')
    with pytest.raises(ValueError):video_info(broken)

def test_darkness_and_single_frame_do_not_count_as_usable():
    _,q=describe_window([np.zeros((96,128,3),np.uint8)])
    assert not q['usable']

def test_no_bridge_across_bad_or_missing_windows():
    def row(start,score=.9,usable=True):return dict(start=start,end=start+4,score=score,usable=usable)
    assert len(make_events([row(0),row(4)],.7))==1
    assert not make_events([row(0),row(4,usable=False),row(8)],.7)
    assert not make_events([row(0),row(40)],.7)

def test_storage_audit(tmp_path):
    p=tmp_path/'db.sqlite3';eid=create_event('B1','test.mp4',.8,p)
    update_event(eid,'false_alarm','Chủ xe lấy xe','Bảo vệ',p)
    with connect(p) as db:
        assert db.execute('SELECT status FROM events').fetchone()[0]=='false_alarm'
        assert db.execute('SELECT COUNT(*) FROM audit').fetchone()[0]==1
    with pytest.raises(ValueError):update_event(eid,'invalid','','',p)
