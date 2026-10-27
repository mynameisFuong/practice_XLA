import numpy as np
from src.small_objects import detection_regions,merge_candidates

def test_regions_cover_frame_and_offsets_preserve_source_coordinates():
    frame=np.arange(100*200*3,dtype=np.uint8).reshape(100,200,3)
    regions=list(detection_regions(frame))
    assert len(regions)==5 and regions[0][3]==1280
    cover=np.zeros(frame.shape[:2],bool)
    for crop,x,y,size in regions[1:]:
        np.testing.assert_array_equal(crop,frame[y:y+crop.shape[0],x:x+crop.shape[1]])
        cover[y:y+crop.shape[0],x:x+crop.shape[1]]=True
        assert size==640
    assert cover.all()

def test_merging_removes_duplicate_tiles_but_retains_person_and_bike():
    candidates=[[10,10,50,80,.8,0],[11,11,51,81,.6,0],
                [10,10,50,80,.7,3],[-5,5,15,90,.4,0],
                [20,20,10,10,.9,0],[0,0,20,20,.1,0],
                [0,0,float('nan'),10,.8,0]]
    merged=merge_candidates(candidates,(100,100,3))
    assert merged.shape==(3,6)
    assert sum(merged[:,5]==0)==2 and sum(merged[:,5]==3)==1
    assert merged[:,0].min()==0
    assert merge_candidates([],(100,100,3)).shape==(0,6)
