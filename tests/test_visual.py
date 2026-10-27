import numpy as np
import pytest
from src.visual_backbone import person_crop

def test_person_context_crop_preserves_pixels_and_clips_boundaries():
    frame=np.arange(100*200*3,dtype=np.uint8).reshape(100,200,3)
    crop=person_crop(frame,[0,0,100,50],padding=.2)
    np.testing.assert_array_equal(crop,frame[:60,:120])
    with pytest.raises(ValueError):person_crop(frame,[50,50,20,20])
    with pytest.raises(ValueError):person_crop(frame,[float('nan'),0,20,20])
