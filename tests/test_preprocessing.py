import cv2
import numpy as np
import pytest
from src.preprocessing import enhance_frame

def test_off_preserves_source_and_dark_auto_is_non_mutating():
    dark=np.tile(np.arange(10,50,dtype=np.uint8),(64,2))
    source=cv2.cvtColor(dark,cv2.COLOR_GRAY2BGR);backup=source.copy()
    unchanged,info=enhance_frame(source)
    np.testing.assert_array_equal(source,unchanged)
    assert info['applied']==[]
    enhanced,info=enhance_frame(source,'auto',True,True)
    np.testing.assert_array_equal(source,backup)
    assert enhanced.shape==source.shape and enhanced.dtype==np.uint8
    assert enhanced.mean()>source.mean()
    assert 'gamma_0.75' in info['applied'] and 'bilateral' in info['applied']

def test_auto_leaves_well_lit_contrasty_image_unchanged():
    frame=np.tile(np.arange(256,dtype=np.uint8),(64,1))
    frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    out,info=enhance_frame(frame,'auto')
    np.testing.assert_array_equal(frame,out)
    assert info['applied']==[]
    with pytest.raises(ValueError):enhance_frame(frame,'unknown')
