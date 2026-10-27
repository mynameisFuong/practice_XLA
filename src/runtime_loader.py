"""Recover a stale Streamlit module after an interface update.

Streamlit may rerun app.py while src.live from an earlier run remains imported.
Reload only when the cached callable cannot accept the current API; don't mask
the mismatch by silently disabling requested behavior features.
"""
import importlib
import inspect

REQUIRED_STREAM_ARGUMENTS={'behavior','use_behavior_alarm','camera_id','capture_evidence','risk_backend','detector_weights',
                           'preprocessing','denoise','sharpen','detector_profile'}

def get_stream_video():
    module=importlib.import_module('src.live')
    if (getattr(module,'STREAM_API_VERSION',0)<5
            or not REQUIRED_STREAM_ARGUMENTS.issubset(inspect.signature(module.stream_video).parameters)):
        importlib.invalidate_caches()
        module=importlib.reload(module)
    if not REQUIRED_STREAM_ARGUMENTS.issubset(inspect.signature(module.stream_video).parameters):
        raise RuntimeError(
            'app.py và src/live.py chưa cùng phiên bản. Hãy dùng đầy đủ mã nguồn mới, '
            'dừng Streamlit bằng Ctrl+C rồi chạy lại bằng .venv/bin/python -m streamlit run app.py.'
        )
    return module.stream_video
