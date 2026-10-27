import inspect
import pytest
from src import live
from src.runtime_loader import get_stream_video

def test_stale_streamlit_callable_recovers_from_current_source(monkeypatch):
    def previous_version(path,output_path,pace=False):
        raise AssertionError('The stale implementation must not run')
    monkeypatch.setattr(live,'stream_video',previous_version)
    recovered=get_stream_video()
    assert recovered is not previous_version
    inspect.signature(recovered).bind('input.mp4','output.mp4',behavior=True,use_behavior_alarm=True)

def test_current_api_does_not_reload_active_module(monkeypatch):
    from src import runtime_loader
    def unexpected_reload(module):raise AssertionError('Current module must remain loaded')
    monkeypatch.setattr(runtime_loader.importlib,'reload',unexpected_reload)
    assert get_stream_video() is live.stream_video

def test_behavior_version_without_evidence_arguments_is_refreshed(monkeypatch):
    def old_behavior_api(path,output_path,pace=False,behavior=False,use_behavior_alarm=False):pass
    monkeypatch.setattr(live,'stream_video',old_behavior_api)
    recovered=get_stream_video()
    inspect.signature(recovered).bind('input.mp4','output.mp4',behavior=True,camera_id='B1',capture_evidence=True)

def test_incomplete_source_shows_actionable_error(monkeypatch):
    from src import runtime_loader
    monkeypatch.setattr(live,'stream_video',lambda path,output_path:None)
    monkeypatch.setattr(runtime_loader.importlib,'reload',lambda module:module)
    with pytest.raises(RuntimeError,match='app.py và src/live.py chưa cùng phiên bản'):
        get_stream_video()

def test_previous_api_without_detector_selection_is_refreshed(monkeypatch):
    def old(path,output_path,behavior=False,use_behavior_alarm=False,camera_id=None,
            capture_evidence=True,risk_backend=None):pass
    monkeypatch.setattr(live,'stream_video',old)
    recovered=get_stream_video()
    inspect.signature(recovered).bind('input.mp4','output.mp4',detector_weights='model/yolo_behavior_model.pt')
