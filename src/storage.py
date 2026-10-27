from pathlib import Path
import sqlite3
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
STORAGE_API_VERSION=2

def connect(path=None):
    path=Path(path or ROOT/'runtime/events.sqlite3');path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, created TEXT, camera TEXT, video TEXT, score REAL, status TEXT, note TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, event_id INTEGER, created TEXT, actor TEXT, action TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY, event_id INTEGER NOT NULL, run_id TEXT, video_time REAL, source_frame INTEGER, original_path TEXT, annotated_path TEXT, metadata_path TEXT, track_ids TEXT, FOREIGN KEY(event_id) REFERENCES events(id))')
    for table in ['events','evidence']:
        if 'risk_level' not in {row[1] for row in db.execute(f'PRAGMA table_info({table})')}:
            db.execute(f"ALTER TABLE {table} ADD COLUMN risk_level TEXT DEFAULT 'unknown'")
    db.commit();return db

def create_event(camera,video,score,path=None):
    with connect(path) as db:
        cur=db.execute('INSERT INTO events(created,camera,video,score,status,note) VALUES(?,?,?,?,?,?)',
                       (datetime.now(timezone.utc).isoformat(),camera,video,score,'pending',''))
        return cur.lastrowid

def update_event(event_id,status,note,actor,path=None):
    if status not in ['pending','confirmed','false_alarm']:
        raise ValueError('Trạng thái không hợp lệ')
    with connect(path) as db:
        cur=db.execute('UPDATE events SET status=?,note=? WHERE id=?',(status,note,event_id))
        if cur.rowcount!=1:raise ValueError('Sự kiện không tồn tại')
        db.execute('INSERT INTO audit(event_id,created,actor,action) VALUES(?,?,?,?)',
                   (event_id,datetime.now(timezone.utc).isoformat(),actor,status))

def register_evidence(record,path=None):
    import json
    with connect(path) as db:
        cur=db.execute('INSERT INTO events(created,camera,video,score,status,note,risk_level) VALUES(?,?,?,?,?,?,?)',
            (record['captured_at_utc'],record['camera_id'],record['source_video'],record['alarm_score'],
             'pending','Ảnh tự chụp khi có dấu hiệu cần xem xét; chưa xác nhận trộm.',record.get('risk_level_at_capture','unknown')))
        event_id=cur.lastrowid
        db.execute('INSERT INTO evidence(id,event_id,run_id,video_time,source_frame,original_path,annotated_path,metadata_path,track_ids,risk_level) VALUES(?,?,?,?,?,?,?,?,?,?)',
            (record['id'],event_id,record['run_id'],record['video_time_seconds'],record['source_frame_index'],
             record['original_path'],record['annotated_path'],record['metadata_path'],json.dumps(record['track_ids']),record.get('risk_level_at_capture','unknown')))
        return event_id

def list_evidence(event_id,path=None):
    with connect(path) as db:
        db.row_factory=sqlite3.Row
        return [dict(r) for r in db.execute('SELECT * FROM evidence WHERE event_id=? ORDER BY video_time',(int(event_id),))]
