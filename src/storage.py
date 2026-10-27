from pathlib import Path
import sqlite3
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]

def connect(path=None):
    path=Path(path or ROOT/'runtime/events.sqlite3');path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, created TEXT, camera TEXT, video TEXT, score REAL, status TEXT, note TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, event_id INTEGER, created TEXT, actor TEXT, action TEXT)')
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
