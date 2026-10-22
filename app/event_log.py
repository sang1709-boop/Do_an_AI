"""Luu phien giam sat/canh bao vao SQLite, khong ghi hinh anh khuon mat."""
import csv
import logging
from logging.handlers import RotatingFileHandler
import sqlite3
from datetime import datetime
from pathlib import Path
from app.paths import DATA, REPORTS

DB_PATH = DATA / 'logs.db'

def now():
    return datetime.now().astimezone().isoformat(timespec='milliseconds')

def setup_logging():
    logger = logging.getLogger('ai_driver')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(REPORTS / 'app.log', maxBytes=2_000_000,
                                      backupCount=3, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    return logger

class EventStore:
    def __init__(self, path=DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                camera_source TEXT NOT NULL,
                alert_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                started_at TEXT NOT NULL,
                ended_at TEXT,
                kind TEXT NOT NULL,
                max_probability REAL,
                ear REAL,
                mar REAL,
                threshold REAL,
                lighting TEXT,
                duration_seconds REAL
            );
            CREATE INDEX IF NOT EXISTS ix_alerts_time ON alerts(started_at);
            ''')
        # Dong nhung ban ghi dang do khi app bi tat dot ngot lan truoc.
        with self.connect() as db:
            t = now()
            db.execute('UPDATE alerts SET ended_at=? WHERE ended_at IS NULL', (t,))
            db.execute('UPDATE sessions SET ended_at=? WHERE ended_at IS NULL', (t,))

    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def start_session(self, camera):
        with self.connect() as db:
            cur = db.execute('INSERT INTO sessions(started_at,camera_source) VALUES (?,?)',
                             (now(), str(camera)))
            return cur.lastrowid

    def end_session(self, session_id, count):
        with self.connect() as db:
            db.execute('UPDATE sessions SET ended_at=?,alert_count=? WHERE id=?',
                       (now(), count, session_id))

    def start_alert(self, session_id, kind, probability, d):
        with self.connect() as db:
            db.execute('''INSERT INTO alerts
                (session_id,started_at,kind,max_probability,ear,mar,threshold,lighting)
                VALUES (?,?,?,?,?,?,?,?)''',
                (session_id, now(), kind, probability, d.get('ear'), d.get('mar'),
                 d.get('threshold_long') if kind == 'Mắt mở bất thường' else d.get('threshold'),
                 d.get('lighting')))

    def update_alert_peak(self, session_id, probability, d):
        if probability is None:
            return
        with self.connect() as db:
            db.execute('''UPDATE alerts SET max_probability=?,ear=?,mar=?
                WHERE id=(SELECT id FROM alerts WHERE session_id=? AND ended_at IS NULL
                ORDER BY id DESC LIMIT 1) AND (max_probability IS NULL OR max_probability < ?)''',
                (probability, d.get('ear'), d.get('mar'), session_id, probability))

    def end_alert(self, session_id):
        end = now()
        with self.connect() as db:
            row = db.execute('''SELECT id,started_at FROM alerts
                WHERE session_id=? AND ended_at IS NULL ORDER BY id DESC LIMIT 1''',
                (session_id,)).fetchone()
            if row:
                elapsed = max(0, (datetime.fromisoformat(end) -
                                  datetime.fromisoformat(row['started_at'])).total_seconds())
                db.execute('UPDATE alerts SET ended_at=?,duration_seconds=? WHERE id=?',
                           (end, round(elapsed, 3), row['id']))

    def count_alerts(self):
        with self.connect() as db:
            return db.execute('SELECT COUNT(*) FROM alerts').fetchone()[0]

    def list_alerts(self, limit=None):
        query = 'SELECT * FROM alerts ORDER BY id DESC'
        args = ()
        if limit is not None:
            query += ' LIMIT ?'
            args = (limit,)
        with self.connect() as db:
            return [dict(row) for row in db.execute(query, args)]

    def export_csv(self, path):
        columns = ('id','session_id','started_at','ended_at','kind',
                   'duration_seconds','max_probability','ear','mar','threshold','lighting')
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in self.list_alerts():
                writer.writerow({key: row[key] for key in columns})
