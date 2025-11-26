import sqlite3
from pathlib import Path
from .config import RESULTS_DB

def get_conn():
    conn = sqlite3.connect(RESULTS_DB)
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            method TEXT,
            energy REAL,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_run(method: str, energy: float, notes: str = ""):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO runs (timestamp, method, energy, notes)
        VALUES (datetime('now'), ?, ?, ?)
    """, (method, energy, notes))
    conn.commit()
    conn.close()
