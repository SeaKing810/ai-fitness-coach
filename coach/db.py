from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict, List, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_ts INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL,
  ts INTEGER NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS profiles (
  session_id INTEGER PRIMARY KEY,
  age INTEGER,
  experience TEXT,
  goals TEXT,
  constraints TEXT,
  updated_ts INTEGER NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS progress (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL,
  ts INTEGER NOT NULL,
  metric TEXT NOT NULL,
  value REAL NOT NULL,
  note TEXT,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_progress_session ON progress(session_id);
"""


class Database:
    def __init__(self, path: str) -> None:
        self.path = path
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def create_session(self) -> int:
        now = int(time.time())
        with self._connect() as conn:
            cur = conn.execute("INSERT INTO sessions(created_ts) VALUES(?)", (now,))
            conn.commit()
            return int(cur.lastrowid)

    def add_message(self, session_id: int, role: str, content: str) -> None:
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, ts, role, content) VALUES(?, ?, ?, ?)",
                (session_id, now, role, content),
            )
            conn.commit()

    def get_messages(self, session_id: int, limit: int = 30) -> List[Dict[str, Any]]:
        limit = max(1, min(200, limit))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        out = [dict(r) for r in rows]
        out.reverse()
        return out

    def upsert_profile(
        self,
        session_id: int,
        age: Optional[int],
        experience: str,
        goals: str,
        constraints: str,
    ) -> None:
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO profiles(session_id, age, experience, goals, constraints, updated_ts)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  age=excluded.age,
                  experience=excluded.experience,
                  goals=excluded.goals,
                  constraints=excluded.constraints,
                  updated_ts=excluded.updated_ts
                """,
                (session_id, age, experience, goals, constraints, now),
            )
            conn.commit()

    def get_profile(self, session_id: int) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, age, experience, goals, constraints, updated_ts FROM profiles WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else {}

    def add_progress(self, session_id: int, metric: str, value: float, note: str = "") -> None:
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO progress(session_id, ts, metric, value, note) VALUES(?, ?, ?, ?, ?)",
                (session_id, now, metric, float(value), note),
            )
            conn.commit()

    def get_progress(self, session_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, metric, value, note
                FROM progress
                WHERE session_id = ?
                ORDER BY ts ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]
