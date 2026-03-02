from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    db_path: str = os.getenv("DB_PATH", "coach_sessions.sqlite3")
    dialogpt_model: str = os.getenv("DIALOGPT_MODEL", "microsoft/DialoGPT-medium")
    voice_enabled: bool = _get_bool("VOICE_ENABLED", False)


settings = Settings()
