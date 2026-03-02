from __future__ import annotations

import re
from typing import Dict


def sanitize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s[:2000]


def detect_intent(user_text: str) -> Dict[str, bool]:
    t = (user_text or "").lower()
    want_form = any(k in t for k in ["check my form", "form check", "fix my squat", "analyze my video", "rate my form"])
    want_plan = any(k in t for k in ["plan", "program", "routine", "workout split", "weekly"])
    want_recs = any(k in t for k in ["recommend", "suggest", "exercise", "what should i do"])
    return {"form_check": want_form, "plan": want_plan, "recommend": want_recs}
